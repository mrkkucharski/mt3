# Copyright 2026 The MT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions that operate on NoteSequence protos."""

import dataclasses
import itertools

from typing import List, MutableMapping, MutableSet, Optional, Sequence, Tuple

from absl import logging
from mt3 import event_codec
from mt3 import general_midi
from mt3 import run_length_encoding
from mt3 import vocabularies

import note_seq

DEFAULT_VELOCITY = 100
DEFAULT_NOTE_DURATION = 0.01

# Quantization can result in zero-length notes; enforce a minimum duration.
MIN_NOTE_DURATION = 0.01

# A note this long, force-closed at end-of-decode, is likely a dropped
# note-off rather than a real sustained note -- flag it rather than silently
# emitting an implausible note. Lookback boundary reconciliation greatly
# reduces this failure mode, but a model can still consistently declare the
# same note active in every later window.
LONG_HELD_NOTE_WARNING_SECONDS = 30.0


@dataclasses.dataclass
class TrackSpec:
  name: str
  program: int = 0
  is_drum: bool = False
  # None: don't filter by rhythm (matches every note regardless of role).
  rhythm: Optional[bool] = None


def extract_track(ns, program, is_drum, rhythm: Optional[bool] = None,
                  rhythm_map_fn=lambda rhythm: rhythm):
  """Extract notes matching `program`/`is_drum`, and `rhythm` if given.

  `rhythm_map_fn` lets callers apply the same granularity collapse (see
  `vocabularies.PROGRAM_GRANULARITIES`) to each note's rhythm flag before
  comparing it against `rhythm`, so filtering stays consistent with grouping
  done under that granularity.
  """
  rhythm_by_instrument = instrument_rhythms(ns) if rhythm is not None else None
  track = note_seq.NoteSequence(ticks_per_quarter=220)
  track_notes = [
      note for note in ns.notes
      if note.program == program and note.is_drum == is_drum
      and (rhythm_by_instrument is None or
           rhythm_map_fn(rhythm_by_instrument.get(note.instrument, False))
           == rhythm)
  ]
  track.notes.extend(track_notes)
  track.total_time = (max(note.end_time for note in track.notes)
                      if track.notes else 0.0)
  return track


def trim_overlapping_notes(ns: note_seq.NoteSequence) -> note_seq.NoteSequence:
  """Trim overlapping notes from a NoteSequence, dropping zero-length notes."""
  ns_trimmed = note_seq.NoteSequence()
  ns_trimmed.CopyFrom(ns)
  channels = set((note.pitch, note.program, note.is_drum)
                 for note in ns_trimmed.notes)
  for pitch, program, is_drum in channels:
    notes = [note for note in ns_trimmed.notes if note.pitch == pitch
             and note.program == program and note.is_drum == is_drum]
    sorted_notes = sorted(notes, key=lambda note: note.start_time)
    for i in range(1, len(sorted_notes)):
      if sorted_notes[i - 1].end_time > sorted_notes[i].start_time:
        sorted_notes[i - 1].end_time = sorted_notes[i].start_time
  valid_notes = [note for note in ns_trimmed.notes
                 if note.start_time < note.end_time]
  del ns_trimmed.notes[:]
  ns_trimmed.notes.extend(valid_notes)
  return ns_trimmed


def assign_instruments(
    ns: note_seq.NoteSequence,
    note_rhythms: Optional[Sequence[bool]] = None,
) -> None:
  """Assign instrument numbers to notes; modifies NoteSequence in place.

  Notes are grouped into instruments by `(program, rhythm)` pair, where
  `rhythm` defaults to False for every note when `note_rhythms` is omitted —
  degenerating to the original program-only grouping. `rhythm` has no field
  on the `Note` proto, so this is the carrier: `ns.instrument_infos` is
  rebuilt with the canonical `<slug>[:rhythm]` name for each assigned
  instrument, which is what a written MIDI file will use as its track name.

  Args:
    ns: NoteSequence to modify in place.
    note_rhythms: optional, one bool per note in `ns.notes`, in order.
  """
  if note_rhythms is not None and len(note_rhythms) != len(ns.notes):
    raise ValueError(
        'note_rhythms must have exactly one entry per note in ns.notes: '
        f'got {len(note_rhythms)} for {len(ns.notes)} notes')

  del ns.instrument_infos[:]
  key_instruments: MutableMapping[Tuple[int, bool], int] = {}
  drums_named = False
  for i, note in enumerate(ns.notes):
    if note.is_drum:
      note.instrument = 9
      if not drums_named:
        ns.instrument_infos.add(
            instrument=9,
            name=general_midi.instrument_name(
                program=None, is_drum=True, rhythm=False))
        drums_named = True
      continue
    rhythm = bool(note_rhythms[i]) if note_rhythms is not None else False
    key = (note.program, rhythm)
    if key not in key_instruments:
      num_instruments = len(key_instruments)
      instrument = (num_instruments if num_instruments < 9
                   else num_instruments + 1)
      key_instruments[key] = instrument
      ns.instrument_infos.add(
          instrument=instrument,
          name=general_midi.instrument_name(
              program=note.program, is_drum=False, rhythm=rhythm))
    note.instrument = key_instruments[key]


def validate_note_sequence(ns: note_seq.NoteSequence) -> None:
  """Raise ValueError if NoteSequence contains invalid notes."""
  for note in ns.notes:
    if note.start_time >= note.end_time:
      raise ValueError('note has start time >= end time: %f >= %f' %
                       (note.start_time, note.end_time))
    if note.velocity == 0:
      raise ValueError('note has zero velocity')


def note_arrays_to_note_sequence(
    onset_times: Sequence[float],
    pitches: Sequence[int],
    offset_times: Optional[Sequence[float]] = None,
    velocities: Optional[Sequence[int]] = None,
    programs: Optional[Sequence[int]] = None,
    is_drums: Optional[Sequence[bool]] = None,
    rhythms: Optional[Sequence[bool]] = None,
) -> note_seq.NoteSequence:
  """Convert note onset / offset / pitch / velocity arrays to NoteSequence."""
  ns = note_seq.NoteSequence(ticks_per_quarter=220)
  note_rhythms = []
  for onset_time, offset_time, pitch, velocity, program, is_drum, rhythm in itertools.zip_longest(
      onset_times, [] if offset_times is None else offset_times,
      pitches, [] if velocities is None else velocities,
      [] if programs is None else programs,
      [] if is_drums is None else is_drums,
      [] if rhythms is None else rhythms):
    if offset_time is None:
      offset_time = onset_time + DEFAULT_NOTE_DURATION
    if velocity is None:
      velocity = DEFAULT_VELOCITY
    if program is None:
      program = 0
    if is_drum is None:
      is_drum = False
    ns.notes.add(
        start_time=onset_time,
        end_time=offset_time,
        pitch=pitch,
        velocity=velocity,
        program=program,
        is_drum=is_drum)
    ns.total_time = max(ns.total_time, offset_time)
    note_rhythms.append(bool(rhythm))
  assign_instruments(ns, note_rhythms=note_rhythms)
  return ns


@dataclasses.dataclass
class NoteEventData:
  pitch: int
  velocity: Optional[int] = None
  program: Optional[int] = None
  is_drum: Optional[bool] = None
  instrument: Optional[int] = None
  # chordal-accompaniment flag; guitar-only, absent (None/False) elsewhere.
  rhythm: Optional[bool] = None


def note_sequence_to_onsets(
    ns: note_seq.NoteSequence
) -> Tuple[Sequence[float], Sequence[NoteEventData]]:
  """Extract note onsets and pitches from NoteSequence proto."""
  # Sort by pitch to use as a tiebreaker for subsequent stable sort.
  notes = sorted(ns.notes, key=lambda note: note.pitch)
  return ([note.start_time for note in notes],
          [NoteEventData(pitch=note.pitch) for note in notes])


def note_sequence_to_onsets_and_offsets(
    ns: note_seq.NoteSequence,
) -> Tuple[Sequence[float], Sequence[NoteEventData]]:
  """Extract onset & offset times and pitches from a NoteSequence proto.

  The onset & offset times will not necessarily be in sorted order.

  Args:
    ns: NoteSequence from which to extract onsets and offsets.

  Returns:
    times: A list of note onset and offset times.
    values: A list of NoteEventData objects where velocity is zero for note
        offsets.
  """
  # Sort by pitch and put offsets before onsets as a tiebreaker for subsequent
  # stable sort.
  notes = sorted(ns.notes, key=lambda note: note.pitch)
  times = ([note.end_time for note in notes] +
           [note.start_time for note in notes])
  values = ([NoteEventData(pitch=note.pitch, velocity=0) for note in notes] +
            [NoteEventData(pitch=note.pitch, velocity=note.velocity)
             for note in notes])
  return times, values


def instrument_rhythms(
    ns: note_seq.NoteSequence) -> MutableMapping[int, bool]:
  """Map instrument index to its rhythm flag, read from `instrument_infos`.

  `rhythm` has no field on the `Note` proto (see `assign_instruments`), so
  this is how it is recovered: an instrument's canonical name carries a
  `:rhythm` suffix exactly when its notes are chordal accompaniment.
  """
  return {info.instrument: info.name.strip().endswith(':rhythm')
          for info in ns.instrument_infos}


def note_sequence_to_onsets_and_offsets_and_programs(
    ns: note_seq.NoteSequence,
) -> Tuple[Sequence[float], Sequence[NoteEventData]]:
  """Extract onset & offset times and pitches & programs from a NoteSequence.

  The onset & offset times will not necessarily be in sorted order.

  Args:
    ns: NoteSequence from which to extract onsets and offsets.

  Returns:
    times: A list of note onset and offset times.
    values: A list of NoteEventData objects where velocity is zero for note
        offsets.
  """
  rhythm_by_instrument = instrument_rhythms(ns)
  def rhythm(note) -> bool:
    return rhythm_by_instrument.get(note.instrument, False)

  # Sort by program, rhythm and pitch (and put offsets before onsets as a
  # tiebreaker) so the token stream groups by state and does not thrash
  # between flag values.
  notes = sorted(ns.notes,
                 key=lambda note: (note.is_drum, note.program, rhythm(note),
                                   note.pitch))
  times = ([note.end_time for note in notes if not note.is_drum] +
           [note.start_time for note in notes])
  values = ([NoteEventData(pitch=note.pitch, velocity=0,
                           program=note.program, is_drum=False,
                           rhythm=rhythm(note))
             for note in notes if not note.is_drum] +
            [NoteEventData(pitch=note.pitch, velocity=note.velocity,
                           program=note.program, is_drum=note.is_drum,
                           rhythm=False if note.is_drum else rhythm(note))
             for note in notes])
  return times, values


@dataclasses.dataclass
class NoteEncodingState:
  """Encoding state for note transcription, keeping track of active pitches."""
  # velocity bin for active (pitch, program, rhythm)
  active_pitches: MutableMapping[Tuple[int, int, bool], int] = (
      dataclasses.field(default_factory=dict))


def note_event_data_to_events(
    state: Optional[NoteEncodingState],
    value: NoteEventData,
    codec: event_codec.Codec,
) -> Sequence[event_codec.Event]:
  """Convert note event data to a sequence of events."""
  if value.velocity is None:
    # onsets only, no program or velocity
    return [event_codec.Event('pitch', value.pitch)]
  else:
    num_velocity_bins = vocabularies.num_velocity_bins_from_codec(codec)
    velocity_bin = vocabularies.velocity_to_bin(
        value.velocity, num_velocity_bins)
    if value.program is None:
      # onsets + offsets + velocities only, no programs
      if state is not None:
        state.active_pitches[(value.pitch, 0, False)] = velocity_bin
      return [event_codec.Event('velocity', velocity_bin),
              event_codec.Event('pitch', value.pitch)]
    else:
      if value.is_drum:
        # drum events use a separate vocabulary
        return [event_codec.Event('velocity', velocity_bin),
                event_codec.Event('drum', value.pitch)]
      else:
        # program + rhythm + velocity + pitch
        rhythm = bool(value.rhythm)
        if state is not None:
          state.active_pitches[
              (value.pitch, int(value.program), rhythm)] = velocity_bin
        return [event_codec.Event('program', value.program),
                event_codec.Event('rhythm', int(rhythm)),
                event_codec.Event('velocity', velocity_bin),
                event_codec.Event('pitch', value.pitch)]


def note_encoding_state_to_events(
    state: NoteEncodingState
) -> Sequence[event_codec.Event]:
  """Output program, rhythm and pitch events for active notes, then a tie event."""
  events = []
  for pitch, program, rhythm in sorted(
      state.active_pitches.keys(), key=lambda k: (k[1], k[2], k[0])):
    if state.active_pitches[(pitch, program, rhythm)]:
      events += [event_codec.Event('program', program),
                 event_codec.Event('rhythm', int(rhythm)),
                 event_codec.Event('pitch', pitch)]
  events.append(event_codec.Event('tie', 0))
  return events


@dataclasses.dataclass
class NoteDecodingState:
  """Decoding state for note transcription."""
  current_time: float = 0.0
  # velocity to apply to subsequent pitch events (zero for note-off)
  current_velocity: int = DEFAULT_VELOCITY
  # program to apply to subsequent pitch events
  current_program: int = 0
  # rhythm flag to apply to subsequent pitch events
  current_rhythm: bool = False
  # when set, every decoded note is stamped with this program; incoming
  # 'program' events are consumed (to keep shift/timing alignment intact)
  # but otherwise ignored, rather than updating current_program. Used for
  # single-instrument transcription where the model's own program guesses
  # are unreliable/irrelevant; as a side effect it also merges note
  # fragments the model would otherwise have split across differing (wrong)
  # program guesses, since active notes are keyed by
  # (pitch, current_program, current_rhythm). Implies no_rhythm: a single
  # forced instrument has no meaningful rhythm/lead distinction, and leaving
  # rhythm free to vary would still fragment the output into two
  # (program, rhythm) instrument groups.
  force_program: Optional[int] = None
  # the same idea for the rhythm/lead role alone: current_rhythm is pinned to
  # False and 'rhythm' events are consumed but ignored, so all notes collapse
  # into the non-rhythm group of whatever program they were decoded under.
  # Used when instrument identity is still worth decoding but the rhythm/lead
  # split is not.
  no_rhythm: bool = False

  def __post_init__(self):
    if self.force_program is not None:
      self.current_program = self.force_program
      self.no_rhythm = True
  # onset time and velocity for active pitches, programs and rhythm flags
  active_pitches: MutableMapping[Tuple[int, int, bool],
                                 Tuple[float, int]] = dataclasses.field(
                                     default_factory=dict)
  # pitches (with programs and rhythm flags) to continue from previous segment
  tied_pitches: MutableSet[Tuple[int, int, bool]] = dataclasses.field(
      default_factory=set)
  # whether or not we are in the tie section at the beginning of a segment
  is_tie_section: bool = False
  # set externally, per event, by run_length_encoding.decode_events when it
  # is given a min_time (lookback): true for an event that falls before the
  # window's kept region and must not affect note_sequence/active_pitches,
  # even though it still runs through decode_note_event so sticky state
  # (current_program/current_velocity/current_rhythm) stays correct for the
  # first kept event. Always false for a decode with no min_time.
  suppress: bool = False
  # While decoding a lookback prefix, note events are applied to this
  # segment-local shadow rather than to the committed state above.  The
  # shadow starts with the committed active pitches, is advanced through the
  # new window's tie section and lookback events, and is reconciled at the
  # kept-region boundary.  This is necessary because independently decoded
  # overlapping windows need not agree: simply discarding every pitch event
  # in the prefix also discards the only evidence that a carried note is no
  # longer active, allowing it to remain open until the end of the file.
  prefix_state: Optional['NoteDecodingState'] = None
  # A shadow tie section may legitimately mention a pitch that the preceding
  # (authoritative) kept region did not predict.  Such a pitch is ignored
  # instead of counted invalid; prefix-only notes are never added to the
  # committed output during reconciliation.
  is_prefix_shadow: bool = False
  # partially-decoded NoteSequence
  note_sequence: note_seq.NoteSequence = dataclasses.field(
      default_factory=lambda: note_seq.NoteSequence(ticks_per_quarter=220))
  # rhythm flag for each note added to `note_sequence`, in order; the carrier
  # `assign_instruments` needs since notes added here have no rhythm field
  note_rhythms: List[bool] = dataclasses.field(default_factory=list)

  def begin_suppressed_prefix(self, start_time: float) -> None:
    begin_suppressed_prefix(self, start_time)

  def finish_suppressed_prefix(self, boundary_time: float) -> None:
    finish_suppressed_prefix(self, boundary_time)


def decode_note_onset_event(
    state: NoteDecodingState,
    time: float,
    event: event_codec.Event,
    codec: event_codec.Codec,
) -> None:
  """Process note onset event and update decoding state."""
  if event.type == 'pitch':
    if state.suppress:
      # Already added to note_sequence by an earlier window's kept region
      # (NoteOnsetEncodingSpec has no active_pitches/tie-section state to
      # check against -- every onset is unconditional -- so without this
      # guard a re-declared onset in the suppressed prefix would duplicate
      # the note outright).
      return
    state.note_sequence.notes.add(
        start_time=time, end_time=time + DEFAULT_NOTE_DURATION,
        pitch=event.value, velocity=DEFAULT_VELOCITY)
    state.note_sequence.total_time = max(state.note_sequence.total_time,
                                         time + DEFAULT_NOTE_DURATION)
    state.note_rhythms.append(False)
  else:
    raise ValueError('unexpected event type: %s' % event.type)


def _add_note_to_sequence(
    state: NoteDecodingState,
    start_time: float, end_time: float, pitch: int, velocity: int,
    program: int = 0, is_drum: bool = False, rhythm: bool = False
) -> None:
  end_time = max(end_time, start_time + MIN_NOTE_DURATION)
  state.note_sequence.notes.add(
      start_time=start_time, end_time=end_time,
      pitch=pitch, velocity=velocity, program=program, is_drum=is_drum)
  state.note_sequence.total_time = max(state.note_sequence.total_time,
                                       end_time)
  state.note_rhythms.append(rhythm)


def decode_note_event(
    state: NoteDecodingState,
    time: float,
    event: event_codec.Event,
    codec: event_codec.Codec
) -> None:
  """Process note event and update decoding state."""
  if state.suppress and state.prefix_state is not None:
    # Decode the overlap according to this window's own event stream without
    # touching the already-committed output.  At min_time, decode_events calls
    # finish_suppressed_prefix() to use the resulting active set solely as a
    # boundary reconciliation signal.
    prefix_state = state.prefix_state
    prefix_state.suppress = False
    decode_note_event(prefix_state, time, event, codec)
    return
  if not state.suppress:
    # A suppressed event's time is by construction earlier than wherever
    # `state` has already advanced to (the previous window's kept region
    # covered up through this window's min_time), so it would fail this
    # monotonicity check spuriously; skip both the check and the advance
    # for it. Monotonicity resumes automatically at the first kept event,
    # since the previous window could not have committed anything at or
    # past this window's min_time (see run_length_encoding.decode_events'
    # half-open [min_time, max_time) interval).
    if time < state.current_time:
      raise ValueError('event time < current time, %f < %f' % (
          time, state.current_time))
    state.current_time = time
  if state.suppress and event.type in ('pitch', 'drum'):
    # This onset/offset was already committed by an earlier window's kept
    # region -- active_pitches (or note_sequence, for a drum hit) already
    # reflects it, or its closure. Re-applying it here would double-commit
    # an onset, or raise on a note-off for a pitch that has since
    # legitimately closed.
    return
  if event.type == 'pitch':
    pitch = event.value
    key = (pitch, state.current_program, state.current_rhythm)
    if state.is_tie_section:
      # "tied" pitch
      if key not in state.active_pitches:
        if state.is_prefix_shadow:
          # This window believes the pitch was already sounding at its own
          # start, but the preceding kept region did not.  The preceding
          # region remains authoritative, so do not synthesize an onset.
          return
        raise ValueError(
            'inactive pitch/program/rhythm in tie section: %d/%d/%d' %
            (pitch, state.current_program, state.current_rhythm))
      if key in state.tied_pitches:
        raise ValueError('pitch/program/rhythm is already tied: %d/%d/%d' %
                         (pitch, state.current_program, state.current_rhythm))
      state.tied_pitches.add(key)
    elif state.current_velocity == 0:
      # note offset
      if key not in state.active_pitches:
        if state.is_prefix_shadow:
          # The preceding kept region may already have closed this key, or
          # the two overlapping predictions may disagree about its identity.
          # This prefix is reconciliation evidence, not committed output, so
          # an unmatched shadow offset is harmless.
          return
        raise ValueError(
            'note-off for inactive pitch/program/rhythm: %d/%d/%d' %
            (pitch, state.current_program, state.current_rhythm))
      onset_time, onset_velocity = state.active_pitches.pop(key)
      _add_note_to_sequence(
          state, start_time=onset_time, end_time=time,
          pitch=pitch, velocity=onset_velocity, program=state.current_program,
          rhythm=state.current_rhythm)
    else:
      # note onset
      if key in state.active_pitches:
        # The pitch is already active; this shouldn't really happen but we'll
        # try to handle it gracefully by ending the previous note and starting a
        # new one.
        onset_time, onset_velocity = state.active_pitches.pop(key)
        _add_note_to_sequence(
            state, start_time=onset_time, end_time=time,
            pitch=pitch, velocity=onset_velocity,
            program=state.current_program, rhythm=state.current_rhythm)
      state.active_pitches[key] = (time, state.current_velocity)
  elif event.type == 'drum':
    # drum onset (drums have no offset)
    if state.current_velocity == 0:
      raise ValueError('velocity cannot be zero for drum event')
    offset_time = time + DEFAULT_NOTE_DURATION
    _add_note_to_sequence(
        state, start_time=time, end_time=offset_time,
        pitch=event.value, velocity=state.current_velocity, is_drum=True)
  elif event.type == 'velocity':
    # velocity change
    num_velocity_bins = vocabularies.num_velocity_bins_from_codec(codec)
    velocity = vocabularies.bin_to_velocity(event.value, num_velocity_bins)
    state.current_velocity = velocity
  elif event.type == 'program':
    # program change -- ignored (beyond having been consumed above) when
    # force_program pins current_program to a fixed value.
    if state.force_program is None:
      state.current_program = event.value
  elif event.type == 'rhythm':
    # rhythm flag change -- ignored (beyond having been consumed above) when
    # no_rhythm pins current_rhythm to False, same as the 'program' branch
    # above. force_program implies no_rhythm, so a forced single instrument
    # collapses to one (program, rhythm) group rather than two.
    if not state.no_rhythm:
      state.current_rhythm = bool(event.value)
  elif event.type == 'tie':
    if state.suppress:
      # The tie section describes what's sounding at the window's first
      # frame, lookback_frames before min_time -- not at the kept region's
      # start, which is what active_pitches (carried over from the
      # previous window's kept region ending exactly there) already
      # reflects correctly. Discard it: closing "untied" notes here would
      # spuriously cut off a note that is legitimately still sounding.
      state.is_tie_section = False
      return
    # end of tie section; end active notes that weren't declared tied
    if not state.is_tie_section:
      raise ValueError('tie section end event when not in tie section')
    for (pitch, program, rhythm) in list(state.active_pitches.keys()):
      if (pitch, program, rhythm) not in state.tied_pitches:
        onset_time, onset_velocity = state.active_pitches.pop(
            (pitch, program, rhythm))
        _add_note_to_sequence(
            state,
            start_time=onset_time, end_time=state.current_time,
            pitch=pitch, velocity=onset_velocity, program=program,
            rhythm=rhythm)
    state.is_tie_section = False
  else:
    raise ValueError('unexpected event type: %s' % event.type)


def begin_tied_pitches_section(state: NoteDecodingState) -> None:
  """Begin the tied pitches section at the start of a segment."""
  state.tied_pitches = set()
  state.is_tie_section = True


def begin_suppressed_prefix(state: NoteDecodingState,
                            start_time: float) -> None:
  """Start a segment-local decode of a lookback prefix.

  The active-pitch keys are copied from the committed state so this window's
  tie section and lookback events can decide which carried notes still exist
  at the kept boundary.  Modal decoder state is reset to the same defaults as
  a fresh segment; the segment's own tokens establish program/rhythm/velocity.
  Completed shadow notes are intentionally thrown away at reconciliation.
  """
  state.prefix_state = NoteDecodingState(
      current_time=start_time,
      active_pitches=dict(state.active_pitches),
      is_tie_section=state.is_tie_section,
      is_prefix_shadow=True,
      force_program=state.force_program,
      no_rhythm=state.no_rhythm)


def finish_suppressed_prefix(state: NoteDecodingState,
                             boundary_time: float) -> None:
  """Reconcile committed active notes with a decoded lookback prefix.

  The preceding kept region remains authoritative for emitted notes, so keys
  seen only by the new window are not copied into the committed state.  A
  committed key that the new window no longer considers active is closed at
  the boundary.  This bounds disagreement between adjacent model windows to
  one kept region instead of letting a missed/mismatched note-off smear to the
  end of the transcription.
  """
  prefix_state = state.prefix_state
  if prefix_state is None:
    return

  for key, committed_onset in list(state.active_pitches.items()):
    shadow_onset = prefix_state.active_pitches.get(key)
    same_note_lifecycle = (
        shadow_onset is not None and
        abs(shadow_onset[0] - committed_onset[0]) < 1e-6 and
        shadow_onset[1] == committed_onset[1])
    if not same_note_lifecycle:
      onset_time, onset_velocity = state.active_pitches.pop(key)
      pitch, program, rhythm = key
      _add_note_to_sequence(
          state, start_time=onset_time, end_time=boundary_time,
          pitch=pitch, velocity=onset_velocity, program=program,
          rhythm=rhythm)

  # Sticky state for the kept suffix must come from this window's own prefix,
  # not from the preceding window's decoder stream.
  state.current_velocity = prefix_state.current_velocity
  state.current_program = prefix_state.current_program
  state.current_rhythm = prefix_state.current_rhythm
  state.current_time = max(state.current_time, boundary_time)
  state.tied_pitches = set()
  state.is_tie_section = False
  state.prefix_state = None


def flush_note_decoding_state(
    state: NoteDecodingState
) -> note_seq.NoteSequence:
  """End all active notes and return resulting NoteSequence."""
  for onset_time, _ in state.active_pitches.values():
    state.current_time = max(state.current_time, onset_time + MIN_NOTE_DURATION)
  for (pitch, program, rhythm) in list(state.active_pitches.keys()):
    onset_time, onset_velocity = state.active_pitches.pop(
        (pitch, program, rhythm))
    if state.current_time - onset_time > LONG_HELD_NOTE_WARNING_SECONDS:
      logging.warning(
          'Force-closing a note held open for %.1fs at end of decode '
          '(pitch=%d, program=%d) -- likely a dropped note-off rather '
          'than a real sustained note.',
          state.current_time - onset_time, pitch, program)
    _add_note_to_sequence(
        state, start_time=onset_time, end_time=state.current_time,
        pitch=pitch, velocity=onset_velocity, program=program, rhythm=rhythm)
  assign_instruments(state.note_sequence, note_rhythms=state.note_rhythms)
  return state.note_sequence


class NoteEncodingSpecType(run_length_encoding.EventEncodingSpec):
  pass


# encoding spec for modeling note onsets only
NoteOnsetEncodingSpec = NoteEncodingSpecType(
    init_encoding_state_fn=lambda: None,
    encode_event_fn=note_event_data_to_events,
    encoding_state_to_events_fn=None,
    init_decoding_state_fn=NoteDecodingState,
    begin_decoding_segment_fn=lambda state: None,
    decode_event_fn=decode_note_onset_event,
    flush_decoding_state_fn=lambda state: state.note_sequence)


# encoding spec for modeling onsets and offsets
NoteEncodingSpec = NoteEncodingSpecType(
    init_encoding_state_fn=lambda: None,
    encode_event_fn=note_event_data_to_events,
    encoding_state_to_events_fn=None,
    init_decoding_state_fn=NoteDecodingState,
    begin_decoding_segment_fn=lambda state: None,
    decode_event_fn=decode_note_event,
    flush_decoding_state_fn=flush_note_decoding_state)


# encoding spec for modeling onsets and offsets, with a "tie" section at the
# beginning of each segment listing already-active notes
NoteEncodingWithTiesSpec = NoteEncodingSpecType(
    init_encoding_state_fn=NoteEncodingState,
    encode_event_fn=note_event_data_to_events,
    encoding_state_to_events_fn=note_encoding_state_to_events,
    init_decoding_state_fn=NoteDecodingState,
    begin_decoding_segment_fn=begin_tied_pitches_section,
    decode_event_fn=decode_note_event,
    flush_decoding_state_fn=flush_note_decoding_state)

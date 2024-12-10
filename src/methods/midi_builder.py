import pandas as pd
import pretty_midi
import matplotlib.pyplot as plt
import librosa.display as display

class MidiBuilder:
    """Build a MIDI from a piano roll sample."""

    def __init__(self, midi_start=48, midi_end=108):
        """
        Initialize the MidiBuilder.

        Args:
            midi_start (int): The first MIDI note in the dataset.
            midi_end (int): The last MIDI note in the dataset.
        """
        self.dtypes = {'piano_roll_name': 'object', 'timestep': 'uint32'}
        self.column_names = [pretty_midi.note_number_to_name(n) for n in range(midi_start, midi_end)]
        for column in self.column_names:
            self.dtypes[column] = 'uint8'

    def midi_from_piano_roll(self, sample, tempo=120):
        """
        Convert a piano roll sample to a MIDI file.

        Args:
            sample (numpy.ndarray): Piano roll array with one-hot encoding for notes.
            tempo (int): Tempo in beats per minute (default is 120).

        Returns:
            pretty_midi.PrettyMIDI: The reconstructed MIDI object.
        """
        piano_roll = pd.DataFrame(sample, columns=self.column_names, dtype='uint8')

        program = 0  # Acoustic Grand Piano
        velocity = 100  # Note velocity
        bps = tempo / 60  # Beats per second
        sps = bps * 4  # Sixteenth notes per second

        # Create a PrettyMIDI object
        piano_midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=program)

        # Iterate through the piano roll to generate notes
        for idx in piano_roll.index:
            for note_name in piano_roll.columns:
                if piano_roll.iloc[idx][note_name] == 1:
                    note_number = pretty_midi.note_name_to_number(note_name)
                    note_start = idx / sps
                    note_end = (idx + 1) / sps

                    # Create and add a Note instance to the instrument
                    note = pretty_midi.Note(
                        velocity=velocity, pitch=note_number, start=note_start, end=note_end)
                    piano.notes.append(note)

        # Add the instrument to the PrettyMIDI object
        piano_midi.instruments.append(piano)
        return piano_midi

    def plot_midi(self, midi_sample):
        """
        Plot the piano roll of a MIDI sample.

        Args:
            midi_sample (pretty_midi.PrettyMIDI): MIDI object to plot.
        """
        display.specshow(midi_sample.get_piano_roll(), y_axis='cqt_note', cmap=plt.cm.hot)

    def play_midi(self, midi_sample):
        """
        Synthesize and play a MIDI sample.

        Args:
            midi_sample (pretty_midi.PrettyMIDI): MIDI object to synthesize.

        Returns:
            tuple: Synthesized audio signal and sampling frequency.
        """
        fs = 44100  # Sampling frequency
        synth = midi_sample.synthesize(fs=fs)
        return [synth], fs

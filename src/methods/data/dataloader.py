import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import pretty_midi

class MidiDataset(Dataset):
    """Pre-processed MIDI dataset."""

    def __init__(self, csv_file, transform, midi_start=48, midi_end=108):
        """
        Initialize the dataset.

        Args:
            csv_file (str): Path to the CSV file with piano rolls per song.
            transform (callable): Transform to be applied on a sample. Must implement `get_sections`.
            midi_start (int): First MIDI note in the dataset.
            midi_end (int): Last MIDI note in the dataset.
        """
        dtypes = {'piano_roll_name': 'object', 'timestep': 'uint32'}
        column_names = [pretty_midi.note_number_to_name(n) for n in range(midi_start, midi_end)]
        for column in column_names:
            dtypes[column] = 'uint8'

        # Load the piano roll dataset
        self.piano_rolls = pd.read_csv(csv_file, sep=';', index_col=['piano_roll_name', 'timestep'], dtype=dtypes)
        self.transform = transform
        self.init_dataset()

    def init_dataset(self):
        """
        Initialize the dataset index mapper for accessing sections of songs.
        """
        indexer = self._get_indexer()
        self.index_mapper = [
            (i, j)
            for i in indexer
            for j in range(self.transform.get_sections(len(self.piano_rolls.loc[i].values)))
        ]

    def __len__(self):
        """
        Return the total number of sections in the dataset.
        """
        return len(self.index_mapper)

    def get_mem_usage(self):
        """
        Get the memory usage of the dataset in MB.
        """
        return self.piano_rolls.memory_usage(deep=True).sum() / 1024**2

    def _get_indexer(self):
        """
        Get unique song names from the dataset index.
        """
        return self.piano_rolls.index.get_level_values(0).unique()

    def __getitem__(self, idx):
        """
        Get a specific sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing the transformed piano roll section.
        """
        song_name, section = self.index_mapper[idx]

        # Add a column for silence notes
        piano_rolls = self.piano_rolls.loc[song_name].values
        silence_col = np.zeros((piano_rolls.shape[0], 1))
        piano_rolls_with_silences = np.append(piano_rolls, silence_col, axis=1)

        # Transform the sample and select the section
        sample = self.transform(piano_rolls_with_silences.astype('float'))[section]

        # Mark silent rows explicitly
        empty_rows = ~sample.any(axis=1)
        if len(sample[empty_rows]) > 0:
            sample[empty_rows, -1] = 1.0

        return {'piano_rolls': sample}

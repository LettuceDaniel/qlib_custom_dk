import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, features, labels, timestamps=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.timestamps = timestamps

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.timestamps is not None:
            ts = self.timestamps[idx]
            if isinstance(ts, np.datetime64):
                ts = str(np.datetime64(ts, 'D'))
            return self.features[idx], self.labels[idx], ts
        return self.features[idx], self.labels[idx]

class DateGroupedBatchSampler:
    def __init__(self, dates, batch_size, shuffle=True, seed=None):
        self.dates = np.asarray(dates)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Build date -> indices mapping once (stable across epochs)
        self.date_to_indices = {}
        for idx, date in enumerate(self.dates):
            date_key = np.datetime64(date, "D")
            # Skip NaT values to avoid grouping them into a spurious single group
            if np.isnat(date_key):
                continue
            if date_key not in self.date_to_indices:
                self.date_to_indices[date_key] = []
            self.date_to_indices[date_key].append(idx)

    def __iter__(self):
        # Rebuild batches each epoch for proper shuffling
        date_keys = list(self.date_to_indices.keys())
        if self.shuffle:
            self._rng.shuffle(date_keys)

        batches = []
        for date_key in date_keys:
            indices = self.date_to_indices[date_key]
            if self.shuffle:
                indices = indices.copy()
                self._rng.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                # Require at least 2 samples per batch for pairwise ranking loss
                if len(batch) >= 2:
                    batches.append(batch)

        if self.shuffle:
            self._rng.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        # Bug #2: __len__ and __iter__ must stay synchronized.
        # If __iter__ adds new filtering (e.g., skip small batches), __len__ must reflect it.
        count = 0
        for indices in self.date_to_indices.values():
            n = len(indices)
            count += (n // self.batch_size)
            if n % self.batch_size >= 2:
                count += 1
        return count

def collate_fn(batch_data):
    """Standard collate function for TimeSeriesDataset.

    Handles both 2-tuple (features, labels) and 3-tuple (features, labels, timestamps)
    return types consistently.
    """
    features = torch.stack([x[0] for x in batch_data])
    labels = torch.stack([x[1] for x in batch_data])
    if len(batch_data[0]) > 2:
        timestamps = [x[2] for x in batch_data]
        return features, labels, timestamps
    return features, labels


def create_date_grouped_dataloader(dataset, dates, batch_size, shuffle=True, num_workers=0, seed=None):
    sampler = DateGroupedBatchSampler(dates, batch_size, shuffle=shuffle, seed=seed)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=collate_fn)
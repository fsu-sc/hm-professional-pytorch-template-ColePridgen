import torch
from torch.utils.data import Dataset
import numpy as np
from base import BaseDataLoader

class FunctionDataset(Dataset):
    def __init__(self, n_samples=100, function='linear'):
        self.n_samples = n_samples
        self.function = function.lower()

        # Generate input x values between 0 and 2π
        self.x = np.random.uniform(0, 2 * np.pi, self.n_samples)
        
        # Add noise ε ~ U(-1, 1)
        noise = np.random.uniform(-1, 1, self.n_samples)

        # Compute y values based on the function type
        if self.function == 'linear':
            y = 1.5 * self.x + 0.3 + noise
        elif self.function == 'quadratic':
            y = 2 * self.x**2 + 0.5 * self.x + 0.3 + noise
        elif self.function == 'harmonic':
            y = 0.5 * self.x**2 + 5 * np.sin(self.x) + 3 * np.cos(3 * self.x) + 2 + noise
        else:
            raise ValueError(f"Unsupported function type: {self.function}")

        # Normalize x and y
        self.x = (self.x - np.mean(self.x)) / np.std(self.x)
        y = (y - np.mean(y)) / np.std(y)

        self.y = y

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.tensor([self.x[idx]], dtype=torch.float32), torch.tensor([self.y[idx]], dtype=torch.float32)


class FunctionDataLoader(BaseDataLoader):
    def __init__(self, data_dir=None, batch_size=32, shuffle=True, validation_split=0.0, num_workers=1,
                 function='linear', n_samples=100):
        self.dataset = FunctionDataset(n_samples, function)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


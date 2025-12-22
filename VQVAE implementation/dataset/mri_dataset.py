import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pywt

class MRIDataset(Dataset):
    def __init__(self, root, split="train", wavelet="haar"):
        self.root = Path(root) / split
        self.files = sorted(self.root.glob("*.npz"))
        self.wavelet = wavelet

        if len(self.files) == 0:
            raise RuntimeError(f"No data found in {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])

        low = d["low"].astype(np.float32)
        high = d["high"].astype(np.float32)

        if high.max() < 1e-5:
            return self.__getitem__((idx + 1) % len(self))

        low = low / (low.max() + 1e-8)
        high = high / (high.max() + 1e-8)

        LL_l, (LH_l, HL_l, HH_l) = pywt.dwt2(low, self.wavelet)
        LL_h, (LH_h, HL_h, HH_h) = pywt.dwt2(high, self.wavelet)

        input_stack = np.stack([LL_l, LH_l, HL_l, HH_l], axis=0)
        target_stack = np.stack([LL_h, LH_h, HL_h, HH_h], axis=0)

        return {
            "input_stack": torch.from_numpy(input_stack),
            "target_stack": torch.from_numpy(target_stack),
            "low_original": torch.from_numpy(low),
            "high_original": torch.from_numpy(high)
        }

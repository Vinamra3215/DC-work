import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict
import random
from tqdm import tqdm

DATA_ROOT = Path("/scratch/b24cm1068")

SRC_DIRS = [
    DATA_ROOT / "multicoil_train",
    DATA_ROOT / "multicoil_val",
]

OUT_ROOT = DATA_ROOT / "processed"
OUT_TRAIN = OUT_ROOT / "train"
OUT_VAL   = OUT_ROOT / "val"
OUT_TEST  = OUT_ROOT / "test"

for d in [OUT_TRAIN, OUT_VAL, OUT_TEST]:
    d.mkdir(parents=True, exist_ok=True)

def rss(kspace):
    """Root Sum of Squares reconstruction with Correct Centering"""
    k_shifted = np.fft.ifftshift(kspace, axes=(-2, -1))
    img_shifted = np.fft.ifft2(k_shifted, axes=(-2, -1))
    img = np.fft.fftshift(img_shifted, axes=(-2, -1))
    return np.sqrt((np.abs(img) ** 2).sum(axis=0)) 


def collect_groups():
    """
    Collect T1 scans from all source dirs and group by patient ID
    """
    groups = defaultdict(list)

    for src in SRC_DIRS:
        for f in src.glob("*_T1*.h5"):
            pid = f.name.split("_")[0]
            groups[pid].append(f)

    return groups


def process_patient(files, out_dir):
    """
    Create (low NSA, high NSA) slice pairs for one patient
    """
    if len(files) < 3:
        return

    low_file = random.choice(files)
    high_files = [f for f in files if f != low_file]

    with h5py.File(low_file, "r") as hf:
        low_k = hf["kspace"][:]

    high_k = []
    for f in high_files:
        with h5py.File(f, "r") as hf:
            high_k.append(hf["kspace"][:])

    high_k = np.mean(np.stack(high_k, axis=0), axis=0)

    low_img = rss(low_k)
    high_img = rss(high_k)

    for i in range(low_img.shape[0]):
        np.savez(
            out_dir / f"{low_file.stem}_slice{i}.npz",
            low=low_img[i],
            high=high_img[i]
        )


if __name__ == "__main__":
    random.seed(42)

    groups = collect_groups()
    patient_ids = list(groups.keys())
    random.shuffle(patient_ids)

    n_total = len(patient_ids)
    n_train = int(0.7 * n_total)
    n_val   = int(0.1 * n_total)

    train_ids = patient_ids[:n_train]
    val_ids   = patient_ids[n_train:n_train + n_val]
    test_ids  = patient_ids[n_train + n_val:]

    print(f"\nPatients total: {n_total}")
    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    print("\n🔹 Processing TRAIN split")
    for pid in tqdm(train_ids):
        process_patient(groups[pid], OUT_TRAIN)

    print("\n🔹 Processing VAL split")
    for pid in tqdm(val_ids):
        process_patient(groups[pid], OUT_VAL)

    print("\n🔹 Processing TEST split")
    for pid in tqdm(test_ids):
        process_patient(groups[pid], OUT_TEST)

    print("\n✅ Finished building processed dataset (patient-safe split)")
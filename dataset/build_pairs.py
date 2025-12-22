import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict
import random
import shutil
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

def rss(kspace):
    k_shifted = np.fft.ifftshift(kspace, axes=(-2, -1))
    img_shifted = np.fft.ifft2(k_shifted, axes=(-2, -1))
    img_centered = np.fft.fftshift(img_shifted, axes=(-2, -1))
    return np.sqrt((np.abs(img_centered) ** 2).sum(axis=0))


def collect_groups():
    groups = defaultdict(list)
    print("🔍 Scanning directories for T1 files...")
    total_files = 0
    for src in SRC_DIRS:
        if not src.exists():
            print(f"⚠️ Warning: Source directory not found: {src}")
            continue
        for f in src.glob("*_T1*.h5"):
            pid = f.name.split("_")[0]
            groups[pid].append(f)
            total_files += 1
    valid_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"✅ Found {total_files} T1 files.")
    print(f"✅ Grouped into {len(valid_groups)} unique patients (with 2+ scans).")
    return valid_groups


def process_patient(patient_id, files, out_dir):
    if not files: return 0
    low_file = random.choice(files)
    try:
        with h5py.File(low_file, "r") as hf:
            low_k = hf["kspace"][:] 
        high_k_list = []
        for f in files:
            with h5py.File(f, "r") as hf:
                high_k_list.append(hf["kspace"][:])
        high_k_avg = np.mean(np.stack(high_k_list, axis=0), axis=0)
        def reconstruct_volume(vol_k):
            slices = []
            for i in range(vol_k.shape[0]):
                sl = rss(vol_k[i])
                slices.append(sl)
            return np.array(slices)

        low_img = reconstruct_volume(low_k)       
        high_img = reconstruct_volume(high_k_avg) 
        num_slices = low_img.shape[0]
        start_slice = 1
        end_slice = num_slices - 1
        if end_slice <= start_slice:
            start_slice = 0
            end_slice = num_slices

        saved_count = 0
        for i in range(start_slice, end_slice):
            if high_img[i].max() > 1e-5:
                np.savez(
                    out_dir / f"{patient_id}_slice{i}.npz",
                    low=low_img[i],
                    high=high_img[i]
                )
                saved_count += 1
        return saved_count
            
    except Exception as e:
        print(f"⚠️ Error processing patient {patient_id}: {e}")
        return 0
    
if __name__ == "__main__":
    random.seed(42)
    print("🧹 Cleaning old processed data...")
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    
    for d in [OUT_TRAIN, OUT_VAL, OUT_TEST]:
        d.mkdir(parents=True, exist_ok=True)

    groups = collect_groups()
    patient_ids = list(groups.keys())
    random.shuffle(patient_ids)
    n_total = len(patient_ids)

    if n_total == 0:
        print("❌ CRITICAL: No valid patients found. Check your DATA_ROOT path.")
        exit()
    n_val   = int(n_total * 0.10)
    n_test  = int(n_total * 0.20)
    if n_total > 2:
        n_val = max(1, n_val)
        n_test = max(1, n_test)
        
    n_train = n_total - n_val - n_test
    train_ids = patient_ids[:n_train]
    val_ids   = patient_ids[n_train : n_train + n_val]
    test_ids  = patient_ids[n_train + n_val :]

    print(f"\n📊 Dataset Split (Total Patients: {n_total})")
    print(f"   Train: {len(train_ids)} patients (70%)")
    print(f"   Val:   {len(val_ids)} patients (10%)")
    print(f"   Test:  {len(test_ids)} patients (20%)")
    
    print("\n🔹 Processing TRAIN split...")
    count_train = 0
    for pid in tqdm(train_ids):
        count_train += process_patient(pid, groups[pid], OUT_TRAIN)

    print("\n🔹 Processing VAL split...")
    count_val = 0
    for pid in tqdm(val_ids):
        count_val += process_patient(pid, groups[pid], OUT_VAL)

    print("\n🔹 Processing TEST split...")
    count_test = 0
    for pid in tqdm(test_ids):
        count_test += process_patient(pid, groups[pid], OUT_TEST)

    print("\n✅ Processing Complete.")
    print(f"   Total Images -> Train: {count_train} | Val: {count_val} | Test: {count_test}")
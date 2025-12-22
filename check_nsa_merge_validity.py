import h5py
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

DATA_DIR = "/scratch/b24cm1068/multicoil_train/"
MODALITY_KEY = "_T1"
PATIENT_ID_LEN = 10

def parse_ismrmrd_header(h5_file):
    header = h5_file["ismrmrd_header"][()].decode("utf-8")
    root = ET.fromstring(header)

    def get_text(path):
        el = root.find(path)
        return el.text if el is not None else None

    info = {
        "matrix_x": get_text(".//encodedSpace/matrixSize/x"),
        "matrix_y": get_text(".//encodedSpace/matrixSize/y"),
        "matrix_z": get_text(".//encodedSpace/matrixSize/z"),
        "fov_x": get_text(".//encodedSpace/fieldOfView_mm/x"),
        "fov_y": get_text(".//encodedSpace/fieldOfView_mm/y"),
        "fov_z": get_text(".//encodedSpace/fieldOfView_mm/z"),
        "TR": get_text(".//sequenceParameters/TR"),
        "TE": get_text(".//sequenceParameters/TE"),
        "TI": get_text(".//sequenceParameters/TI"),
        "flipAngle": get_text(".//sequenceParameters/flipAngle_deg"),
        "sequenceType": get_text(".//sequenceParameters/sequence_type"),
    }
    return info

def check_group(files):
    print("\n" + "=" * 80)
    print("Checking files:")
    for f in files:
        print(" ", f.name)

    shapes = []
    headers = []

    for f in files:
        with h5py.File(f, "r") as hf:
            kspace = hf["kspace"]
            shapes.append(kspace.shape)
            headers.append(parse_ismrmrd_header(hf))

    if len(set(shapes)) != 1:
        print("❌ Matrix size mismatch:")
        for f, s in zip(files, shapes):
            print(f"   {f.name}: {s}")
        return False

    print("✅ Matrix size identical:", shapes[0])

    matrix_xyz = [
        (h["matrix_x"], h["matrix_y"], h["matrix_z"]) for h in headers
    ]
    if len(set(matrix_xyz)) != 1:
        print("❌ Slice geometry mismatch:")
        for f, m in zip(files, matrix_xyz):
            print(f"   {f.name}: matrix={m}")
        return False

    print("✅ Slice geometry identical:", matrix_xyz[0])

    contrast_keys = ["TR", "TE", "TI", "flipAngle", "sequenceType"]
    for key in contrast_keys:
        values = [h[key] for h in headers]
        if len(set(values)) != 1:
            print(f"❌ Contrast mismatch in {key}:")
            for f, v in zip(files, values):
                print(f"   {f.name}: {key}={v}")
            return False

    print("✅ Acquisition parameters identical (TR/TE/TI/FlipAngle)")
    print("🎯 RESULT: SAFE TO MERGE AS HIGH-NSA")
    return True

def main():
    files = sorted(Path(DATA_DIR).glob("*.h5"))
    groups = defaultdict(list)

    for f in files:
        patient_id = f.name[:PATIENT_ID_LEN]
        if "_T1" in f.name:
            groups[patient_id].append(f)

    for pid, flist in groups.items():
        if len(flist) < 2:
            continue
        print(f"\nPatient {pid}:")
        check_group(flist)

if __name__ == "__main__":
    main()

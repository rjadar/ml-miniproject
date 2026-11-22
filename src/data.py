"""Read SisFall recordings, split by subject, normalize, and create CNN windows."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "SisFall" / "SisFall_dataset"
SAMPLE_RATE = 200
WINDOW_SIZE = 15 * SAMPLE_RATE
CHANNELS = 9
SEED = 2025
FILE_PATTERN = re.compile(r"^(?P<activity>[DF]\d{2})_(?P<subject>(?:SA|SE)\d{2})_R\d{2}\.txt$", re.I)

# Fixed subject split used for the reported results. SA15 and SE15 are both in
# test because the supplied archive has five conflicting D17 filenames.
SUBJECT_SPLIT = {
    "train": set("SA01 SA03 SA04 SA06 SA07 SA08 SA09 SA10 SA12 SA13 SA14 SA16 SA18 SA19 SA21 SA23 SE03 SE04 SE05 SE07 SE08 SE09 SE10 SE11 SE12 SE13".split()),
    "validation": set("SA02 SA05 SA11 SA22 SE02 SE14".split()),
    "test": set("SA15 SA17 SA20 SE01 SE06 SE15".split()),
}


def read_recording(path: Path) -> np.ndarray:
    """Read one SisFall text file as samples by nine sensor channels."""
    rows = [line.strip().rstrip(b";").strip() for line in path.read_bytes().splitlines() if line.strip()]
    values = np.fromstring(b",".join(rows).decode("ascii"), sep=",", dtype=np.float32)
    if values.size == 0 or values.size % CHANNELS:
        raise ValueError(f"Expected nine numeric sensor values per row: {path}")
    return values.reshape(-1, CHANNELS)


def to_physical_units(raw: np.ndarray) -> np.ndarray:
    """Convert SisFall sensor counts to g and degrees per second."""
    scale = np.array([32 / 2**13] * 3 + [4000 / 2**16] * 3 + [16 / 2**14] * 3, dtype=np.float32)
    return raw * scale


def load_windows(normalization: dict | None = None) -> tuple[dict, dict]:
    """Load recordings, fit scaling on training subjects, and return split windows."""
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"SisFall dataset folder not found: {DATA_DIR}")

    files = []
    for path in sorted(DATA_DIR.rglob("*.txt")):
        match = FILE_PATTERN.match(path.name)
        if match:
            subject = match.group("subject").upper()
            split = next((name for name, subjects in SUBJECT_SPLIT.items() if subject in subjects), None)
            if split is None:
                raise ValueError(f"No subject split assigned for {path.name}")
            files.append((path, split, int(match.group("activity").upper().startswith("F"))))
    if not files:
        raise FileNotFoundError(f"No SisFall activity recordings found under {DATA_DIR}")

    if normalization is None:
        channel_sum = np.zeros(CHANNELS, dtype=np.float64)
        channel_sum_sq = np.zeros(CHANNELS, dtype=np.float64)
        sample_count = 0
        for path, split, _ in files:
            if split == "train":
                values = to_physical_units(read_recording(path)).astype(np.float64)
                channel_sum += values.sum(axis=0)
                channel_sum_sq += np.square(values).sum(axis=0)
                sample_count += len(values)
        mean = channel_sum / sample_count
        std = np.sqrt(np.maximum(channel_sum_sq / sample_count - mean**2, 1e-12))
        normalization = {"mean": mean.astype(np.float32).tolist(), "std": std.astype(np.float32).tolist()}
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)

    window_lists = {name: [] for name in SUBJECT_SPLIT}
    label_lists = {name: [] for name in SUBJECT_SPLIT}
    recording_lists = {name: [] for name in SUBJECT_SPLIT}
    for recording_index, (path, split, label) in enumerate(files):
        values = (to_physical_units(read_recording(path)) - mean) / std
        for start in range(0, len(values), WINDOW_SIZE):
            block = values[start:start + WINDOW_SIZE]
            window = np.zeros((CHANNELS, WINDOW_SIZE), dtype=np.float32)
            window[:, :len(block)] = block.T
            window_lists[split].append(window)
            label_lists[split].append(label)
            recording_lists[split].append(recording_index)

    splits = {}
    for name in SUBJECT_SPLIT:
        x = np.stack(window_lists[name])
        y = np.asarray(label_lists[name], dtype=np.float32)
        recording_ids = np.asarray(recording_lists[name], dtype=np.int64)
        splits[name] = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(recording_ids))
    return splits, normalization

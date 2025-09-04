# -*- coding: utf-8 -*-
"""
LabChart (.adicht) → HDF5 converter driven by an index file (single input: parent path)

This script:
- Prompts the user for the parent path (no argparse).
- Prompts for **expected channel order** (comma-separated, e.g., `vHPC,EMG,FC`) and enforces it per recording
  using the `channel_user` labels from the index.
- Expects inside that folder:
  - labchart_data/  → contains your `.adicht` files
  - file_index.csv → index summarizing file/block/channel mapping (must include
    `recording_id, file_name, channel_name, channel_user, block_index, animal_id`).
- Creates h5_data/ inside the parent folder.
- Before full conversion, quickly probes each file by reading a tiny slice to confirm access.
- Converts each unique `recording_id` group into **one H5** with windows at **100 Hz**,
  **5 seconds per window**, skipping blocks shorter than **10 minutes**.

Zero‑phase decimation
---------------------
We use `scipy.signal.decimate(..., zero_phase=True)`, which applies the anti‑aliasing filter
forward **and** backward (like filtfilt), producing ~0° net phase shift. That prevents the
waveform delays a one‑pass IIR/FIR decimator would introduce.
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal
import tables  # PyTables
import adi

# ========================= Constants ========================= #
TARGET_FS = 100          # Hz after decimation
WINDOW_SEC = 5.0         # seconds per window
MIN_BLOCK_SEC = 600.0    # 10 minutes cutoff
INDEX_CSV  = "file_index.csv"
LABCHART_DIRNAME = "labchart_data"
H5_DIRNAME = "h5_data"

# ========================= Core class ======================== #

class IndexToH5Converter:
    """Convert LabChart blocks to HDF5 windows using an index file.

    Parameters
    ----------
    parent_path : str
        Parent directory containing `labchart_data/` and `file_index.csv`.
    expected_order : list[str] | None
        Canonical channel labels (as they appear in `channel_user`) to **enforce and order**
        per recording (e.g., ["vHPC","EMG","FC"]). If None/empty, the CSV order is used.
    """

    def __init__(self, parent_path: str, expected_order=None):
        self.parent_path = os.path.abspath(parent_path)
        self.labchart_path = os.path.join(self.parent_path, LABCHART_DIRNAME)
        self.h5_path = os.path.join(self.parent_path, H5_DIRNAME)
        os.makedirs(self.h5_path, exist_ok=True)

        # Load index
        csv_path  = os.path.join(self.parent_path, INDEX_CSV)
        if os.path.exists(csv_path):
            self.index_file = csv_path
            self.index = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(
                f"No index file found. Place '{INDEX_CSV}' in {self.parent_path}."
            )

        # Validate columns
        self.required_cols = [
            "recording_id", "file_name", "channel_name", "channel_user",
            "block_index", "animal_id",
        ]
        missing = [c for c in self.required_cols if c not in self.index.columns]
        if missing:
            raise ValueError(f"Index file missing required columns: {missing}")

        # Derived constants
        self.win_cols = int(TARGET_FS * WINDOW_SEC)
        self.chunk_windows = 128  # windows per append

        # Expected channel order
        self.expected_order = [s.strip() for s in (expected_order or []) if str(s).strip()]
        self.expected_order_lc = [s.lower() for s in self.expected_order]

    # ---------------------- utilities ---------------------- #
    def _find_channel_index(self, fobj, exact_name: str) -> int:
        """Return 0-based ADI channel index by exact channel name; raise if not found."""
        for i in range(fobj.n_channels):
            if str(fobj.channels[i].name) == str(exact_name):
                return i
        raise ValueError(f"Channel name not found in ADI: {exact_name}")

    # --------------------- quick probe --------------------- #
    def quick_probe(self):
        """Quickly check that each file can be opened and a small portion read."""
        checked = set()
        for file_name in self.index["file_name"].unique():
            if file_name in checked:
                continue
            checked.add(file_name)
            path = os.path.join(self.labchart_path, file_name)
            if not os.path.exists(path):
                print(f"---> Missing LabChart file: {path}")
                continue
            try:
                f = adi.read_file(path)
                # Try first channel, first block (if present)
                _ = f.channels[0].get_data(1, start_sample=1, stop_sample=10)
                print(f"Probe OK: {file_name}")
            except Exception as e:
                print(f"Probe FAILED: {file_name} ({e})")

    # ---------------------- main run ----------------------- #
    def run(self):
        """Run the conversion over all `recording_id` groups (one H5 per group)."""
        groups = self.index.groupby("recording_id", dropna=False)
        for rec_id, df in tqdm(groups, desc="Converting to H5"):
            file_name = df["file_name"].iloc[0]
            source_path = os.path.join(self.labchart_path, str(file_name))
            if not os.path.exists(source_path):
                print(f"---> Missing LabChart file; skipping: {source_path}")
                continue
            try:
                self._process_recording(source_path, file_name, rec_id, df)
            except Exception as e:
                print(f"!!! Error processing (file={file_name}, rec_id={rec_id}): {e}")
        print("---> Conversion complete.")

    # ------------------ per-recording work ----------------- #
    def _process_recording(self, source_path: str, file_name: str, rec_id: str, df: pd.DataFrame):
        """Convert one recording group into an H5 file, enforcing expected order & reading only those channels.

        Steps
        -----
        - Enforce expected channel order (if provided) using `channel_user`.
        - Map those rows to exact ADI channel indices and read **only** those channels.
        - Per-channel integer decimation to ~100 Hz (within ±2% tolerance), zero‑phase.
        - Align to shortest channel, window into 5 s (500 cols), and write compressed H5.
        """
        f = adi.read_file(source_path)
        block_idx = int(df["block_index"].iloc[0])  # 0-based from CSV

        # Enforce expected channel set/order (by channel_user)
        if self.expected_order:
            cu = df["channel_user"].astype(str)
            cu_lc = cu.str.lower()
            exp_set = set(self.expected_order_lc)

            missing = [lbl for lbl in self.expected_order_lc if (cu_lc == lbl).sum() == 0]
            dups    = [lbl for lbl in self.expected_order_lc if (cu_lc == lbl).sum() > 1]
            extra   = sorted(set(cu_lc.unique()) - exp_set)
            if missing or dups or extra:
                print(
                    f"  Skip {rec_id}: channel set mismatch. "
                    f"Missing={missing or []}, Duplicates={dups or []}, Unexpected={extra or []}"
                )
                return

            # Keep only expected labels and order by expected order
            order_map = {lbl: i for i, lbl in enumerate(self.expected_order_lc)}
            df = df[cu_lc.isin(exp_set)].sort_values(
                by="channel_user",
                key=lambda s: s.astype(str).str.lower().map(order_map)
            ).copy()
        else:
            # Fallback: stable order by channel_user then channel_name
            df = df.sort_values(by=["channel_user", "channel_name"]).copy()

        ch_names    = df["channel_name"].tolist()    # exact ADI names
        user_labels = df["channel_user"].tolist()

        # Map to ADI indices & gather per-channel fs and block length
        ch_indices = []
        fs_list    = []
        n_list     = []
        for nm in ch_names:
            idx = self._find_channel_index(f, nm)
            ch_indices.append(idx)
            fs_list.append(float(f.channels[idx].fs[block_idx]))
            n_list.append(int(f.channels[idx].n_samples[block_idx]))

        # Duration cutoff (use shortest across selected channels)
        block_len = int(min(n_list))
        fs0 = fs_list[0]
        duration = block_len / fs0
        if duration < MIN_BLOCK_SEC:
            print(f"  Block {block_idx} too short ({duration:.1f}s); skipping {rec_id}.")
            return

        # Prepare output file (sized exactly to selected channels)
        out_name = f"{rec_id}.h5"
        out_path = os.path.join(self.h5_path, out_name)
        with tables.open_file(out_path, mode="w") as h5:
            earr = h5.create_earray(
                where=h5.root,
                name="data",
                atom=tables.Float64Atom(),
                shape=(0, self.win_cols, len(ch_names)),
                chunkshape=(self.chunk_windows, self.win_cols, len(ch_names)),
                filters=tables.Filters(complevel=5, complib="blosc"),
            )

            # Metadata
            h5.root._v_attrs.recording_id = str(rec_id)
            h5.root._v_attrs.source_file = str(file_name)
            h5.root._v_attrs.block_index_0based = block_idx
            h5.root._v_attrs.target_fs = float(TARGET_FS)
            h5.root._v_attrs.window_sec = float(WINDOW_SEC)
            h5.root._v_attrs.window_cols = int(self.win_cols)
            h5.root._v_attrs.channel_names = ch_names
            h5.root._v_attrs.channel_user = user_labels

            decimated = []
            down_factors = []
            eff_fs_list = []
            orig_fs_list = []

            # Read ONLY the selected channels for THIS recording_id
            for idx, fs, n in zip(ch_indices, fs_list, n_list):
                if n <= 0:
                    print(f"  Skip {rec_id}: empty channel data for block {block_idx+1}.")
                    return

                x = f.channels[idx].get_data(block_idx + 1, start_sample=1, stop_sample=n)  # ADI is 1-based
                x = np.asarray(x, dtype=np.float64)

                d = int(round(fs / TARGET_FS))
                if d < 1:
                    print(f"  Skip {rec_id}: TARGET_FS exceeds source fs {fs}.")
                    return
                eff = fs / d
                if abs(eff - TARGET_FS) > 0.02 * TARGET_FS:
                    print(f"  Skip {rec_id}: fs {fs} / {d} = {eff:.2f} Hz (not within 2% of {TARGET_FS}).")
                    return

                trim = (len(x) // d) * d
                x = x[:trim]
                x_dec = signal.decimate(x, d, zero_phase=True)  # zero-phase: forward+reverse filtering, ~0° net phase
                decimated.append(x_dec)
                down_factors.append(int(d))
                eff_fs_list.append(float(eff))
                orig_fs_list.append(float(fs))

            # Align to shortest, window, append
            L = min(map(len, decimated))
            decimated = [d[:L] for d in decimated]
            X = np.stack(decimated, axis=1)  # (samples, n_channels)

            drop = L % self.win_cols
            if drop:
                X = X[:-drop, :]
                L -= drop
            if L <= 0:
                print(f"  Skip {rec_id}: insufficient samples after alignment.")
                return

            rows = L // self.win_cols
            X = X[: rows * self.win_cols, :].reshape(rows, self.win_cols, len(ch_names))
            earr.append(X)

            # Metadata continued
            h5.root._v_attrs.n_windows = int(rows)
            h5.root._v_attrs.original_fs_list = orig_fs_list
            h5.root._v_attrs.downsample_factors = down_factors
            h5.root._v_attrs.effective_fs_list = eff_fs_list

        print(f"  Wrote {out_name} with {int(rows)} windows @ {TARGET_FS} Hz")


if __name__ == "__main__":
    parent_path = input("Enter parent path where labchart_data and file_index.csv reside: ").strip()
    expected_str = input("Enter expected channel order (comma-separated, e.g., vHPC,EMG,FC): ").strip()
    expected_order = [s.strip() for s in expected_str.split(',') if s.strip()]

    conv = IndexToH5Converter(parent_path, expected_order=expected_order)

    # Probe before running full conversion
    print("---> Probing files before conversion...")
    conv.quick_probe()

    # Save run properties
    props = {
        "parent_path": conv.parent_path,
        "labchart_path": conv.labchart_path,
        "h5_path": conv.h5_path,
        "index_file": os.path.basename(conv.index_file),
        "target_fs": TARGET_FS,
        "window_sec": WINDOW_SEC,
        "min_block_sec": MIN_BLOCK_SEC,
        "expected_order": expected_order,
    }
    with open(os.path.join(conv.h5_path, "conversion_properties.json"), "w") as f:
        json.dump(props, f, indent=2)

    conv.run()

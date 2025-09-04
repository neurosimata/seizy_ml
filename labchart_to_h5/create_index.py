# -*- coding: utf-8 -*-
"""
Summarize LabChart (.adicht) files using the `adi` reader.

Per file, per channel, per block, records:
- file_name
- sampling_rate (Hz)
- block_duration (seconds)
- channel_name
- block_index  (0-based here; recording_id uses 1-based)
- animal_id    (parsed from channel_name via split("-")[1])
- recording_id (unique per file_name + animal_id + block, e.g., "<file>_an<id>_block1")

Extras:
- channel_user : canonical name mapped from user-provided expected channels via substring match (case-insensitive)
- Optional dropping of rows whose channel_name contains any user-provided keywords (e.g., "bio", "drop", "empty")
- Final validation: each recording_id must contain exactly the expected channels; otherwise print details and raise ValueError
"""

import os
import sys
import numpy as np
import pandas as pd
import adi

def summarize_file(path: str) -> pd.DataFrame:
    """
    Summarize a single .adicht file into per-channel, per-block rows.

    Parameters
    ----------
    path : str
        Full path to the .adicht file.

    Returns
    -------
    pd.DataFrame
        One row per (channel, block) with columns:
        ["file_name", "sampling_rate", "block_duration", "channel_name", "block_index"]
    """
    
    columns = ["file_name", "sampling_rate", "block_duration",
               "channel_name", "block_index"]
    
    try:
        adi_obj = adi.read_file(path)
    except Exception as e:
        print(f"--> Skipping unreadable file: {path} ({e})")
        return pd.DataFrame(columns=columns)
    
    rows = []
    file_name = os.path.basename(path)

    # Iterate over channels
    for ch_idx in range(adi_obj.n_channels):
        channel = adi_obj.channels[ch_idx]
        n_blocks = len(channel.n_samples)

        # Iterate over blocks
        for block_idx in range(n_blocks):
            n_samples = int(channel.n_samples[block_idx])

            # Determine sampling rate (prefer fs[], else use tick_dt)
            if hasattr(channel, "fs") and len(channel.fs) > block_idx:
                fs = float(channel.fs[block_idx])
            else:
                fs = 1.0 / float(channel.tick_dt[block_idx])

            # Duration in seconds
            duration = n_samples / fs if fs > 0 else np.nan

            # append data
            rows.append((file_name, fs, duration, channel.name, block_idx,))

    return  pd.DataFrame(rows, columns=columns)


def summarize_path(path: str) -> pd.DataFrame:
    """
    Summarize all .adicht files under a directory, or one file if path is a file.
    """
    file_paths = []
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for fn in files:
                if fn.lower().endswith(".adicht"):
                    file_paths.append(os.path.join(root, fn))
    elif os.path.isfile(path) and path.lower().endswith(".adicht"):
        file_paths = [path]

    dfs = [summarize_file(fp) for fp in file_paths]
    dfs = [d for d in dfs if not d.empty]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def parse_animal_id(channel_name: str) -> str:
    """
    Parse animal_id from channel_name by hyphen split.
    Example '-4641-m_wt_EMG' -> '4641'
    """
    parts = str(channel_name).split("-")
    return parts[1] if len(parts) > 1 else ""


def parse_channel_label(channel_name: str) -> str:
    """
    Extract the channel label token used for validation (e.g., BIO, vHPC, EMG, FC).
    Example '-4641-m_wt_EMG' -> 'EMG'
    """
    return str(channel_name).split("_")[-1]


def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add animal_id and block-aware recording_id (no validation here).
    """
    if df.empty:
        return df

    df["animal_id"] = df["channel_name"].map(parse_animal_id)
    base_names = df["file_name"].map(lambda x: os.path.splitext(x)[0])
    # recording_id uses 1-based block index for clarity
    df["recording_id"] = (
        base_names + "_an" + df["animal_id"] + "_block" + (1 + df["block_index"]).astype(str)
    )
    return df


def map_channels(df: pd.DataFrame, expected_channels: list, new_col: str = "channel_user") -> pd.DataFrame:
    """
    For each row, if any expected channel name appears in channel_name (case-insensitive),
    assign that canonical name to `new_col`; otherwise set to 'no_match'.
    Keeps 'no_match' (for debugging/inspection).
    """
    if df.empty:
        df[new_col] = []
        return df

    df[new_col] = "no_match"
    cname = df["channel_name"].astype(str)

    # preserve user order: first pattern that matches claims the row
    for label in expected_channels:
        mask = cname.str.contains(label, case=False, na=False)
        df.loc[(df[new_col] == "no_match") & mask, new_col] = label

    return df


def filter_drop_keywords(df: pd.DataFrame, drop_keywords: list, name_col: str = "channel_name") -> pd.DataFrame:
    """
    Remove rows whose channel_name contains any of the given keywords (case-insensitive).
    """
    if df.empty or not drop_keywords:
        return df

    name = df[name_col].astype(str)
    drop_mask = pd.Series(False, index=df.index)
    for kw in drop_keywords:
        kw = (kw or "").strip()
        if kw:
            drop_mask |= name.str.contains(kw, case=False, na=False)

    return df[~drop_mask].copy()


def check_blocks_have_channels(df: pd.DataFrame, expected_channels: list, mapped_col: str = "channel_user") -> None:
    """
    Validate that every recording_id group has exactly the expected channels.
    Prints detailed issues and raises ValueError on failure (no sys.exit).
    """
    if df.empty:
        raise ValueError("No data to validate after mapping/filtering.")

    exp = [s.lower() for s in expected_channels]
    exp_set = set(exp)
    problems = []

    # Use recording_id directly (already encodes file + animal + block1-based)
    for rid, g in df.groupby("recording_id"):
        found = [s.lower() for s in g[mapped_col].astype(str).tolist()]
        counts = pd.Series(found).value_counts()

        missing = sorted([c for c in exp if counts.get(c, 0) == 0])
        dups    = sorted([c for c in exp if counts.get(c, 0) > 1])
        extra   = sorted([c for c in counts.index if c not in exp_set and c != "no_match"])
        no_match_cnt = int(counts.get("no_match", 0))

        if missing or dups or extra or no_match_cnt > 0:
            msg = [f"[{rid}]"]
            if missing:      msg.append(f"Missing: {missing}")
            if dups:         msg.append(f"Duplicate expected labels: {dups}")
            if extra:        msg.append(f"Unexpected labels: {extra}")
            if no_match_cnt: msg.append(f"Rows with no_match: {no_match_cnt}")
            problems.append(" ".join(msg))

    if problems:
        print("--> Channel-set validation FAILED for the following recording_ids:")
        for p in problems:
            print("   -", p)
        raise ValueError("Channel-set validation failed. See messages above.")


if __name__ == "__main__":
    # Get path from user
    parent_path = input("Enter parent path where labchart_data resides: ").strip()

    # Canonical channel labels to KEEP (substring match, case-insensitive)
    # expected_channels = ["vHPC", "FC"]  # e.g., ["BIO", "vHPC", "EMG", "FC"]
    channel_input = input("Enter expetected channels: ").strip()
    expected_channels = [s.strip() for s in channel_input.split(",") if s.strip()] if channel_input else []

    # Optional: keywords to DROP entirely (e.g., 'bio', 'drop', 'empty')
    drop_input = input("Enter comma-separated channel keywords to drop (optional): ").strip()
    drop_keywords = [s.strip() for s in drop_input.split(",") if s.strip()] if drop_input else []

    # Basic path checks
    if not os.path.exists(parent_path):
        print(f"--> Cannot find input path: {parent_path}")
        sys.exit(1)

    labchart_path = os.path.join(parent_path, "labchart_data")
    if not os.path.isdir(labchart_path):
        print(f"--> Cannot find folder: {labchart_path}")
        sys.exit(1)

    # Build raw summary from .adicht files
    df = summarize_path(labchart_path)
    if df.empty:
        print("--> No .adicht files found.")
        sys.exit(0)

    # 1) Enrich with animal_id and block-aware recording_id
    df = enrich_df(df)

    # 2) OPTIONAL: drop rows whose channel_name contains any of the user-specified keywords
    df = filter_drop_keywords(df, drop_keywords, name_col="channel_name")

    # 3) Map messy channel_name → user canonical labels via substring search (case-insensitive)
    #    NOTE: retains 'no_match' for debugging visibility
    df = map_channels(df, expected_channels, new_col="channel_user")

    # 4) Validate that each unique recording_id has exactly the expected channels
    try:
        check_blocks_have_channels(df, expected_channels, mapped_col="channel_user")
    except ValueError as e:
        # Still save the debug CSV so users can inspect 'no_match' rows and fix naming/filters
        debug_csv = os.path.join(parent_path, "file_index_debug.csv")
        df.to_csv(debug_csv, index=False)
        print(f"--> Wrote debug summary (with 'no_match') to: {debug_csv}")
        raise

    # Save final results
    output_csv = os.path.join(parent_path, "file_index.csv")
    df.to_csv(output_csv, index=False)
    print(f"--> Wrote summary with mapped channels and recording IDs to: {output_csv}")
    with pd.option_context("display.max_rows", 20, "display.width", 160):
        print(df.head().to_string(index=False))

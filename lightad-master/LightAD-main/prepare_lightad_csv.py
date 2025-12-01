#!/usr/bin/env python3
"""
Prepare CSV/TXT data for LightAD anomaly detection.

This script helps convert raw log data into formats suitable for LightAD processing.

Features:
- Parse structured log CSV files (like HDFS logs from Drain/Spell)
- Extract event sequences per machine/block
- Handle timestamp columns
- Support label extraction

Example usages:
  - Basic HDFS structured log:
      python prepare_lightad_csv.py --input HDFS.log_structured.csv \
        --output hdfs_prepared --format hdfs --block-col BlockId --event-col EventTemplate

  - Supercomputer logs:
      python prepare_lightad_csv.py --input BGL.log_structured.csv \
        --output bgl_prepared --format super --machine-col Node --event-col EventTemplate

Notes:
- For HDFS: Expects BlockId (or similar) to group events
- For Supercomputer logs: Expects machine/node identifier
- Labels should be in format: "-" for normal, "+" or other for anomaly
"""
import argparse
import sys
from typing import Optional, List
import pandas as pd
import os


def prepare_hdfs_data(
    input_file: str,
    output_file: str,
    block_col: str = "BlockId",
    event_col: str = "EventTemplate",
    label_col: Optional[str] = None,
    delimiter: str = ",",
):
    """
    Prepare HDFS structured log data.
    
    HDFS logs are typically grouped by BlockId, where each block contains
    a sequence of events. This function:
    1. Groups events by block
    2. Extracts event sequences
    3. Optionally maps labels
    """
    print(f"Loading HDFS data from {input_file}...")
    df = pd.read_csv(input_file, sep=delimiter)
    
    # Check required columns
    if block_col not in df.columns:
        raise ValueError(f"Block column '{block_col}' not found. Available: {list(df.columns)}")
    if event_col not in df.columns:
        raise ValueError(f"Event column '{event_col}' not found. Available: {list(df.columns)}")
    
    # Group by block
    grouped = df.groupby(block_col, sort=False)
    
    # Build output format: one line per block, space-separated events
    lines = []
    labels = {}
    
    for block_id, group in grouped:
        events = group[event_col].tolist()
        # Remove digits/numbers from events (preprocessing)
        events_clean = []
        for event in events:
            # Convert to string and clean
            event_str = str(event).strip()
            # Remove tokens containing digits
            tokens = event_str.split()
            clean_tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
            if clean_tokens:
                events_clean.append(" ".join(clean_tokens))
        
        # Join events with space
        line = " ".join(events_clean) if events_clean else "EMPTY"
        lines.append(f"{block_id}\t{line}\n")
        
        # Extract label if available
        if label_col and label_col in df.columns:
            # Assume all events in same block have same label
            label = group[label_col].iloc[0]
            # Normalize label: "-" for normal, "+" for anomaly
            if pd.notna(label):
                if str(label).lower() in ["normal", "0", "-"]:
                    labels[str(block_id)] = "Normal"
                else:
                    labels[str(block_id)] = "Anomaly"
    
    # Write output
    print(f"Writing {len(lines)} blocks to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    # Write labels if available
    if labels:
        label_file = output_file + "_labels.csv"
        print(f"Writing labels to {label_file}...")
        with open(label_file, "w", encoding="utf-8") as f:
            f.write("BlockId,Label\n")
            for block_id, label in labels.items():
                f.write(f"{block_id},{label}\n")
    
    print(f"Prepared {len(lines)} sequences")
    print(f"Sample line: {lines[0][:100]}...")


def prepare_super_data(
    input_file: str,
    output_file: str,
    machine_col: str = "Node",
    event_col: str = "EventTemplate",
    label_col: Optional[str] = "Label",
    delimiter: str = ",",
    nrows: Optional[int] = None,
):
    """
    Prepare supercomputer log data (BGL, Spirit, TBird, Liberty).
    
    Supercomputer logs are typically chronological, with each line having:
    - Timestamp
    - Machine/Node ID
    - Event/Template
    - Optional label
    """
    print(f"Loading supercomputer data from {input_file}...")
    df = pd.read_csv(input_file, sep=delimiter, nrows=nrows)
    
    # Check required columns
    if machine_col not in df.columns:
        raise ValueError(f"Machine column '{machine_col}' not found. Available: {list(df.columns)}")
    if event_col not in df.columns:
        raise ValueError(f"Event column '{event_col}' not found. Available: {list(df.columns)}")
    
    # Process each row
    lines = []
    for idx, row in df.iterrows():
        # Extract label
        if label_col and label_col in df.columns:
            label = row[label_col]
            if pd.notna(label):
                if str(label).lower() in ["normal", "0", "-"]:
                    label_str = "-"
                else:
                    label_str = "+"
            else:
                label_str = "-"
        else:
            label_str = "-"
        
        # Extract event (remove digits)
        event = str(row[event_col]).strip()
        tokens = event.split()
        clean_tokens = [t for t in tokens if not any(c.isdigit() for c in t) and any(c.isalpha() for c in t)]
        event_clean = " ".join(clean_tokens) if clean_tokens else "EMPTY"
        
        # Format: label + event tokens
        # This matches the format expected by LightAD for supercomputer logs
        line = f"{label_str} " + " " * 7 + event_clean + "\n"
        lines.append(line)
        
        if (idx + 1) % 100000 == 0:
            print(f"Processed {idx + 1} rows...")
    
    # Write output
    print(f"Writing {len(lines)} lines to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    print(f"Prepared {len(lines)} log lines")
    print(f"Sample line: {lines[0][:100]}...")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare data for LightAD")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output file (without extension)")
    parser.add_argument(
        "--format",
        required=True,
        choices=["hdfs", "super"],
        help="Data format: 'hdfs' for HDFS logs, 'super' for supercomputer logs",
    )
    
    # Column specification
    parser.add_argument("--block-col", default="BlockId", help="Block ID column (for HDFS format)")
    parser.add_argument("--machine-col", default="Node", help="Machine/Node column (for super format)")
    parser.add_argument("--event-col", default="EventTemplate", help="Event/Template column")
    parser.add_argument("--label-col", default="Label", help="Label column (optional)")
    
    # Misc
    parser.add_argument("--delimiter", default=",", help="CSV delimiter")
    parser.add_argument("--nrows", type=int, help="Load only first n rows (for testing)")
    
    args = parser.parse_args(argv)
    
    try:
        if args.format == "hdfs":
            prepare_hdfs_data(
                input_file=args.input,
                output_file=args.output,
                block_col=args.block_col,
                event_col=args.event_col,
                label_col=args.label_col,
                delimiter=args.delimiter,
            )
        else:  # super
            prepare_super_data(
                input_file=args.input,
                output_file=args.output,
                machine_col=args.machine_col,
                event_col=args.event_col,
                label_col=args.label_col,
                delimiter=args.delimiter,
                nrows=args.nrows,
            )
        
        print("\n✓ Data preparation completed successfully!")
        print(f"✓ Output file: {args.output}")
        if args.format == "hdfs":
            print(f"✓ Label file: {args.output}_labels.csv (if labels were present)")
        print(f"\nNext steps:")
        print(f"1. Place the output file in datasets/original_datasets/ directory")
        print(f"2. Run: lightad_preprocess(dataset='{os.path.basename(args.output)}')")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


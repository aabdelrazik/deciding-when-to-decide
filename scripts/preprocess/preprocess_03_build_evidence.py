"""Step 3 of preprocessing: per subject trial records and evidence dictionaries.

Takes the reward augmented trial table and produces the files the fitting code
actually reads:

    behdat_preprocessed.pkl                     per subject trial records
    all_subject_evidence_dicts.pkl              draw sequences, cut at the
                                                subject's decision
    all_subject_evidence_dicts_full_sequence.pkl  the same games run on to the
                                                deadline
    plus _short and _long variants of both

Each game is a list of rows

    [draw, cumulative_yellow, cumulative_blue, action, outcome]

with cumulative counts, not per draw counts, and actions 0 commit yellow,
1 commit blue, 2 wait. Outcome is +2 correct, -2 incorrect, -1 the deadline was
reached with no decision.

The plain file stops where the subject decided, so its length is that subject's
draw count. The full sequence file continues to the predetermined endpoint and
parks the action on the remaining rows, which is what the scoring analyses
replay.

Run scripts/preprocess/preprocess_02_add_reward.py first.

Usage (from the repository root):
    python3 scripts/preprocess/preprocess_03_build_evidence.py
"""
import argparse
import contextlib
import io
import os
import sys

import pandas as pd

ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

from src.utils.preprocessing import (  # noqa: E402
    analyze_all_subjects_behavior,
    get_subject_data,
    format_human_data_for_modeling,
    format_human_data_for_modeling_full_sequence,
)

DEFAULT_DATA_DIR = os.path.join(ROOT, "data", "TrHu_NHB_light", "data_MEG")


def build_evidence(processed, formatter):
    """Run one formatter over every subject.

    Each formatter returns (summary, short, long, both). We keep the three
    evidence dictionaries and drop the summary.
    """
    both, short, long_ = {}, {}, {}
    for subject_id in processed["userID"].unique():
        subject_df = get_subject_data(processed, subject_id)
        if subject_df is None or subject_df.empty:
            print(f"  skipping {subject_id}, no data")
            continue
        # the extracted notebook code prints each game as it goes
        with contextlib.redirect_stdout(io.StringIO()):
            _, ev_short, ev_long, ev_both = formatter(subject_df, subject_id)
        both[subject_id] = ev_both
        short[subject_id] = ev_short
        long_[subject_id] = ev_long
    return (pd.DataFrame(both).T, pd.DataFrame(short).T, pd.DataFrame(long_).T)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    reward_path = os.path.join(args.data_dir, "behdat_reward.pkl")
    if not os.path.exists(reward_path):
        sys.exit(f"{reward_path} not found. Run scripts/preprocess/preprocess_02_add_reward.py "
                 "first; the reward column is required.")

    raw = pd.read_pickle(reward_path)
    with contextlib.redirect_stdout(io.StringIO()):
        processed, n_equal_evidence, _ = analyze_all_subjects_behavior(raw)
    out = os.path.join(args.data_dir, "behdat_preprocessed.pkl")
    processed.to_pickle(out)
    print(f"{processed['userID'].nunique()} subjects "
          f"({n_equal_evidence} trials with equal evidence) -> {out}")

    for suffix, formatter in (("", format_human_data_for_modeling),
                              ("_full_sequence",
                               format_human_data_for_modeling_full_sequence)):
        both, short, long_ = build_evidence(processed, formatter)
        for tag, frame in (("", both), ("_short", short), ("_long", long_)):
            dest = os.path.join(
                args.data_dir, f"all_subject_evidence_dicts{suffix}{tag}.pkl")
            frame.to_pickle(dest)
            print(f"  {len(frame)} subjects -> {os.path.basename(dest)}")


if __name__ == "__main__":
    main()

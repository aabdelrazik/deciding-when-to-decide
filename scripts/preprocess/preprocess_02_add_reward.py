"""Step 2 of preprocessing: recover the per game reward. Do not skip this.

behdat.mat does not carry the reward a participant actually received. That
information lives in a second file, sNdraws8.mat, which holds the same sessions
with a `postfeedback` column. This script matches the two datasets subject by
subject and derives a `reward` column from it.

Everything in the manuscript that is measured in points depends on this step:
the cost of departing from optimality, the normative benchmark, and every
accuracy figure. Running the rest of the pipeline without it produces results
that look plausible and are wrong, so it runs before
preprocess_03_build_evidence.py and the README says so.

How the matching works. The two files do not share subject identifiers, so
subjects are aligned by their behaviour: for each subject in behdat.pkl we look
for the subject in sNdraws8.mat whose trial, termination, choiceTrial,
currEvLeft and currEvRight columns start with exactly the same rows. behdat is a
prefix of the feedback dataset, so a prefix comparison is the right test. The
match is required to be unique and the script fails loudly if any subject is
unmatched.

How the reward is derived. Feedback for a game is shown at the start of the
following game, so a game's reward is the `postfeedback` value of the next
(block, game) pair in chronological order. Games are sorted on (block, game)
because game numbering restarts within each block. The final game of a session
has no successor and falls back to its own last non-null postfeedback.

Input:  <data_dir>/behdat.pkl  (from preprocess_01_convert_mat.py)
        <data_dir>/sNdraws8.mat
Output: <data_dir>/behdat_reward.pkl

Usage (from the repository root):
    python3 scripts/preprocess/preprocess_02_add_reward.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.io

ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_DATA_DIR = os.path.join(ROOT, "data", "TrHu_NHB_light", "data_MEG")

# Columns used to align the two datasets. They are behavioural rather than
# identifying, which is the point: the files share no subject IDs.
MATCH_COLS = ["trial", "termination", "choiceTrial", "currEvLeft", "currEvRight"]


def as_column_names(descr):
    if isinstance(descr, np.ndarray):
        return [c.decode("utf-8") if isinstance(c, (bytes, np.bytes_)) else c
                for c in descr]
    if isinstance(descr, (bytes, np.bytes_)):
        return [descr.decode("utf-8")]
    return [descr]


def load_feedback(mat_path):
    """Read sNdraws8.mat. Subjects get positional ids; real ones come from the
    behavioural match below."""
    mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    frames = []
    for subject in np.atleast_1d(mat["s"].sub):
        beh = subject.beh
        frames.append(pd.DataFrame(np.atleast_2d(beh.dat),
                                   columns=as_column_names(beh.descr)))
    return pd.DataFrame({"userID": range(len(frames)), "data": frames})


def build_mapping(feedback, original):
    """original userID -> feedback positional id, matched on behaviour."""
    mapping = {}
    for i in range(len(original)):
        orig = original["data"].iloc[i][MATCH_COLS].reset_index(drop=True)
        user_id = original["userID"].iloc[i]
        match = None
        for j in range(len(feedback)):
            cand = feedback["data"].iloc[j][MATCH_COLS].reset_index(drop=True)
            if len(cand) < len(orig):
                continue
            if orig.equals(cand.iloc[:len(orig)]):
                match = feedback["userID"].iloc[j]
                break
        if match is None:
            sys.exit(f"no feedback subject matches behdat subject {user_id}. "
                     "The two files may be from different sessions.")
        mapping[user_id] = match
    return mapping


def attach_postfeedback(original, feedback, mapping):
    for i in range(len(original)):
        user_id = original["userID"].iloc[i]
        matched = feedback[feedback["userID"] == mapping[user_id]].iloc[0]
        col = matched["data"]["postfeedback"].reset_index(drop=True)
        data = original.at[i, "data"]
        # behdat is a prefix of the feedback session, so trim to its length
        data["postfeedback"] = col.iloc[:len(data)].values
        original.at[i, "data"] = data
    return original


def add_reward_column(data):
    """A game's reward is the postfeedback of the next game in time."""
    data = data.copy()
    data["reward"] = np.nan

    games = sorted(data[["block", "game"]].drop_duplicates()
                   .itertuples(index=False, name=None))
    for i, (block_id, game_id) in enumerate(games):
        mask = (data["block"] == block_id) & (data["game"] == game_id)
        if i + 1 < len(games):
            nb, ng = games[i + 1]
            reward = data.loc[(data["block"] == nb) & (data["game"] == ng),
                              "postfeedback"].iloc[0]
        else:
            pf = data.loc[mask, "postfeedback"].dropna()
            reward = pf.iloc[-1] if not pf.empty else np.nan
        data.loc[mask, "reward"] = reward
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    beh_path = os.path.join(args.data_dir, "behdat.pkl")
    fb_path = os.path.join(args.data_dir, "sNdraws8.mat")
    for p, hint in ((beh_path, "run scripts/preprocess/preprocess_01_convert_mat.py first"),
                    (fb_path, "download it with the rest of the dataset from "
                              "https://osf.io/fks97/")):
        if not os.path.exists(p):
            sys.exit(f"{p} not found. {hint}")

    original = pd.read_pickle(beh_path)
    feedback = load_feedback(fb_path)
    print(f"{len(original)} subjects in behdat, {len(feedback)} in the feedback file")

    mapping = build_mapping(feedback, original)
    print(f"matched all {len(mapping)} subjects on {', '.join(MATCH_COLS)}")

    original = attach_postfeedback(original, feedback, mapping)
    for i in range(len(original)):
        original.at[i, "data"] = add_reward_column(original["data"].iloc[i])

    out = os.path.join(args.data_dir, "behdat_reward.pkl")
    original.to_pickle(out)

    filled = int(sum(d["reward"].notna().sum() for d in original["data"]))
    total = int(sum(len(d) for d in original["data"]))
    print(f"reward present on {filled} of {total} rows")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

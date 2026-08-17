"""Step 1 of preprocessing: turn the downloaded behdat.mat into behdat.pkl.

The OSF download ships MATLAB structs. This reads behdat.mat with scipy, so no
MATLAB is needed, and writes one pandas DataFrame per subject into a single
pickle that every later step consumes.

behdat.mat holds a struct array `s`, one entry per subject, each with a `beh`
field carrying `descr` (the column names) and `dat` (the trial by column
matrix), plus a `userID`.

Output: <data_dir>/behdat.pkl, a DataFrame with columns
    userID : subject identifier
    data   : that subject's trial level DataFrame

Run this before preprocess_02_add_reward.py.

Usage (from the repository root):
    python3 scripts/preprocess/preprocess_01_convert_mat.py
    python3 scripts/preprocess/preprocess_01_convert_mat.py --data-dir /path/to/data_MEG
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


def as_column_names(descr):
    """Column names arrive as bytes, str, or an array of either."""
    if isinstance(descr, np.ndarray):
        return [c.decode("utf-8") if isinstance(c, (bytes, np.bytes_)) else c
                for c in descr]
    if isinstance(descr, (bytes, np.bytes_)):
        return [descr.decode("utf-8")]
    return [descr]


def convert(mat_path):
    mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    subjects = np.atleast_1d(mat["s"])

    frames, user_ids = [], []
    for subject in subjects:
        beh = subject.beh
        user_id = subject.userID
        if isinstance(user_id, bytes):
            user_id = user_id.decode("utf-8")
        frames.append(pd.DataFrame(np.atleast_2d(beh.dat),
                                   columns=as_column_names(beh.descr)))
        user_ids.append(user_id)
    return pd.DataFrame({"userID": user_ids, "data": frames})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                    help="directory holding behdat.mat (default: %(default)s)")
    args = ap.parse_args()

    mat_path = os.path.join(args.data_dir, "behdat.mat")
    if not os.path.exists(mat_path):
        sys.exit(f"behdat.mat not found at {mat_path}\n"
                 "Download the dataset from https://osf.io/fks97/ and unpack it "
                 "into data/TrHu_NHB_light/ first. See the README.")

    df = convert(mat_path)
    out = os.path.join(args.data_dir, "behdat.pkl")
    df.to_pickle(out)

    n_trials = int(sum(len(d) for d in df["data"]))
    print(f"{len(df)} subjects, {n_trials} trials")
    print(f"columns: {list(df['data'].iloc[0].columns)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""
Profiling script for scikit_pierre/tradeoff module.
Run from PycharmProjects parent directory.
"""
import cProfile
import pstats
import tracemalloc
import io
import random
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scikit_pierre.tradeoff.calibration import LinearCalibration, LogarithmBias

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_USERS = 20
N_ITEMS = 200
GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Adventure"]
N_PREF_PER_USER = 30
N_CAND_PER_USER = 50
LIST_SIZE = 10


def make_data():
    item_ids = list(range(1, N_ITEMS + 1))
    item_rows = []
    for iid in item_ids:
        n_genres = random.randint(1, 3)
        genres = "|".join(random.sample(GENRES, n_genres))
        item_rows.append({"ITEM_ID": iid, "GENRES": genres})
    items_df = pd.DataFrame(item_rows)

    pref_rows = []
    for uid in range(1, N_USERS + 1):
        sampled = random.sample(item_ids, N_PREF_PER_USER)
        for iid in sampled:
            pref_rows.append({"USER_ID": uid, "ITEM_ID": iid,
                               "TRANSACTION_VALUE": round(random.uniform(1, 5), 2)})
    pref_df = pd.DataFrame(pref_rows)

    cand_rows = []
    for uid in range(1, N_USERS + 1):
        sampled = random.sample(item_ids, N_CAND_PER_USER)
        for iid in sampled:
            cand_rows.append({"USER_ID": uid, "ITEM_ID": iid,
                               "TRANSACTION_VALUE": round(random.uniform(0, 1), 4)})
    cand_df = pd.DataFrame(cand_rows)

    return pref_df, cand_df, items_df


def run_linear(pref_df, cand_df, items_df):
    lc = LinearCalibration(pref_df, cand_df, items_df)
    lc.config(distribution_component="CWS", fairness_component="KL",
              relevance_component="SUM", tradeoff_weight_component="STD",
              list_size=LIST_SIZE)
    return lc.fit()


def run_logarithm_bias(pref_df, cand_df, items_df):
    lb = LogarithmBias(pref_df, cand_df, items_df)
    lb.config(distribution_component="CWS", fairness_component="KL",
              relevance_component="SUM", tradeoff_weight_component="STD",
              list_size=LIST_SIZE)
    return lb.fit()


if __name__ == "__main__":
    pref_df, cand_df, items_df = make_data()

    print("=" * 60)
    print("PROFILING: LinearCalibration")
    print("=" * 60)

    # --- tracemalloc ---
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    pr = cProfile.Profile()
    pr.enable()
    result = run_linear(pref_df, cand_df, items_df)
    pr.disable()

    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())

    top_stats = snap_after.compare_to(snap_before, "lineno")
    print("\n--- Top 10 memory allocations (LinearCalibration) ---")
    for stat in top_stats[:10]:
        print(stat)

    print("\n" + "=" * 60)
    print("PROFILING: LogarithmBias")
    print("=" * 60)

    tracemalloc.start()
    snap_before2 = tracemalloc.take_snapshot()

    pr2 = cProfile.Profile()
    pr2.enable()
    result2 = run_logarithm_bias(pref_df, cand_df, items_df)
    pr2.disable()

    snap_after2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    s2 = io.StringIO()
    ps2 = pstats.Stats(pr2, stream=s2).sort_stats("cumulative")
    ps2.print_stats(30)
    print(s2.getvalue())

    top_stats2 = snap_after2.compare_to(snap_before2, "lineno")
    print("\n--- Top 10 memory allocations (LogarithmBias) ---")
    for stat in top_stats2[:10]:
        print(stat)

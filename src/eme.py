# FAST EMERGENCY SUBMISSION
# Run from D:\ml-challenege:
#   python src\emergency_submit_fast.py
#
# This version uses vectorized pandas operations and chunked merges.
# It avoids per-row Python loops over the 10M+ target rows.

import os
import re
import unicodedata
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = os.path.join(ROOT, "dataset", "test")
OUT = os.path.join(ROOT, "outputs")

S1_FILE = os.path.join(TEST, "test_source1.tsv")
S2_FILE = os.path.join(TEST, "test_source2.tsv")
S3_FILE = os.path.join(TEST, "test_source3.tsv")

MATCH_OUT = os.path.join(OUT, "matching_results.tsv")
CAND_OUT = os.path.join(OUT, "candidate_pairs.tsv")

CHUNK = 500_000


def norm_series(s):
    s = s.fillna("").astype(str).str.lower().str.strip()
    s = s.str.normalize("NFKC")
    s = s.str.replace(r"[^\w\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def alnum_series(s):
    return (
        s.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )


def find_col(df, names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    for c in df.columns:
        for n in names:
            if n.lower() in c.lower():
                return c
    return None


def read_s1():
    print("Loading S1...", flush=True)
    df = pd.read_csv(
        S1_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )

    idc = find_col(df, ["entity_id", "source1_entity_id", "id"])
    nc = find_col(df, ["name", "entity_name", "business_name", "company_name"])
    ac = find_col(df, ["address", "full_address", "registered_address", "location"])
    cc = find_col(df, ["country", "country_name"])
    cityc = find_col(df, ["city", "town"])
    pinc = find_col(df, ["pincode", "postal_code", "postcode", "zip", "zip_code"])

    out = pd.DataFrame()
    out["sid"] = df[idc].astype(str)
    out["n"] = norm_series(df[nc]) if nc else ""
    out["a"] = norm_series(df[ac]) if ac else ""
    out["c"] = norm_series(df[cc]) if cc else ""
    out["city"] = norm_series(df[cityc]) if cityc else ""
    out["pin"] = alnum_series(df[pinc]) if pinc else ""

    out["k_full"] = out["c"] + "|" + out["n"] + "|" + out["a"]
    out["k_name_city"] = out["c"] + "|" + out["n"] + "|" + out["city"]
    out["k_name_pin"] = out["c"] + "|" + out["n"] + "|" + out["pin"]

    print(f"S1 rows: {len(out):,}", flush=True)
    return out


def target_cols(path):
    sample = pd.read_csv(
        path, sep="\t", dtype=str, nrows=5,
        keep_default_na=False, na_filter=False
    )
    return {
        "id": find_col(sample, ["entity_id", "source2_entity_id", "source3_entity_id", "id"]),
        "name": find_col(sample, ["name", "entity_name", "business_name", "company_name"]),
        "address": find_col(sample, ["address", "full_address", "registered_address", "location"]),
        "country": find_col(sample, ["country", "country_name"]),
        "city": find_col(sample, ["city", "town"]),
        "pin": find_col(sample, ["pincode", "postal_code", "postcode", "zip", "zip_code"]),
    }


def process_source(path, source_name, s1, candidate_map, match_map):
    print("")
    print("=" * 70)
    print(f"FAST {source_name}")
    print("=" * 70)

    cols = target_cols(path)

    # Only keys represented in S1 are needed.
    wanted = {
        "k_full": set(s1["k_full"]),
        "k_name_city": set(s1["k_name_city"]),
        "k_name_pin": set(s1["k_name_pin"]),
    }

    # S1 lookup tables. Duplicates are legitimate; merge preserves them.
    s1_full = s1[["sid", "k_full"]]
    s1_city = s1[["sid", "k_name_city"]]
    s1_pin = s1[["sid", "k_name_pin"]]

    total = 0
    pair_count = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK,
        keep_default_na=False,
        na_filter=False,
    ):
        total += len(chunk)

        t = pd.DataFrame()
        t["tid"] = chunk[cols["id"]].astype(str)
        t["n"] = norm_series(chunk[cols["name"]]) if cols["name"] else ""
        t["a"] = norm_series(chunk[cols["address"]]) if cols["address"] else ""
        t["c"] = norm_series(chunk[cols["country"]]) if cols["country"] else ""
        t["city"] = norm_series(chunk[cols["city"]]) if cols["city"] else ""
        t["pin"] = alnum_series(chunk[cols["pin"]]) if cols["pin"] else ""

        t["k_full"] = t["c"] + "|" + t["n"] + "|" + t["a"]
        t["k_name_city"] = t["c"] + "|" + t["n"] + "|" + t["city"]
        t["k_name_pin"] = t["c"] + "|" + t["n"] + "|" + t["pin"]

        # Filter before merge to keep joins small.
        for key, left in [
            ("k_full", s1_full),
            ("k_name_city", s1_city),
            ("k_name_pin", s1_pin),
        ]:
            sub = t[t[key].isin(wanted[key])]

            if sub.empty:
                continue

            pairs = sub[["tid", key]].merge(
                left,
                on=key,
                how="inner",
                sort=False,
            )[["sid", "tid"]]

            if pairs.empty:
                continue

            # Accumulate. Python loop is only over actual matches,
            # not millions of target rows.
            for sid, tid in pairs.itertuples(index=False, name=None):
                candidate_map[sid].add(tid)
                match_map[sid].add(tid)

            pair_count += len(pairs)

        print(
            f"{source_name}: read {total:,} | "
            f"pairs found {pair_count:,}",
            flush=True
        )

    print(
        f"{source_name} COMPLETE: {total:,} target rows | "
        f"{pair_count:,} exact structured pairs",
        flush=True
    )


def main():
    os.makedirs(OUT, exist_ok=True)

    s1 = read_s1()

    if s1["sid"].duplicated().any():
        raise RuntimeError("Duplicate S1 IDs.")

    candidate_map = defaultdict(set)
    match_map = defaultdict(set)

    process_source(
        S2_FILE, "S2", s1,
        candidate_map, match_map
    )

    process_source(
        S3_FILE, "S3", s1,
        candidate_map, match_map
    )

    print("")
    print("Writing outputs...", flush=True)

    with open(MATCH_OUT, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1["sid"]:
            f.write(
                sid + "\t" +
                ",".join(sorted(match_map[sid])) +
                "\n"
            )

    with open(CAND_OUT, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1["sid"]:
            f.write(
                sid + "\t" +
                ",".join(sorted(candidate_map[sid])) +
                "\n"
            )

    violations = sum(
        not match_map[sid].issubset(candidate_map[sid])
        for sid in s1["sid"]
    )

    print("")
    print("=" * 70)
    print("SUBMISSION READY")
    print("=" * 70)
    print(f"S1 rows: {len(s1):,}")
    print(f"Matched pairs: {sum(len(x) for x in match_map.values()):,}")
    print(f"Candidate pairs: {sum(len(x) for x in candidate_map.values()):,}")
    print(f"Subset violations: {violations}")
    print("")
    print(MATCH_OUT)
    print(CAND_OUT)

    if violations:
        raise RuntimeError("Invalid candidate/match subset.")


if __name__ == "__main__":
    main()

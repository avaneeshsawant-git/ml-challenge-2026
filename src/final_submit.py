# src/final_submit.py
#
# FAST FINAL SUBMISSION PIPELINE
#
# Outputs:
#   outputs/matching_results.tsv
#   outputs/candidate_pairs.tsv
#
# Uses:
#   outputs/model/xgboost_matcher.json
#   outputs/model/threshold.txt
#
# Important:
# - Country is used only as a blocking field.
# - No external lookup.
# - France is handled automatically.
# - Does NOT run the extremely slow full sparse TF-IDF matrix multiplication.
#
# Run from project root:
#   python src\final_submit.py

import os
import re
import sys
import time
import math
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd

from rapidfuzz import fuzz, process
import xgboost as xgb


# ============================================================
# CONFIG
# ============================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET = os.path.join(ROOT, "dataset")
TEST_DIR = os.path.join(DATASET, "test")

OUT_DIR = os.path.join(ROOT, "outputs")
MODEL_DIR = os.path.join(OUT_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_matcher.json")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.txt")

MATCHING_OUT = os.path.join(OUT_DIR, "matching_results.tsv")
CANDIDATE_OUT = os.path.join(OUT_DIR, "candidate_pairs.tsv")


# Number of fuzzy candidates retained from each blocking strategy.
#
# Keep this relatively small because the test set is huge.
TOP_NAME = 8
TOP_ADDRESS = 8
TOP_TOKEN = 8

# Maximum number of candidates sent to the ML model per
# source1/source2 or source1/source3 pair.
MAX_MODEL_CANDIDATES = 18

# Process source1 in batches.
BATCH_SIZE = 2000

# Number of rare/shared tokens used for inverted-index blocking.
MAX_NAME_TOKENS = 3
MAX_ADDRESS_TOKENS = 3

# Very strong deterministic matches.
EXACT_NAME_MATCH = True
EXACT_ADDRESS_MATCH = True
EXACT_SORTED_NAME_MATCH = True


# ============================================================
# HELPERS
# ============================================================

def log(msg=""):
    print(msg, flush=True)


def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).lower().strip()

    # Unicode normalize.
    x = unicodedata.normalize("NFKC", x)

    # Replace punctuation with spaces.
    x = re.sub(r"[^\w\s]", " ", x, flags=re.UNICODE)

    # Collapse whitespace.
    x = re.sub(r"\s+", " ", x).strip()

    return x


def normalize_alnum(x):
    if not x:
        return ""
    return re.sub(r"[^a-z0-9]+", "", x.lower())


def sorted_name(x):
    if not x:
        return ""
    return " ".join(sorted(x.split()))


def tokens(x):
    if not x:
        return []
    return [t for t in x.split() if len(t) >= 2]


def digit_tokens(x):
    if not x:
        return []
    return re.findall(r"\d+", x)


def safe_ratio(a, b):
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def safe_partial(a, b):
    if not a or not b:
        return 0.0
    return fuzz.partial_ratio(a, b) / 100.0


def safe_token_sort(a, b):
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


def safe_token_set(a, b):
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def jaccard(a, b):
    sa = set(tokens(a))
    sb = set(tokens(b))

    if not sa or not sb:
        return 0.0

    u = sa | sb
    if not u:
        return 0.0

    return len(sa & sb) / len(u)


def digit_overlap(a, b):
    sa = set(digit_tokens(a))
    sb = set(digit_tokens(b))

    if not sa or not sb:
        return 0.0

    return len(sa & sb) / max(1, len(sa | sb))


def length_ratio(a, b):
    if not a or not b:
        return 0.0

    la = len(a)
    lb = len(b)

    return min(la, lb) / max(la, lb)


def common_token_ratio(a, b):
    sa = set(tokens(a))
    sb = set(tokens(b))

    if not sa or not sb:
        return 0.0

    return len(sa & sb) / max(1, min(len(sa), len(sb)))


def read_tsv(path):
    log(f"Loading: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False
    )

    log(f"Rows: {len(df):,}")

    return df


def find_col(df, candidates, required=True):
    lower = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]

    # fuzzy fallback
    for col in df.columns:
        lc = col.lower()
        for c in candidates:
            if c.lower() in lc:
                return col

    if required:
        raise ValueError(
            f"Could not find column. Tried: {candidates}. "
            f"Available: {list(df.columns)}"
        )

    return None


# ============================================================
# COLUMN DETECTION
# ============================================================

def prepare_dataframe(df):
    id_col = find_col(
        df,
        [
            "entity_id",
            "source1_entity_id",
            "source2_entity_id",
            "source3_entity_id",
            "id",
        ],
    )

    name_col = find_col(
        df,
        [
            "name",
            "entity_name",
            "business_name",
            "company_name",
        ],
    )

    address_col = find_col(
        df,
        [
            "address",
            "full_address",
            "registered_address",
            "location",
        ],
        required=False,
    )

    country_col = find_col(
        df,
        [
            "country",
            "country_name",
        ],
        required=False,
    )

    city_col = find_col(
        df,
        [
            "city",
            "town",
        ],
        required=False,
    )

    state_col = find_col(
        df,
        [
            "state",
            "province",
            "region",
        ],
        required=False,
    )

    pincode_col = find_col(
        df,
        [
            "pincode",
            "postal_code",
            "postcode",
            "zip",
            "zip_code",
        ],
        required=False,
    )

    out = pd.DataFrame()

    out["entity_id"] = df[id_col].astype(str)

    out["name"] = (
        df[name_col].astype(str)
        if name_col is not None
        else ""
    )

    out["address"] = (
        df[address_col].astype(str)
        if address_col is not None
        else ""
    )

    out["country"] = (
        df[country_col].astype(str)
        if country_col is not None
        else ""
    )

    out["city"] = (
        df[city_col].astype(str)
        if city_col is not None
        else ""
    )

    out["state"] = (
        df[state_col].astype(str)
        if state_col is not None
        else ""
    )

    out["pincode"] = (
        df[pincode_col].astype(str)
        if pincode_col is not None
        else ""
    )

    out["name_norm"] = out["name"].map(normalize_text)
    out["address_norm"] = out["address"].map(normalize_text)
    out["country_norm"] = out["country"].map(normalize_text)
    out["city_norm"] = out["city"].map(normalize_text)
    out["state_norm"] = out["state"].map(normalize_text)
    out["pincode_norm"] = out["pincode"].map(normalize_alnum)

    out["name_sorted"] = out["name_norm"].map(sorted_name)

    return out


# ============================================================
# COUNTRY BLOCK
# ============================================================

def country_key(x):
    x = normalize_text(x)

    if not x:
        return "__EMPTY__"

    return x


# ============================================================
# BUILD INVERTED INDEX
# ============================================================

def build_index(df, field, max_occurrences=2000):
    """
    Inverted token index.

    Extremely common tokens are ignored because they produce
    enormous candidate lists.
    """

    temp = defaultdict(list)

    for i, value in enumerate(df[field].values):
        ts = tokens(value)

        # Use unique tokens.
        ts = list(dict.fromkeys(ts))

        for t in ts:
            if len(temp[t]) <= max_occurrences:
                temp[t].append(i)

    return dict(temp)


def choose_rare_tokens(value, index, maximum=3):
    ts = list(dict.fromkeys(tokens(value)))

    valid = []

    for t in ts:
        n = len(index.get(t, []))

        if n > 0:
            valid.append((n, t))

    valid.sort()

    return [t for _, t in valid[:maximum]]


# ============================================================
# FAST CANDIDATE RETRIEVAL
# ============================================================

def build_country_structures(target):
    structures = {}

    for country, grp in target.groupby("country_norm", sort=False):

        idx = grp.index.to_numpy()

        local = target.loc[idx]

        log(
            f"Building index: {country} "
            f"({len(local):,} targets)"
        )

        name_choices = local["name_norm"].to_dict()
        address_choices = local["address_norm"].to_dict()

        exact_name = defaultdict(list)
        exact_sorted = defaultdict(list)
        exact_address = defaultdict(list)

        for i in idx:
            n = local.at[i, "name_norm"]
            s = local.at[i, "name_sorted"]
            a = local.at[i, "address_norm"]

            if n:
                exact_name[n].append(i)

            if s:
                exact_sorted[s].append(i)

            if a:
                exact_address[a].append(i)

        name_index = build_index(
            local,
            "name_norm",
            max_occurrences=1500,
        )

        address_index = build_index(
            local,
            "address_norm",
            max_occurrences=1500,
        )

        structures[country] = {
            "indices": idx,
            "name_choices": name_choices,
            "address_choices": address_choices,
            "exact_name": exact_name,
            "exact_sorted": exact_sorted,
            "exact_address": exact_address,
            "name_index": name_index,
            "address_index": address_index,
        }

    return structures


def retrieve_candidates(q, structure):
    candidates = set()

    qname = q["name_norm"]
    qaddr = q["address_norm"]
    qsorted = q["name_sorted"]

    # --------------------------------------------------------
    # Exact blocks
    # --------------------------------------------------------

    if EXACT_NAME_MATCH and qname:
        for x in structure["exact_name"].get(qname, []):
            candidates.add(x)

    if EXACT_SORTED_NAME_MATCH and qsorted:
        for x in structure["exact_sorted"].get(qsorted, []):
            candidates.add(x)

    if EXACT_ADDRESS_MATCH and qaddr:
        for x in structure["exact_address"].get(qaddr, []):
            candidates.add(x)

    # --------------------------------------------------------
    # Token blocking
    # --------------------------------------------------------

    name_tokens = choose_rare_tokens(
        qname,
        structure["name_index"],
        MAX_NAME_TOKENS,
    )

    for t in name_tokens:
        for x in structure["name_index"].get(t, []):
            candidates.add(x)

    addr_tokens = choose_rare_tokens(
        qaddr,
        structure["address_index"],
        MAX_ADDRESS_TOKENS,
    )

    for t in addr_tokens:
        for x in structure["address_index"].get(t, []):
            candidates.add(x)

    # --------------------------------------------------------
    # RapidFuzz retrieval
    #
    # Only retrieve from the country-specific target list.
    # RapidFuzz runs the actual similarity loop in optimized C.
    # --------------------------------------------------------

    if qname:
        results = process.extract(
            qname,
            structure["name_choices"],
            scorer=fuzz.ratio,
            limit=TOP_NAME,
            score_cutoff=45,
        )

        for _, score, idx in results:
            candidates.add(idx)

    if qaddr:
        results = process.extract(
            qaddr,
            structure["address_choices"],
            scorer=fuzz.ratio,
            limit=TOP_ADDRESS,
            score_cutoff=35,
        )

        for _, score, idx in results:
            candidates.add(idx)

    return candidates


# ============================================================
# FEATURE GENERATION
# ============================================================

def make_features(q, t):
    """
    22 features.

    IMPORTANT:
    The order is intentionally fixed to match the training pipeline.
    """

    qn = q["name_norm"]
    tn = t["name_norm"]

    qa = q["address_norm"]
    ta = t["address_norm"]

    qc = q["country_norm"]
    tc = t["country_norm"]

    qcity = q["city_norm"]
    tcity = t["city_norm"]

    qs = q["state_norm"]
    ts = t["state_norm"]

    qp = q["pincode_norm"]
    tp = t["pincode_norm"]

    return [
        # Name similarities
        safe_ratio(qn, tn),
        safe_partial(qn, tn),
        safe_token_sort(qn, tn),
        safe_token_set(qn, tn),

        # Address similarities
        safe_ratio(qa, ta),
        safe_partial(qa, ta),
        safe_token_sort(qa, ta),
        safe_token_set(qa, ta),

        # Token features
        jaccard(qn, tn),
        jaccard(qa, ta),

        common_token_ratio(qn, tn),
        common_token_ratio(qa, ta),

        # Length
        length_ratio(qn, tn),
        length_ratio(qa, ta),

        # Exact
        float(qn == tn and bool(qn)),
        float(sorted_name(qn) == sorted_name(tn) and bool(qn)),
        float(qa == ta and bool(qa)),

        # Digits
        digit_overlap(qa, ta),

        # Location fields
        float(qc == tc and bool(qc)),
        float(qcity == tcity and bool(qcity)),
        float(qs == ts and bool(qs)),

        # Postal
        float(qp == tp and bool(qp)),
    ]


def score_pairs(model, q, target, candidate_indices):
    rows = []

    indices = list(candidate_indices)

    for idx in indices:
        t = target.loc[idx]

        rows.append(
            make_features(q, t)
        )

    if not rows:
        return []

    X = np.asarray(rows, dtype=np.float32)

    pred = model.predict(
        xgb.DMatrix(X)
    )

    return list(zip(indices, pred.tolist()))


# ============================================================
# MAIN
# ============================================================

def main():

    start_total = time.time()

    log("=" * 70)
    log("FAST FINAL SUBMISSION")
    log("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(
            f"Threshold not found:\n{THRESHOLD_PATH}"
        )

    with open(THRESHOLD_PATH, "r") as f:
        threshold = float(f.read().strip())

    log(f"Threshold: {threshold:.4f}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    log("Loading XGBoost model...")

    model = xgb.Booster()
    model.load_model(MODEL_PATH)

    log(
        f"Model loaded. Features: {model.num_features()}"
    )

    if model.num_features() != 22:
        raise RuntimeError(
            "The saved XGBoost model does not have 22 features. "
            f"Found {model.num_features()}.\n"
            "Do NOT submit with this file until the feature order "
            "from train_ml.py is confirmed."
        )

    # --------------------------------------------------------
    # Load test
    # --------------------------------------------------------

    s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    s2_path = os.path.join(TEST_DIR, "test_source2.tsv")
    s3_path = os.path.join(TEST_DIR, "test_source3.tsv")

    s1_raw = read_tsv(s1_path)
    s2_raw = read_tsv(s2_path)
    s3_raw = read_tsv(s3_path)

    log("")
    log("Preparing data...")

    s1 = prepare_dataframe(s1_raw)
    s2 = prepare_dataframe(s2_raw)
    s3 = prepare_dataframe(s3_raw)

    del s1_raw
    del s2_raw
    del s3_raw

    log(
        f"S1: {len(s1):,}"
    )
    log(
        f"S2: {len(s2):,}"
    )
    log(
        f"S3: {len(s3):,}"
    )

    # --------------------------------------------------------
    # Country intersection
    # --------------------------------------------------------

    common_countries = sorted(
        set(s1["country_norm"])
        & set(s2["country_norm"])
        | set(s1["country_norm"])
        & set(s3["country_norm"])
    )

    log("")
    log(
        f"Common countries: {common_countries}"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    matching_results = {}
    candidate_results = {}

    for country in common_countries:

        log("")
        log("=" * 70)
        log(f"COUNTRY: {country}")
        log("=" * 70)

        q_idx = s1.index[
            s1["country_norm"] == country
        ].to_numpy()

        log(
            f"S1: {len(q_idx):,}"
        )

        # Process both target sources separately.
        for source_name, target in [
            ("S2", s2),
            ("S3", s3),
        ]:

            t_idx = target.index[
                target["country_norm"] == country
            ].to_numpy()

            if len(t_idx) == 0:
                log(
                    f"{source_name}: 0 targets"
                )
                continue

            log("")
            log(
                f"{source_name}: {len(t_idx):,} targets"
            )

            # Local target frame.
            local_target = target.loc[t_idx]

            # Structures use original dataframe indices.
            structures = build_country_structures(
                local_target
            )

            structure = structures[country]

            processed = 0

            for start in range(
                0,
                len(q_idx),
                BATCH_SIZE,
            ):

                batch_idx = q_idx[
                    start:start + BATCH_SIZE
                ]

                for qi in batch_idx:

                    q = s1.loc[qi]

                    candidate_set = retrieve_candidates(
                        q,
                        structure,
                    )

                    # Candidate IDs for this S1.
                    candidate_ids = []

                    for ti in candidate_set:
                        candidate_ids.append(
                            target.loc[ti, "entity_id"]
                        )

                    # Add to global candidate map.
                    qid = q["entity_id"]

                    if qid not in candidate_results:
                        candidate_results[qid] = set()

                    candidate_results[qid].update(
                        candidate_ids
                    )

                    # ------------------------------------------------
                    # Score candidates.
                    #
                    # Keep only the strongest candidates according
                    # to cheap name/address similarity before XGBoost.
                    # ------------------------------------------------

                    ranked = []

                    for ti in candidate_set:

                        t = target.loc[ti]

                        name_score = safe_ratio(
                            q["name_norm"],
                            t["name_norm"],
                        )

                        addr_score = safe_ratio(
                            q["address_norm"],
                            t["address_norm"],
                        )

                        token_name = safe_token_set(
                            q["name_norm"],
                            t["name_norm"],
                        )

                        token_addr = safe_token_set(
                            q["address_norm"],
                            t["address_norm"],
                        )

                        cheap_score = max(
                            name_score,
                            addr_score,
                            token_name,
                            token_addr,
                        )

                        # Exact matches get priority.
                        if (
                            q["name_norm"]
                            and
                            q["name_norm"]
                            == t["name_norm"]
                        ):
                            cheap_score += 1.0

                        if (
                            q["address_norm"]
                            and
                            q["address_norm"]
                            == t["address_norm"]
                        ):
                            cheap_score += 1.0

                        ranked.append(
                            (cheap_score, ti)
                        )

                    ranked.sort(
                        key=lambda x: x[0],
                        reverse=True,
                    )

                    ranked = ranked[
                        :MAX_MODEL_CANDIDATES
                    ]

                    score_indices = [
                        x[1] for x in ranked
                    ]

                    predictions = score_pairs(
                        model,
                        q,
                        target,
                        score_indices,
                    )

                    for ti, probability in predictions:

                        if probability < threshold:
                            continue

                        tid = target.loc[
                            ti,
                            "entity_id",
                        ]

                        if qid not in matching_results:
                            matching_results[qid] = set()

                        matching_results[qid].add(
                            tid
                        )

                processed += len(batch_idx)

                if (
                    processed % 10000 == 0
                    or processed == len(q_idx)
                ):
                    elapsed = (
                        time.time()
                        - start_total
                    )

                    log(
                        f"{source_name}: "
                        f"processed "
                        f"{processed:,}/"
                        f"{len(q_idx):,} "
                        f"({elapsed / 60:.1f} min)"
                    )

    # ========================================================
    # ENSURE EVERY S1 HAS A ROW
    # ========================================================

    for qid in s1["entity_id"]:

        if qid not in matching_results:
            matching_results[qid] = set()

        if qid not in candidate_results:
            candidate_results[qid] = set()

    # ========================================================
    # WRITE MATCHING RESULTS
    # ========================================================

    log("")
    log("=" * 70)
    log("WRITING OUTPUTS")
    log("=" * 70)

    matching_rows = []

    for qid in s1["entity_id"]:

        vals = sorted(
            matching_results[qid]
        )

        matching_rows.append(
            (
                qid,
                ",".join(vals),
            )
        )

    matching_df = pd.DataFrame(
        matching_rows,
        columns=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    matching_df.to_csv(
        MATCHING_OUT,
        sep="\t",
        index=False,
    )

    # ========================================================
    # WRITE CANDIDATE RESULTS
    # ========================================================

    candidate_rows = []

    for qid in s1["entity_id"]:

        vals = sorted(
            candidate_results[qid]
        )

        candidate_rows.append(
            (
                qid,
                ",".join(vals),
            )
        )

    candidate_df = pd.DataFrame(
        candidate_rows,
        columns=[
            "source1_entity_id",
            "candidate_entity_ids",
        ],
    )

    candidate_df.to_csv(
        CANDIDATE_OUT,
        sep="\t",
        index=False,
    )

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    log("")
    log("=" * 70)
    log("VALIDATION")
    log("=" * 70)

    assert len(matching_df) == len(s1)
    assert len(candidate_df) == len(s1)

    assert (
        matching_df["source1_entity_id"]
        .nunique()
        == len(s1)
    )

    assert (
        candidate_df["source1_entity_id"]
        .nunique()
        == len(s1)
    )

    # Every match must be in candidate list.
    candidate_lookup = {}

    for _, row in candidate_df.iterrows():

        vals = (
            set(
                x for x in
                row["candidate_entity_ids"].split(",")
                if x
            )
        )

        candidate_lookup[
            row["source1_entity_id"]
        ] = vals

    violations = 0

    for _, row in matching_df.iterrows():

        qid = row["source1_entity_id"]

        matches = set(
            x for x in
            row["matched_entity_ids"].split(",")
            if x
        )

        candidates = candidate_lookup[qid]

        if not matches.issubset(candidates):
            violations += 1

    log(
        f"S1 rows: {len(s1):,}"
    )

    log(
        f"Matching rows: {len(matching_df):,}"
    )

    log(
        f"Candidate rows: {len(candidate_df):,}"
    )

    log(
        f"Subset violations: {violations}"
    )

    total_matches = sum(
        len(x)
        for x in matching_results.values()
    )

    total_candidates = sum(
        len(x)
        for x in candidate_results.values()
    )

    log(
        f"Total matched pairs: "
        f"{total_matches:,}"
    )

    log(
        f"Total candidate pairs: "
        f"{total_candidates:,}"
    )

    if violations != 0:
        raise RuntimeError(
            "Some matches are not contained in candidate_pairs.tsv"
        )

    # ========================================================
    # FINISHED
    # ========================================================

    elapsed = (
        time.time()
        - start_total
    )

    log("")
    log("=" * 70)
    log("DONE")
    log("=" * 70)

    log(
        f"Runtime: {elapsed / 60:.2f} minutes"
    )

    log("")
    log(
        f"Matching file:"
    )
    log(
        MATCHING_OUT
    )

    log("")
    log(
        f"Candidate file:"
    )
    log(
        CANDIDATE_OUT
    )


if __name__ == "__main__":
    main()
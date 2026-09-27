# ============================================================
# blocker.py
# Business Entity Resolution - Candidate Generation / Blocking
#
# Strategy:
#   1. Country-aware partitioning
#   2. Character TF-IDF name Top-30
#   3. Character TF-IDF address Top-30
#   4. Exact normalized-name block
#   5. Exact sorted-name block
#   6. Exact normalized-address block
#
# Design:
#   - Process one country at a time
#   - Process TF-IDF retrieval in query batches
#   - Write temporary candidate files
#   - Merge one country at a time
#   - Never keep the entire candidate set in RAM
#
# TEST:
#   S1 = first 10,000 rows
#   S2/S3 = first 200,000 rows
#
# FULL:
#   Entire datasets
#
# Outputs:
#   outputs/candidate_pairs_S2_TEST.tsv
#   outputs/candidate_pairs_S3_TEST.tsv
#
#   or FULL:
#   outputs/candidate_pairs_S2_FULL.tsv
#   outputs/candidate_pairs_S3_FULL.tsv
# ============================================================

from pathlib import Path
import os
import re
import sys
import gc
import shutil
import subprocess
import importlib.util

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIG
# ============================================================

RUN_MODE = "FULL"
# Change to:
# RUN_MODE = "FULL"

TEST_S1_ROWS = 10_000
TEST_TARGET_ROWS = 200_000

NAME_TOP_K = 30
ADDRESS_TOP_K = 30

# Number of query rows processed at once.
# Lower = less RAM.
QUERY_BATCH_SIZE = 2_000

# TF-IDF configuration
TFIDF_NGRAM_RANGE = (2, 5)
TFIDF_MIN_DF = 1
TFIDF_SUBLINEAR_TF = True

# sparse_dot_topn threshold
# 0.0 means retrieve the top K even if similarity is very small.
SIMILARITY_THRESHOLD = 0.0

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset" / "train"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TMP_DIR = OUTPUT_DIR / "_blocker_tmp"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

S1_PATH = DATASET_DIR / "train_source1.tsv"
S2_PATH = DATASET_DIR / "train_source2.tsv"
S3_PATH = DATASET_DIR / "train_source3.tsv"
GT_PATH = DATASET_DIR / "train_ground_truth.tsv"


# ============================================================
# OPTIONAL DEPENDENCY CHECK
# ============================================================

def ensure_sparse_dot_topn():
    """
    Make sure sparse_dot_topn is installed.
    """
    if importlib.util.find_spec("sparse_dot_topn") is None:
        print("sparse_dot_topn not found.")
        print("Installing sparse-dot-topn...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "sparse-dot-topn"]
        )

    from sparse_dot_topn import sp_matmul_topn
    return sp_matmul_topn


sp_matmul_topn = ensure_sparse_dot_topn()


# ============================================================
# BASIC TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Unicode-safe normalization for blocking.

    Important:
    - Does NOT transliterate.
    - Keeps non-Latin scripts.
    - Lowercases.
    - Normalizes whitespace.
    - Removes punctuation.
    """
    if pd.isna(value):
        return ""

    value = str(value)

    # Unicode normalization
    value = value.lower()

    # Keep Unicode letters/numbers, replace punctuation with spaces.
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)

    # Collapse whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value


def sorted_normalized_text(value):
    """
    Normalized text with tokens alphabetically sorted.

    Helps with:
        ABC PRIVATE LIMITED
        PRIVATE LIMITED ABC

    but is intentionally supplementary rather than the main blocker.
    """
    value = normalize_text(value)

    if not value:
        return ""

    tokens = value.split()

    return " ".join(sorted(tokens))


def normalize_country(value):
    """
    Normalize country for country-aware partitioning.
    """
    if pd.isna(value):
        return ""

    return normalize_text(value)


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset():
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    print(f"S1: {S1_PATH}")
    print(f"S2: {S2_PATH}")
    print(f"S3: {S3_PATH}")
    print(f"GT: {GT_PATH}")

    s1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s2 = pd.read_csv(
        S2_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s3 = pd.read_csv(
        S3_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    gt = None

    if GT_PATH.exists():
        gt = pd.read_csv(
            GT_PATH,
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )

    print()
    print("Loaded:")
    print("S1:", s1.shape)
    print("S2:", s2.shape)
    print("S3:", s3.shape)

    if gt is not None:
        print("GT:", gt.shape)

    return s1, s2, s3, gt


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

def validate_columns(df, source_name):
    required = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{source_name} is missing columns: {missing}"
        )


# ============================================================
# PREPARE BLOCKING COLUMNS
# ============================================================

def prepare_dataframe(df):
    """
    Adds lightweight blocking columns.
    """
    df = df.copy()

    df["block_name"] = df["business_name"].map(normalize_text)

    df["block_name_sorted"] = (
        df["business_name"].map(sorted_normalized_text)
    )

    df["block_address"] = (
        df["business_address"].map(normalize_text)
    )

    df["block_country"] = (
        df["country"].map(normalize_country)
    )

    return df


# ============================================================
# TEST SAMPLING
# ============================================================

def apply_test_mode(s1, s2, s3):
    if RUN_MODE.upper() != "TEST":
        return s1, s2, s3

    print()
    print("=" * 70)
    print("TEST MODE")
    print("=" * 70)

    s1 = s1.iloc[:TEST_S1_ROWS].copy()
    s2 = s2.iloc[:TEST_TARGET_ROWS].copy()
    s3 = s3.iloc[:TEST_TARGET_ROWS].copy()

    print("S1:", s1.shape)
    print("S2:", s2.shape)
    print("S3:", s3.shape)

    return s1, s2, s3


# ============================================================
# COUNTRY PARTITIONING
# ============================================================

def get_common_countries(query_df, target_df):
    """
    Return countries appearing in both datasets.

    We deliberately do NOT hard-code countries.
    This keeps the blocker open-set for test data.
    """
    q_countries = set(
        query_df["block_country"].dropna().unique()
    )

    t_countries = set(
        target_df["block_country"].dropna().unique()
    )

    common = sorted(q_countries & t_countries)

    return common


# ============================================================
# TF-IDF MATRIX
# ============================================================

def build_tfidf_matrices(query_texts, target_texts):
    """
    Fit character TF-IDF on the combined query + target text
    for the current country/field.

    Returns:
        query_matrix
        target_matrix
        vectorizer
    """

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        dtype=np.float32,
    )

    combined = list(target_texts) + list(query_texts)

    vectorizer.fit(combined)

    target_matrix = vectorizer.transform(target_texts)
    query_matrix = vectorizer.transform(query_texts)

    return query_matrix, target_matrix, vectorizer


# ============================================================
# TOP-K SPARSE RETRIEVAL
# ============================================================

def retrieve_field_to_temp(
    query_df,
    target_df,
    query_global_indices,
    target_global_indices,
    field_name,
    top_k,
    temp_path,
):
    """
    Retrieve top-K target rows for every query row using
    sparse matrix multiplication.

    Writes:
        q_global
        t_global
        similarity

    to a temporary TSV.
    """

    print()
    print(f"TF-IDF {field_name.upper()} retrieval")

    query_texts = (
        query_df[field_name]
        .fillna("")
        .astype(str)
        .tolist()
    )

    target_texts = (
        target_df[field_name]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(
        f"Building TF-IDF matrix for target "
        f"({len(target_texts):,} rows)..."
    )

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        dtype=np.float32,
    )

    # Fit only on target + query strings for this country.
    vectorizer.fit(target_texts + query_texts)

    target_matrix = vectorizer.transform(target_texts)

    print(
        f"TF-IDF {field_name.upper()} target matrix",
        target_matrix.shape,
        "nnz",
        target_matrix.nnz,
    )

    # Header
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write("q_global\tt_global\tsimilarity\n")

    total_queries = len(query_df)

    for start in range(0, total_queries, QUERY_BATCH_SIZE):

        end = min(
            start + QUERY_BATCH_SIZE,
            total_queries,
        )

        batch_texts = query_texts[start:end]

        query_matrix = vectorizer.transform(batch_texts)

        # ----------------------------------------------------
        # Sparse top-K cosine retrieval.
        #
        # TF-IDF vectors are L2 normalized by sklearn, so
        # dot product == cosine similarity.
        # ----------------------------------------------------

        topk_matrix = sp_matmul_topn(
            query_matrix,
            target_matrix.T.tocsr(),
            top_n=top_k,
            threshold=SIMILARITY_THRESHOLD,
            sort=True,
        )

        topk_matrix = topk_matrix.tocsr()

        rows = []

        for local_q in range(topk_matrix.shape[0]):

            row_start = topk_matrix.indptr[local_q]
            row_end = topk_matrix.indptr[local_q + 1]

            target_positions = (
                topk_matrix.indices[row_start:row_end]
            )

            similarities = (
                topk_matrix.data[row_start:row_end]
            )

            if len(target_positions) == 0:
                continue

            q_global = query_global_indices[start + local_q]

            for target_position, similarity in zip(
                target_positions,
                similarities,
            ):
                t_global = target_global_indices[target_position]

                rows.append(
                    (
                        q_global,
                        t_global,
                        float(similarity),
                    )
                )

        if rows:
            with open(
                temp_path,
                "a",
                encoding="utf-8",
            ) as f:

                for q_global, t_global, similarity in rows:
                    f.write(
                        f"{q_global}\t"
                        f"{t_global}\t"
                        f"{similarity:.8f}\n"
                    )

        print(
            f"processed {end:,}/{total_queries:,}"
        )

        del query_matrix
        del topk_matrix
        rows.clear()

        gc.collect()

    # Free memory
    del vectorizer
    del target_matrix
    gc.collect()


# ============================================================
# EXACT BLOCKS
# ============================================================

def build_exact_candidates(
    query_df,
    target_df,
    query_global_indices,
    target_global_indices,
):
    """
    Build supplementary deterministic blocks.

    Blocks:
        exact normalized name
        exact sorted name
        exact normalized address

    Returns:
        list[(q_global, t_global)]
    """

    candidates = []

    # --------------------------------------------------------
    # Name
    # --------------------------------------------------------

    name_map = {}

    for local_idx, value in enumerate(
        target_df["block_name"]
    ):
        if not value:
            continue

        name_map.setdefault(value, []).append(local_idx)

    for q_local, value in enumerate(
        query_df["block_name"]
    ):
        if not value:
            continue

        target_positions = name_map.get(value, [])

        q_global = query_global_indices[q_local]

        for t_local in target_positions:
            t_global = target_global_indices[t_local]

            candidates.append(
                (q_global, t_global)
            )

    # --------------------------------------------------------
    # Sorted name
    # --------------------------------------------------------

    sorted_name_map = {}

    for local_idx, value in enumerate(
        target_df["block_name_sorted"]
    ):
        if not value:
            continue

        sorted_name_map.setdefault(
            value,
            []
        ).append(local_idx)

    for q_local, value in enumerate(
        query_df["block_name_sorted"]
    ):
        if not value:
            continue

        target_positions = sorted_name_map.get(
            value,
            []
        )

        q_global = query_global_indices[q_local]

        for t_local in target_positions:
            t_global = target_global_indices[t_local]

            candidates.append(
                (q_global, t_global)
            )

    # --------------------------------------------------------
    # Address
    # --------------------------------------------------------

    address_map = {}

    for local_idx, value in enumerate(
        target_df["block_address"]
    ):
        if not value:
            continue

        address_map.setdefault(
            value,
            []
        ).append(local_idx)

    for q_local, value in enumerate(
        query_df["block_address"]
    ):
        if not value:
            continue

        target_positions = address_map.get(
            value,
            []
        )

        q_global = query_global_indices[q_local]

        for t_local in target_positions:
            t_global = target_global_indices[t_local]

            candidates.append(
                (q_global, t_global)
            )

    return candidates


# ============================================================
# MERGE COUNTRY CANDIDATES
# ============================================================

def merge_country_files(
    query_df,
    target_df,
    query_global_indices,
    target_global_indices,
    name_temp_path,
    address_temp_path,
    output_path,
):
    """
    Merge:

        name TF-IDF candidates
        address TF-IDF candidates
        exact name candidates
        exact sorted-name candidates
        exact address candidates

    into the final candidate output.

    IMPORTANT:
    The previous version had a .loc bug here.

    q_global / t_global are GLOBAL INTEGER POSITIONS.

    Therefore we explicitly map:
        global position -> entity_id

    rather than using .loc with those integers.
    """

    print("merging country candidates...")

    # --------------------------------------------------------
    # Build global-position -> entity_id maps.
    #
    # DO NOT use:
    #     query_df.loc[q_global]
    #
    # because q_global is a positional index, not necessarily
    # a pandas index label.
    # --------------------------------------------------------

    query_id_map = {
        int(global_idx): str(
            query_df.iloc[local_idx]["entity_id"]
        )
        for local_idx, global_idx
        in enumerate(query_global_indices)
    }

    target_id_map = {
        int(global_idx): str(
            target_df.iloc[local_idx]["entity_id"]
        )
        for local_idx, global_idx
        in enumerate(target_global_indices)
    }

    # --------------------------------------------------------
    # Candidate pairs as integer positions.
    #
    # We use a Python set only for THIS COUNTRY.
    # This keeps memory bounded by country size.
    # --------------------------------------------------------

    country_pairs = set()

    # --------------------------------------------------------
    # Read NAME TF-IDF candidates
    # --------------------------------------------------------

    if Path(name_temp_path).exists():

        name_df = pd.read_csv(
            name_temp_path,
            sep="\t",
            dtype={
                "q_global": "int64",
                "t_global": "int64",
                "similarity": "float32",
            },
        )

        for q, t in zip(
            name_df["q_global"].to_numpy(),
            name_df["t_global"].to_numpy(),
        ):
            country_pairs.add(
                (int(q), int(t))
            )

        del name_df
        gc.collect()

    # --------------------------------------------------------
    # Read ADDRESS TF-IDF candidates
    # --------------------------------------------------------

    if Path(address_temp_path).exists():

        address_df = pd.read_csv(
            address_temp_path,
            sep="\t",
            dtype={
                "q_global": "int64",
                "t_global": "int64",
                "similarity": "float32",
            },
        )

        for q, t in zip(
            address_df["q_global"].to_numpy(),
            address_df["t_global"].to_numpy(),
        ):
            country_pairs.add(
                (int(q), int(t))
            )

        del address_df
        gc.collect()

    # --------------------------------------------------------
    # Exact deterministic blocks
    # --------------------------------------------------------

    exact_candidates = build_exact_candidates(
        query_df=query_df,
        target_df=target_df,
        query_global_indices=query_global_indices,
        target_global_indices=target_global_indices,
    )

    country_pairs.update(exact_candidates)

    del exact_candidates
    gc.collect()

    print(
        f"country unique candidate pairs: "
        f"{len(country_pairs):,}"
    )

    # --------------------------------------------------------
    # Convert global integer positions -> entity IDs
    # --------------------------------------------------------

    output_rows = []

    missing_query_ids = 0
    missing_target_ids = 0

    for q_global, t_global in country_pairs:

        q_id = query_id_map.get(q_global)
        t_id = target_id_map.get(t_global)

        if q_id is None:
            missing_query_ids += 1
            continue

        if t_id is None:
            missing_target_ids += 1
            continue

        output_rows.append(
            (
                q_id,
                t_id,
            )
        )

    if missing_query_ids or missing_target_ids:
        raise RuntimeError(
            "Global-index mapping failure: "
            f"missing_query_ids={missing_query_ids}, "
            f"missing_target_ids={missing_target_ids}"
        )

    # --------------------------------------------------------
    # Write country candidates
    # --------------------------------------------------------

    if output_rows:

        with open(
            output_path,
            "a",
            encoding="utf-8",
        ) as f:

            for q_id, t_id in output_rows:

                f.write(
                    f"{q_id}\t{t_id}\n"
                )

    print(
        f"wrote {len(output_rows):,} "
        f"candidate pairs"
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del country_pairs
    del output_rows
    del query_id_map
    del target_id_map

    gc.collect()


# ============================================================
# BUILD CANDIDATES FOR ONE TARGET SOURCE
# ============================================================

def build_candidates(
    query_df,
    target_df,
    target_name,
    output_path,
):
    """
    Complete streaming candidate generation for S1 -> S2
    or S1 -> S3.
    """

    print()
    print("=" * 70)
    print(f"BUILDING CANDIDATES: {target_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Remove previous output
    # --------------------------------------------------------

    if output_path.exists():

        print(
            "Removing previous output:"
        )
        print(output_path)

        output_path.unlink()

    # Header
    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "source1_entity_id\t"
            "matched_entity_id\n"
        )

    # --------------------------------------------------------
    # Global integer positions
    # --------------------------------------------------------

    query_global_all = np.arange(
        len(query_df),
        dtype=np.int64,
    )

    target_global_all = np.arange(
        len(target_df),
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Common countries
    # --------------------------------------------------------

    countries = get_common_countries(
        query_df,
        target_df,
    )

    print(
        f"Common countries: {len(countries)}"
    )

    print(
        "Countries:",
        countries,
    )

    # --------------------------------------------------------
    # Process country-by-country
    # --------------------------------------------------------

    for country_number, country in enumerate(
        countries,
        start=1,
    ):

        print()
        print("-" * 70)
        print(
            f"COUNTRY {country_number}/{len(countries)}: "
            f"'{country}'"
        )
        print("-" * 70)

        # ----------------------------------------------------
        # Get global positions for this country.
        #
        # np.flatnonzero is positional and therefore gives us
        # stable global integer positions.
        # ----------------------------------------------------

        query_country_mask = (
            query_df["block_country"].to_numpy()
            == country
        )

        target_country_mask = (
            target_df["block_country"].to_numpy()
            == country
        )

        query_global_indices = np.flatnonzero(
            query_country_mask
        ).astype(np.int64)

        target_global_indices = np.flatnonzero(
            target_country_mask
        ).astype(np.int64)

        print(
            f"S1 rows: {len(query_global_indices):,}"
        )

        print(
            f"{target_name} rows: "
            f"{len(target_global_indices):,}"
        )

        if (
            len(query_global_indices) == 0
            or len(target_global_indices) == 0
        ):
            print(
                "Skipping empty country."
            )
            continue

        # ----------------------------------------------------
        # Extract country dataframes.
        #
        # Reset index so .iloc/local positions are clean.
        # ----------------------------------------------------

        query_country = (
            query_df
            .iloc[query_global_indices]
            .reset_index(drop=True)
            .copy()
        )

        target_country = (
            target_df
            .iloc[target_global_indices]
            .reset_index(drop=True)
            .copy()
        )

        # ----------------------------------------------------
        # Temporary files
        # ----------------------------------------------------

        country_safe = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            str(country),
        )

        target_short = target_name.lower()

        name_temp_path = (
            TMP_DIR
            / f"{target_short}_{country_safe}_name.tsv"
        )

        address_temp_path = (
            TMP_DIR
            / f"{target_short}_{country_safe}_address.tsv"
        )

        # Remove stale temp files.
        for temp_path in [
            name_temp_path,
            address_temp_path,
        ]:
            if temp_path.exists():
                temp_path.unlink()

        # ----------------------------------------------------
        # NAME TF-IDF Top-K
        # ----------------------------------------------------

        retrieve_field_to_temp(
            query_df=query_country,
            target_df=target_country,
            query_global_indices=query_global_indices,
            target_global_indices=target_global_indices,
            field_name="block_name",
            top_k=NAME_TOP_K,
            temp_path=name_temp_path,
        )

        # ----------------------------------------------------
        # ADDRESS TF-IDF Top-K
        # ----------------------------------------------------

        retrieve_field_to_temp(
            query_df=query_country,
            target_df=target_country,
            query_global_indices=query_global_indices,
            target_global_indices=target_global_indices,
            field_name="block_address",
            top_k=ADDRESS_TOP_K,
            temp_path=address_temp_path,
        )

        # ----------------------------------------------------
        # Merge everything for this country
        # ----------------------------------------------------

        merge_country_files(
            query_df=query_country,
            target_df=target_country,
            query_global_indices=query_global_indices,
            target_global_indices=target_global_indices,
            name_temp_path=name_temp_path,
            address_temp_path=address_temp_path,
            output_path=output_path,
        )

        # ----------------------------------------------------
        # Remove temporary files
        # ----------------------------------------------------

        for temp_path in [
            name_temp_path,
            address_temp_path,
        ]:

            if temp_path.exists():
                temp_path.unlink()

        # ----------------------------------------------------
        # Cleanup country objects
        # ----------------------------------------------------

        del query_country
        del target_country
        del query_global_indices
        del target_global_indices

        gc.collect()

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    del query_global_all
    del target_global_all

    gc.collect()

    print()
    print(
        f"Finished {target_name}:"
    )
    print(output_path)


# ============================================================
# OUTPUT STATISTICS
# ============================================================

def count_output_pairs(path):
    """
    Count unique candidate pairs in final output.

    Output has:
        source1_entity_id
        matched_entity_id
    """

    if not path.exists():
        return 0

    total = 0

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        # Skip header
        next(f, None)

        for _ in f:
            total += 1

    return total


# ============================================================
# OPTIONAL TEST RECALL
# ============================================================

def build_gt_pairs(gt):
    """
    Expand ground truth:

        source1_entity_id
        matched_entity_ids

    into a set of:
        (source1_entity_id, matched_entity_id)
    """

    if gt is None:
        return None

    if (
        "source1_entity_id" not in gt.columns
        or "matched_entity_ids" not in gt.columns
    ):
        print(
            "GT columns not found; "
            "skipping recall evaluation."
        )
        return None

    pairs = set()

    for row in gt.itertuples(index=False):

        source1_id = str(
            getattr(
                row,
                "source1_entity_id",
            )
        )

        matched_value = getattr(
            row,
            "matched_entity_ids",
        )

        if (
            matched_value is None
            or str(matched_value).strip() == ""
        ):
            continue

        for target_id in str(
            matched_value
        ).split(","):

            target_id = target_id.strip()

            if target_id:
                pairs.add(
                    (
                        source1_id,
                        target_id,
                    )
                )

    return pairs


def evaluate_output_recall(
    output_path,
    gt_pairs,
):
    """
    Evaluate candidate recall against GT.

    Only useful when GT entity IDs are from the same
    benchmark subset.

    In TEST mode, because S1/S2/S3 are truncated,
    this measures recall of GT pairs whose entities are
    present in the TEST subset.
    """

    if gt_pairs is None:
        return

    print()
    print("=" * 70)
    print(
        f"EVALUATING {output_path.name}"
    )
    print("=" * 70)

    if not output_path.exists():
        print("Output does not exist.")
        return

    candidate_pairs = set()

    with open(
        output_path,
        "r",
        encoding="utf-8",
    ) as f:

        next(f, None)

        for line in f:

            line = line.rstrip("\n")

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 2:
                continue

            candidate_pairs.add(
                (
                    parts[0],
                    parts[1],
                )
            )

    # --------------------------------------------------------
    # Only GT pairs whose IDs occur in candidate universe
    # --------------------------------------------------------

    candidate_s1 = {
        q
        for q, _ in candidate_pairs
    }

    candidate_target = {
        t
        for _, t in candidate_pairs
    }

    eligible_gt = {
        pair
        for pair in gt_pairs
        if (
            pair[0] in candidate_s1
            and pair[1] in candidate_target
        )
    }

    retrieved = (
        eligible_gt
        & candidate_pairs
    )

    missed = (
        eligible_gt
        - candidate_pairs
    )

    if len(eligible_gt) == 0:

        print(
            "No eligible GT pairs for this output."
        )
        return

    recall = (
        len(retrieved)
        / len(eligible_gt)
        * 100.0
    )

    print(
        f"Candidate pairs: "
        f"{len(candidate_pairs):,}"
    )

    print(
        f"Eligible GT pairs: "
        f"{len(eligible_gt):,}"
    )

    print(
        f"Retrieved GT pairs: "
        f"{len(retrieved):,}"
    )

    print(
        f"Missed GT pairs: "
        f"{len(missed):,}"
    )

    print(
        f"Recall: {recall:.4f}%"
    )

    if missed:
        print()
        print("First 20 missed pairs:")

        for pair in list(missed)[:20]:
            print(pair)

    del candidate_pairs
    del eligible_gt
    del retrieved
    del missed

    gc.collect()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION BLOCKER")
    print("=" * 70)

    print(
        f"RUN_MODE = {RUN_MODE}"
    )

    print(
        f"NAME_TOP_K = {NAME_TOP_K}"
    )

    print(
        f"ADDRESS_TOP_K = {ADDRESS_TOP_K}"
    )

    print(
        f"QUERY_BATCH_SIZE = {QUERY_BATCH_SIZE}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    s1, s2, s3, gt = load_dataset()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_columns(s1, "S1")
    validate_columns(s2, "S2")
    validate_columns(s3, "S3")

    # --------------------------------------------------------
    # Test subset
    # --------------------------------------------------------

    s1, s2, s3 = apply_test_mode(
        s1,
        s2,
        s3,
    )

    # --------------------------------------------------------
    # Prepare blocking fields
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print("PREPARING BLOCKING FIELDS")
    print(
        "=" * 70
    )

    s1 = prepare_dataframe(s1)
    s2 = prepare_dataframe(s2)
    s3 = prepare_dataframe(s3)

    print("Prepared S1:", s1.shape)
    print("Prepared S2:", s2.shape)
    print("Prepared S3:", s3.shape)

    # --------------------------------------------------------
    # Output names
    # --------------------------------------------------------

    mode_suffix = RUN_MODE.upper()

    s2_output = (
        OUTPUT_DIR
        / f"candidate_pairs_S2_{mode_suffix}.tsv"
    )

    s3_output = (
        OUTPUT_DIR
        / f"candidate_pairs_S3_{mode_suffix}.tsv"
    )

    # --------------------------------------------------------
    # S1 -> S2
    # --------------------------------------------------------

    build_candidates(
        query_df=s1,
        target_df=s2,
        target_name="S2",
        output_path=s2_output,
    )

    # --------------------------------------------------------
    # S1 -> S3
    # --------------------------------------------------------

    build_candidates(
        query_df=s1,
        target_df=s3,
        target_name="S3",
        output_path=s3_output,
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL OUTPUT")
    print("=" * 70)

    s2_count = count_output_pairs(
        s2_output
    )

    s3_count = count_output_pairs(
        s3_output
    )

    print(
        f"S2 candidate pairs: "
        f"{s2_count:,}"
    )

    print(
        f"S3 candidate pairs: "
        f"{s3_count:,}"
    )

    print(
        f"Total candidate pairs: "
        f"{s2_count + s3_count:,}"
    )

    print()
    print("S2 output:")
    print(s2_output)

    print()
    print("S3 output:")
    print(s3_output)

    # --------------------------------------------------------
    # TEST recall
    #
    # Important:
    # GT is full training GT.
    # We only evaluate GT pairs whose IDs are represented
    # in the TEST candidate universe.
    # --------------------------------------------------------

    if RUN_MODE.upper() == "TEST" and gt is not None:

        print()
        print(
            "=" * 70
        )
        print("BUILDING GROUND-TRUTH PAIRS")
        print(
            "=" * 70
        )

        gt_pairs = build_gt_pairs(gt)

        if gt_pairs is not None:

            print(
                f"Unique GT pairs: "
                f"{len(gt_pairs):,}"
            )

            evaluate_output_recall(
                s2_output,
                gt_pairs,
            )

            evaluate_output_recall(
                s3_output,
                gt_pairs,
            )

            del gt_pairs

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    del s1
    del s2
    del s3
    del gt

    gc.collect()

    print()
    print("=" * 70)
    print("BLOCKING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
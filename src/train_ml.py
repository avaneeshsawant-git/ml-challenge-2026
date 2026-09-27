# ============================================================
# train_ml.py
#
# FAST ML MATCHER TRAINING
#
# Uses the SUCCESSFUL TEST blocker candidates:
#
#   candidate_pairs_S2_TEST.tsv
#   candidate_pairs_S3_TEST.tsv
#
# Training data:
#   train_source1.tsv
#   train_source2.tsv
#   train_source3.tsv
#   train_ground_truth.tsv
#
# Model:
#   XGBoost binary classifier
#
# Validation:
#   80/20 split by S1 entity
#
# Metric:
#   Macro F0.5
#
# IMPORTANT:
#   The model is saved immediately after training,
#   BEFORE threshold search.
#
# ============================================================

from pathlib import Path
import sys
import subprocess
import importlib.util
import gc
import re
import time

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset" / "train"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "model"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

S1_PATH = DATASET_DIR / "train_source1.tsv"
S2_PATH = DATASET_DIR / "train_source2.tsv"
S3_PATH = DATASET_DIR / "train_source3.tsv"
GT_PATH = DATASET_DIR / "train_ground_truth.tsv"

S2_CANDIDATES = (
    OUTPUT_DIR
    / "candidate_pairs_S2_TEST.tsv"
)

S3_CANDIDATES = (
    OUTPUT_DIR
    / "candidate_pairs_S3_TEST.tsv"
)

# ------------------------------------------------------------
# TEST blocker universe
#
# These correspond to the blocker that achieved:
# S2 ~99.70% recall
# S3 ~99.70% recall
# ------------------------------------------------------------

S1_LIMIT = 10_000
TARGET_LIMIT = 200_000

# ------------------------------------------------------------
# Randomness
# ------------------------------------------------------------

RANDOM_SEED = 42

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

VALIDATION_FRACTION = 0.20

# ------------------------------------------------------------
# Negative sampling
# ------------------------------------------------------------

MAX_HARD_NEGATIVES = 75_000
MAX_RANDOM_NEGATIVES = 75_000

# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

N_ESTIMATORS = 500
MAX_DEPTH = 7
LEARNING_RATE = 0.05
SUBSAMPLE = 0.85
COLSAMPLE = 0.85

# ------------------------------------------------------------
# Feature processing
# ------------------------------------------------------------

FEATURE_BATCH_SIZE = 100_000


# ============================================================
# TIMER
# ============================================================

def elapsed_minutes(start_time):
    return (
        time.time() - start_time
    ) / 60.0


# ============================================================
# DEPENDENCIES
# ============================================================

def ensure_package(
    package_name,
    import_name=None,
):

    if import_name is None:
        import_name = package_name

    if importlib.util.find_spec(
        import_name
    ) is None:

        print(
            f"Installing {package_name}..."
        )

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                package_name,
            ]
        )


ensure_package(
    "rapidfuzz",
    "rapidfuzz",
)

ensure_package(
    "xgboost",
    "xgboost",
)

from rapidfuzz import fuzz
from xgboost import XGBClassifier


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def norm_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower()

    value = re.sub(
        r"[^\w]+",
        " ",
        value,
        flags=re.UNICODE,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


def sorted_tokens(value):

    if not value:
        return ""

    return " ".join(
        sorted(
            value.split()
        )
    )


def digit_set(value):

    return set(
        re.findall(
            r"\d+",
            value,
        )
    )


def token_set(value):

    return set(
        value.split()
    )


def safe_ratio(a, b):

    if not a or not b:
        return 0.0

    return (
        min(
            len(a),
            len(b),
        )
        /
        max(
            len(a),
            len(b),
        )
    )


def jaccard_tokens(a, b):

    if not a or not b:
        return 0.0

    sa = token_set(a)
    sb = token_set(b)

    if not sa and not sb:
        return 1.0

    if not sa or not sb:
        return 0.0

    union = sa | sb

    if not union:
        return 0.0

    return (
        len(sa & sb)
        /
        len(union)
    )


def digit_overlap(a, b):

    da = digit_set(a)
    db = digit_set(b)

    if not da and not db:
        return 1.0

    if not da or not db:
        return 0.0

    return (
        len(da & db)
        /
        len(da | db)
    )


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_data():

    print()
    print("=" * 70)
    print("LOADING TRAINING DATA")
    print("=" * 70)

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

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        "S1:",
        s1.shape,
    )

    print(
        "S2:",
        s2.shape,
    )

    print(
        "S3:",
        s3.shape,
    )

    print(
        "GT:",
        gt.shape,
    )

    return (
        s1,
        s2,
        s3,
        gt,
    )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare(df):

    df = df.copy()

    df["name_norm"] = (
        df[
            "business_name"
        ]
        .map(norm_text)
    )

    df["address_norm"] = (
        df[
            "business_address"
        ]
        .map(norm_text)
    )

    df["name_sorted"] = (
        df[
            "name_norm"
        ]
        .map(sorted_tokens)
    )

    df["country_norm"] = (
        df[
            "country"
        ]
        .map(norm_text)
    )

    return df


# ============================================================
# LOAD BLOCKER CANDIDATES
# ============================================================

def load_candidates():

    print()
    print("=" * 70)
    print("LOADING BLOCKER CANDIDATES")
    print("=" * 70)

    if not S2_CANDIDATES.exists():
        raise FileNotFoundError(
            f"Missing:\n{S2_CANDIDATES}"
        )

    if not S3_CANDIDATES.exists():
        raise FileNotFoundError(
            f"Missing:\n{S3_CANDIDATES}"
        )

    s2 = pd.read_csv(
        S2_CANDIDATES,
        sep="\t",
        dtype=str,
    )

    s3 = pd.read_csv(
        S3_CANDIDATES,
        sep="\t",
        dtype=str,
    )

    s2["source"] = "S2"
    s3["source"] = "S3"

    s2.columns = [
        "source1_entity_id",
        "matched_entity_id",
        "source",
    ]

    s3.columns = [
        "source1_entity_id",
        "matched_entity_id",
        "source",
    ]

    print(
        "S2 candidate rows:",
        f"{len(s2):,}",
    )

    print(
        "S3 candidate rows:",
        f"{len(s3):,}",
    )

    candidates = pd.concat(
        [
            s2,
            s3,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Keep source in duplicate identity.
    #
    # Same textual target ID cannot normally occur across
    # sources, but retaining source is safer.
    # --------------------------------------------------------

    candidates = candidates.drop_duplicates(
        subset=[
            "source1_entity_id",
            "matched_entity_id",
            "source",
        ]
    ).reset_index(
        drop=True
    )

    print(
        "Combined candidates:",
        f"{len(candidates):,}",
    )

    return candidates


# ============================================================
# GROUND TRUTH
# ============================================================

def build_gt_set(gt):

    print()
    print(
        "=" * 70
    )
    print(
        "BUILDING GROUND TRUTH"
    )
    print(
        "=" * 70
    )

    gt_pairs = set()

    for row in gt.itertuples(
        index=False
    ):

        s1_id = str(
            row.source1_entity_id
        )

        matched_value = str(
            row.matched_entity_ids
        ).strip()

        if not matched_value:
            continue

        for target_id in (
            matched_value.split(",")
        ):

            target_id = (
                target_id.strip()
            )

            if target_id:

                gt_pairs.add(
                    (
                        s1_id,
                        target_id,
                    )
                )

    print(
        "Unique GT pairs:",
        f"{len(gt_pairs):,}",
    )

    return gt_pairs


# ============================================================
# LABEL CANDIDATES
# ============================================================

def add_labels(
    candidates,
    gt_pairs,
):

    print()
    print(
        "=" * 70
    )
    print(
        "LABELING CANDIDATES"
    )
    print(
        "=" * 70
    )

    labels = np.fromiter(
        (
            1
            if (
                row.source1_entity_id,
                row.matched_entity_id,
            )
            in gt_pairs
            else 0
            for row
            in candidates.itertuples(
                index=False
            )
        ),
        dtype=np.int8,
        count=len(candidates),
    )

    candidates["label"] = labels

    positives = int(
        labels.sum()
    )

    negatives = (
        len(labels)
        - positives
    )

    print(
        "Positive candidates:",
        f"{positives:,}",
    )

    print(
        "Negative candidates:",
        f"{negatives:,}",
    )

    return candidates


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def calculate_features(
    candidates,
    s1,
    s2,
    s3,
):

    print()
    print("=" * 70)
    print("BUILDING PAIRWISE FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # Lookup dictionaries
    # --------------------------------------------------------

    s1_map = (
        s1.set_index(
            "entity_id"
        )
        .to_dict(
            "index"
        )
    )

    s2_map = (
        s2.set_index(
            "entity_id"
        )
        .to_dict(
            "index"
        )
    )

    s3_map = (
        s3.set_index(
            "entity_id"
        )
        .to_dict(
            "index"
        )
    )

    feature_names = [
        "name_ratio",
        "name_token_sort",
        "name_token_set",
        "name_partial",
        "name_jaccard",
        "name_length_ratio",
        "name_exact",
        "name_sorted_exact",

        "address_ratio",
        "address_token_sort",
        "address_token_set",
        "address_partial",
        "address_jaccard",
        "address_length_ratio",
        "address_exact",
        "address_digit_overlap",

        "country_match",

        "max_name_address",
        "avg_name_address",
        "weighted_similarity",

        "name_length",
        "address_length",
    ]

    n = len(candidates)

    feature_blocks = []

    start_time = time.time()

    for start in range(
        0,
        n,
        FEATURE_BATCH_SIZE,
    ):

        end = min(
            start
            + FEATURE_BATCH_SIZE,
            n,
        )

        batch = candidates.iloc[
            start:end
        ]

        rows = []

        for row in batch.itertuples(
            index=False
        ):

            q = s1_map.get(
                row.source1_entity_id
            )

            if row.source == "S2":

                t = s2_map.get(
                    row.matched_entity_id
                )

            else:

                t = s3_map.get(
                    row.matched_entity_id
                )

            # ------------------------------------------------
            # Safety fallback
            # ------------------------------------------------

            if q is None or t is None:

                rows.append(
                    [0.0]
                    * len(feature_names)
                )

                continue

            # ------------------------------------------------
            # Values
            # ------------------------------------------------

            q_name = q[
                "name_norm"
            ]

            t_name = t[
                "name_norm"
            ]

            q_addr = q[
                "address_norm"
            ]

            t_addr = t[
                "address_norm"
            ]

            q_sorted = q[
                "name_sorted"
            ]

            t_sorted = t[
                "name_sorted"
            ]

            q_country = q[
                "country_norm"
            ]

            t_country = t[
                "country_norm"
            ]

            # ------------------------------------------------
            # NAME
            # ------------------------------------------------

            name_ratio = (
                fuzz.ratio(
                    q_name,
                    t_name,
                )
                / 100.0
            )

            name_token_sort = (
                fuzz.token_sort_ratio(
                    q_name,
                    t_name,
                )
                / 100.0
            )

            name_token_set = (
                fuzz.token_set_ratio(
                    q_name,
                    t_name,
                )
                / 100.0
            )

            name_partial = (
                fuzz.partial_ratio(
                    q_name,
                    t_name,
                )
                / 100.0
            )

            name_jaccard = (
                jaccard_tokens(
                    q_name,
                    t_name,
                )
            )

            name_length_ratio = (
                safe_ratio(
                    q_name,
                    t_name,
                )
            )

            name_exact = float(
                bool(q_name)
                and q_name == t_name
            )

            name_sorted_exact = float(
                bool(q_sorted)
                and q_sorted == t_sorted
            )

            # ------------------------------------------------
            # ADDRESS
            # ------------------------------------------------

            address_ratio = (
                fuzz.ratio(
                    q_addr,
                    t_addr,
                )
                / 100.0
            )

            address_token_sort = (
                fuzz.token_sort_ratio(
                    q_addr,
                    t_addr,
                )
                / 100.0
            )

            address_token_set = (
                fuzz.token_set_ratio(
                    q_addr,
                    t_addr,
                )
                / 100.0
            )

            address_partial = (
                fuzz.partial_ratio(
                    q_addr,
                    t_addr,
                )
                / 100.0
            )

            address_jaccard = (
                jaccard_tokens(
                    q_addr,
                    t_addr,
                )
            )

            address_length_ratio = (
                safe_ratio(
                    q_addr,
                    t_addr,
                )
            )

            address_exact = float(
                bool(q_addr)
                and q_addr == t_addr
            )

            address_digit_overlap = (
                digit_overlap(
                    q_addr,
                    t_addr,
                )
            )

            # ------------------------------------------------
            # COMBINED
            # ------------------------------------------------

            country_match = float(
                bool(q_country)
                and q_country == t_country
            )

            max_name_address = max(
                name_ratio,
                address_ratio,
            )

            avg_name_address = (
                name_ratio
                + address_ratio
            ) / 2.0

            weighted_similarity = (
                0.60 * name_ratio
                + 0.40 * address_ratio
            )

            rows.append(
                [
                    name_ratio,
                    name_token_sort,
                    name_token_set,
                    name_partial,
                    name_jaccard,
                    name_length_ratio,
                    name_exact,
                    name_sorted_exact,

                    address_ratio,
                    address_token_sort,
                    address_token_set,
                    address_partial,
                    address_jaccard,
                    address_length_ratio,
                    address_exact,
                    address_digit_overlap,

                    country_match,

                    max_name_address,
                    avg_name_address,
                    weighted_similarity,

                    float(
                        len(q_name)
                    ),

                    float(
                        len(q_addr)
                    ),
                ]
            )

        feature_blocks.append(
            np.asarray(
                rows,
                dtype=np.float32,
            )
        )

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"processed "
            f"{end:,}/{n:,} "
            f"("
            f"{100.0 * end / n:.1f}%"
            f") "
            f"elapsed="
            f"{elapsed / 60:.1f} min"
        )

        del rows
        del batch

        gc.collect()

    X = np.vstack(
        feature_blocks
    ).astype(
        np.float32,
        copy=False,
    )

    del feature_blocks

    gc.collect()

    print()
    print(
        "Feature matrix:",
        X.shape,
    )

    print(
        "Feature memory:",
        f"{X.nbytes / 1024 / 1024:.1f} MB",
    )

    return (
        X,
        feature_names,
    )


# ============================================================
# ENTITY-LEVEL TRAIN / VALIDATION SPLIT
# ============================================================

def make_entity_split(
    candidates,
):

    print()
    print("=" * 70)
    print("ENTITY-LEVEL TRAIN / VALIDATION SPLIT")
    print("=" * 70)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    entities = (
        candidates[
            "source1_entity_id"
        ]
        .unique()
    )

    rng.shuffle(
        entities
    )

    split_index = int(
        len(entities)
        * (
            1.0
            - VALIDATION_FRACTION
        )
    )

    train_entities = set(
        entities[
            :split_index
        ]
    )

    validation_entities = set(
        entities[
            split_index:
        ]
    )

    train_mask = (
        candidates[
            "source1_entity_id"
        ]
        .isin(
            train_entities
        )
        .to_numpy()
    )

    validation_mask = (
        candidates[
            "source1_entity_id"
        ]
        .isin(
            validation_entities
        )
        .to_numpy()
    )

    print(
        "Train S1:",
        f"{len(train_entities):,}",
    )

    print(
        "Validation S1:",
        f"{len(validation_entities):,}",
    )

    print(
        "Train candidate rows:",
        f"{int(train_mask.sum()):,}",
    )

    print(
        "Validation candidate rows:",
        f"{int(validation_mask.sum()):,}",
    )

    return (
        train_mask,
        validation_mask,
    )


# ============================================================
# TRAINING SAMPLE
# ============================================================

def build_training_subset(
    candidates,
    X,
    train_mask,
):

    print()
    print("=" * 70)
    print("BUILDING TRAINING SAMPLE")
    print("=" * 70)

    train_indices = np.flatnonzero(
        train_mask
    )

    train_labels = (
        candidates.iloc[
            train_indices
        ]["label"]
        .to_numpy()
    )

    positive_indices = (
        train_indices[
            train_labels == 1
        ]
    )

    negative_indices = (
        train_indices[
            train_labels == 0
        ]
    )

    print(
        "Training positives:",
        f"{len(positive_indices):,}",
    )

    print(
        "Training negatives available:",
        f"{len(negative_indices):,}",
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Hard negatives
    #
    # Strong lexical similarity but NOT ground truth.
    # --------------------------------------------------------

    negative_score = (
        0.55
        * X[
            negative_indices,
            0,
        ]
        +
        0.45
        * X[
            negative_indices,
            8,
        ]
    )

    hard_order = np.argsort(
        negative_score
    )[::-1]

    hard_count = min(
        MAX_HARD_NEGATIVES,
        len(hard_order),
    )

    hard_negative_indices = (
        negative_indices[
            hard_order[
                :hard_count
            ]
        ]
    )

    # --------------------------------------------------------
    # Random negatives
    # --------------------------------------------------------

    remaining = np.setdiff1d(
        negative_indices,
        hard_negative_indices,
        assume_unique=False,
    )

    random_count = min(
        MAX_RANDOM_NEGATIVES,
        len(remaining),
    )

    if random_count > 0:

        random_negative_indices = (
            rng.choice(
                remaining,
                size=random_count,
                replace=False,
            )
        )

    else:

        random_negative_indices = (
            np.array(
                [],
                dtype=np.int64,
            )
        )

    selected_indices = (
        np.concatenate(
            [
                positive_indices,
                hard_negative_indices,
                random_negative_indices,
            ]
        )
    )

    rng.shuffle(
        selected_indices
    )

    print(
        "Selected positives:",
        f"{len(positive_indices):,}",
    )

    print(
        "Selected hard negatives:",
        f"{len(hard_negative_indices):,}",
    )

    print(
        "Selected random negatives:",
        f"{len(random_negative_indices):,}",
    )

    print(
        "Total training rows:",
        f"{len(selected_indices):,}",
    )

    return selected_indices


# ============================================================
# FAST VALIDATION THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    model,
    X_val,
    val_candidates,
    gt_pairs,
):

    print()
    print("=" * 70)
    print("VALIDATION THRESHOLD SEARCH")
    print("=" * 70)

    start_time = time.time()

    # --------------------------------------------------------
    # Predict ONCE
    # --------------------------------------------------------

    print(
        "Generating validation probabilities..."
    )

    probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    print(
        f"Probability prediction done "
        f"in {elapsed_minutes(start_time):.2f} min"
    )

    # --------------------------------------------------------
    # Validation entities
    # --------------------------------------------------------

    validation_entities = (
        val_candidates[
            "source1_entity_id"
        ]
        .unique()
    )

    entity_to_index = {
        entity_id: idx
        for idx, entity_id
        in enumerate(
            validation_entities
        )
    }

    n_entities = len(
        validation_entities
    )

    # --------------------------------------------------------
    # Build GT lookup ONLY ONCE
    # --------------------------------------------------------

    print(
        "Building validation GT lookup..."
    )

    gt_lookup = set()

    true_counts = np.zeros(
        n_entities,
        dtype=np.int32,
    )

    for q_id, t_id in gt_pairs:

        idx = entity_to_index.get(
            q_id
        )

        if idx is None:
            continue

        gt_lookup.add(
            (
                q_id,
                t_id,
            )
        )

        true_counts[idx] += 1

    print(
        "Validation GT pairs:",
        f"{len(gt_lookup):,}",
    )

    # --------------------------------------------------------
    # Candidate arrays
    # --------------------------------------------------------

    q_ids = (
        val_candidates[
            "source1_entity_id"
        ]
        .to_numpy()
    )

    t_ids = (
        val_candidates[
            "matched_entity_id"
        ]
        .to_numpy()
    )

    # --------------------------------------------------------
    # Entity index per candidate
    # --------------------------------------------------------

    entity_indices = np.fromiter(
        (
            entity_to_index[q_id]
            for q_id in q_ids
        ),
        dtype=np.int32,
        count=len(q_ids),
    )

    # --------------------------------------------------------
    # Candidate GT labels
    # --------------------------------------------------------

    print(
        "Building candidate GT labels..."
    )

    candidate_is_true = np.fromiter(
        (
            (
                q_id,
                t_id,
            ) in gt_lookup
            for q_id, t_id
            in zip(
                q_ids,
                t_ids,
            )
        ),
        dtype=np.bool_,
        count=len(q_ids),
    )

    print(
        "Candidate GT positives:",
        f"{int(candidate_is_true.sum()):,}",
    )

    # --------------------------------------------------------
    # Threshold grid
    # --------------------------------------------------------

    thresholds = np.concatenate(
        [
            np.arange(
                0.10,
                0.91,
                0.02,
            ),

            np.arange(
                0.91,
                0.991,
                0.005,
            ),
        ]
    )

    print(
        "Thresholds to evaluate:",
        len(thresholds),
    )

    results = []

    # --------------------------------------------------------
    # Vectorized threshold evaluation
    # --------------------------------------------------------

    for i, threshold in enumerate(
        thresholds,
        start=1,
    ):

        predicted_mask = (
            probabilities
            >= threshold
        )

        predicted_counts = (
            np.bincount(
                entity_indices[
                    predicted_mask
                ],
                minlength=n_entities,
            )
        )

        tp_counts = (
            np.bincount(
                entity_indices[
                    predicted_mask
                    & candidate_is_true
                ],
                minlength=n_entities,
            )
        )

        fp_counts = (
            predicted_counts
            - tp_counts
        )

        fn_counts = (
            true_counts
            - tp_counts
        )

        denominator = (
            1.25 * tp_counts
            + fp_counts
            + 0.25 * fn_counts
        )

        scores = np.ones(
            n_entities,
            dtype=np.float64,
        )

        nonzero = (
            denominator > 0
        )

        scores[nonzero] = (
            1.25
            * tp_counts[nonzero]
            / denominator[nonzero]
        )

        macro_f05 = float(
            scores.mean()
        )

        predicted_total = int(
            predicted_mask.sum()
        )

        results.append(
            (
                float(threshold),
                macro_f05,
                predicted_total,
            )
        )

        # Progress every 10 thresholds.
        if (
            i == 1
            or i % 10 == 0
            or i == len(thresholds)
        ):

            print(
                f"threshold "
                f"{i}/{len(thresholds)} "
                f"done"
            )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    print()
    print(
        "TOP VALIDATION THRESHOLDS"
    )
    print(
        "-" * 70
    )

    for (
        threshold,
        score,
        count,
    ) in results[:15]:

        print(
            f"threshold={threshold:.3f} "
            f"F0.5={score:.6f} "
            f"predictions={count:,}"
        )

    best_threshold = (
        results[0][0]
    )

    best_score = (
        results[0][1]
    )

    print()
    print(
        f"BEST THRESHOLD = "
        f"{best_threshold:.4f}"
    )

    print(
        f"BEST VALIDATION F0.5 = "
        f"{best_score:.6f}"
    )

    print(
        f"Threshold search time = "
        f"{elapsed_minutes(start_time):.2f} min"
    )

    return (
        best_threshold,
        best_score,
        probabilities,
        results,
    )


# ============================================================
# SAVE MODEL CHECKPOINT
# ============================================================

def save_model_checkpoint(
    model,
    feature_names,
):

    print()
    print(
        "=" * 70
    )
    print(
        "SAVING MODEL CHECKPOINT"
    )
    print(
        "=" * 70
    )

    model_path = (
        MODEL_DIR
        / "xgboost_matcher.json"
    )

    model.save_model(
        str(model_path)
    )

    feature_path = (
        MODEL_DIR
        / "feature_names.txt"
    )

    with open(
        feature_path,
        "w",
        encoding="utf-8",
    ) as f:

        for name in feature_names:

            f.write(
                name + "\n"
            )

    print(
        "MODEL SAVED:"
    )

    print(
        model_path
    )

    print(
        "FEATURE NAMES SAVED:"
    )

    print(
        feature_path
    )

    return (
        model_path,
        feature_path,
    )


# ============================================================
# SAVE VALIDATION OUTPUTS
# ============================================================

def save_validation_outputs(
    model,
    candidates,
    X_val,
    val_candidates,
    best_threshold,
    best_score,
    probabilities,
    feature_names,
    threshold_results,
):

    print()
    print(
        "=" * 70
    )
    print(
        "SAVING VALIDATION RESULTS"
    )
    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    threshold_path = (
        MODEL_DIR
        / "threshold.txt"
    )

    with open(
        threshold_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            str(best_threshold)
        )

    # --------------------------------------------------------
    # Threshold table
    # --------------------------------------------------------

    threshold_df = pd.DataFrame(
        threshold_results,
        columns=[
            "threshold",
            "macro_f05",
            "predicted_count",
        ],
    )

    threshold_table_path = (
        MODEL_DIR
        / "threshold_results.tsv"
    )

    threshold_df.to_csv(
        threshold_table_path,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": (
                model.feature_importances_
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance_path = (
        MODEL_DIR
        / "feature_importance.tsv"
    )

    importance_df.to_csv(
        importance_path,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Validation report
    # --------------------------------------------------------

    report_path = (
        MODEL_DIR
        / "validation_report.txt"
    )

    validation_positive_count = int(
        val_candidates[
            "label"
        ].sum()
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "BUSINESS ENTITY RESOLUTION\n"
        )

        f.write(
            "===========================\n\n"
        )

        f.write(
            f"S1 universe: {S1_LIMIT}\n"
        )

        f.write(
            f"Target universe: {TARGET_LIMIT}\n"
        )

        f.write(
            f"Validation S1 entities: "
            f"{val_candidates['source1_entity_id'].nunique()}\n"
        )

        f.write(
            f"Validation candidate rows: "
            f"{len(val_candidates)}\n"
        )

        f.write(
            f"Validation positive candidates: "
            f"{validation_positive_count}\n"
        )

        f.write(
            f"Best threshold: "
            f"{best_threshold}\n"
        )

        f.write(
            f"Macro F0.5: "
            f"{best_score}\n"
        )

        f.write(
            "\nFeatures:\n"
        )

        for name in feature_names:

            f.write(
                f"- {name}\n"
            )

    print(
        "Threshold:",
        threshold_path,
    )

    print(
        "Threshold table:",
        threshold_table_path,
    )

    print(
        "Feature importance:",
        importance_path,
    )

    print(
        "Validation report:",
        report_path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.time()

    print()
    print("=" * 70)
    print(
        "BUSINESS ENTITY RESOLUTION"
    )
    print(
        "FAST ML TRAINING"
    )
    print("=" * 70)

    print()
    print(
        "This run uses the already-tested"
    )

    print(
        "99.7% blocking-recall candidate set."
    )

    # ========================================================
    # 1. LOAD
    # ========================================================

    stage_start = time.time()

    (
        s1,
        s2,
        s3,
        gt,
    ) = load_data()

    print(
        f"Load time: "
        f"{elapsed_minutes(stage_start):.2f} min"
    )

    # ========================================================
    # 2. LIMIT TO THE SUCCESSFUL BLOCKER UNIVERSE
    # ========================================================

    print()
    print("=" * 70)
    print(
        "USING SAME UNIVERSE AS SUCCESSFUL BLOCKER"
    )
    print("=" * 70)

    s1 = (
        s1
        .iloc[
            :S1_LIMIT
        ]
        .copy()
    )

    s2 = (
        s2
        .iloc[
            :TARGET_LIMIT
        ]
        .copy()
    )

    s3 = (
        s3
        .iloc[
            :TARGET_LIMIT
        ]
        .copy()
    )

    print(
        "S1:",
        s1.shape,
    )

    print(
        "S2:",
        s2.shape,
    )

    print(
        "S3:",
        s3.shape,
    )

    # ========================================================
    # 3. PREPARE TEXT
    # ========================================================

    stage_start = time.time()

    print()
    print(
        "Preparing normalized text..."
    )

    s1 = prepare(s1)
    s2 = prepare(s2)
    s3 = prepare(s3)

    print(
        f"Preparation time: "
        f"{elapsed_minutes(stage_start):.2f} min"
    )

    # ========================================================
    # 4. LOAD CANDIDATES
    # ========================================================

    stage_start = time.time()

    candidates = (
        load_candidates()
    )

    print(
        f"Candidate loading time: "
        f"{elapsed_minutes(stage_start):.2f} min"
    )

    # ========================================================
    # 5. GROUND TRUTH
    # ========================================================

    stage_start = time.time()

    gt_pairs = (
        build_gt_set(gt)
    )

    candidates = (
        add_labels(
            candidates,
            gt_pairs,
        )
    )

    print(
        f"Labeling time: "
        f"{elapsed_minutes(stage_start):.2f} min"
    )

    # ========================================================
    # 6. FEATURES
    # ========================================================

    stage_start = time.time()

    (
        X,
        feature_names,
    ) = calculate_features(
        candidates,
        s1,
        s2,
        s3,
    )

    print(
        f"Feature time: "
        f"{elapsed_minutes(stage_start):.2f} min"
    )

    # ========================================================
    # 7. ENTITY SPLIT
    # ========================================================

    (
        train_mask,
        validation_mask,
    ) = make_entity_split(
        candidates
    )

    # ========================================================
    # 8. TRAINING SAMPLE
    # ========================================================

    selected_indices = (
        build_training_subset(
            candidates,
            X,
            train_mask,
        )
    )

    X_train = (
        X[
            selected_indices
        ]
    )

    y_train = (
        candidates.iloc[
            selected_indices
        ]["label"]
        .to_numpy(
            dtype=np.int8
        )
    )

    # ========================================================
    # 9. VALIDATION
    # ========================================================

    validation_indices = (
        np.flatnonzero(
            validation_mask
        )
    )

    X_val = (
        X[
            validation_indices
        ]
    )

    val_candidates = (
        candidates.iloc[
            validation_indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    y_val = (
        val_candidates[
            "label"
        ]
        .to_numpy(
            dtype=np.int8
        )
    )

    print()
    print("=" * 70)
    print(
        "FINAL TRAINING MATRICES"
    )
    print("=" * 70)

    print(
        "X_train:",
        X_train.shape,
    )

    print(
        "y_train positives:",
        int(y_train.sum()),
    )

    print(
        "X_val:",
        X_val.shape,
    )

    print(
        "y_val positives:",
        int(y_val.sum()),
    )

    # ========================================================
    # 10. XGBOOST
    # ========================================================

    print()
    print("=" * 70)
    print(
        "TRAINING XGBOOST"
    )
    print("=" * 70)

    stage_start = time.time()

    model = XGBClassifier(

        objective="binary:logistic",

        n_estimators=N_ESTIMATORS,

        max_depth=MAX_DEPTH,

        learning_rate=LEARNING_RATE,

        subsample=SUBSAMPLE,

        colsample_bytree=COLSAMPLE,

        min_child_weight=3,

        gamma=0.0,

        reg_alpha=0.0,

        reg_lambda=1.0,

        tree_method="hist",

        eval_metric="logloss",

        random_state=RANDOM_SEED,

        n_jobs=-1,

        verbosity=1,
    )

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (
                X_val,
                y_val,
            )
        ],

        verbose=True,
    )

    print()
    print(
        f"XGBoost training time: "
        f"{elapsed_minutes(stage_start):.2f} min"
    )

    # ========================================================
    # 11. SAVE MODEL IMMEDIATELY
    #
    # THIS IS DELIBERATELY BEFORE THRESHOLD SEARCH.
    # ========================================================

    (
        model_path,
        feature_path,
    ) = save_model_checkpoint(
        model,
        feature_names,
    )

    # ========================================================
    # 12. THRESHOLD SEARCH
    # ========================================================

    try:

        (
            best_threshold,
            best_score,
            probabilities,
            threshold_results,
        ) = find_best_threshold(
            model,
            X_val,
            val_candidates,
            gt_pairs,
        )

    except Exception as exc:

        print()
        print("=" * 70)
        print(
            "THRESHOLD SEARCH FAILED"
        )
        print("=" * 70)

        print(
            repr(exc)
        )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "The trained model was already saved."
        )

        print(
            model_path
        )

        print()
        print(
            "You can continue from the saved model."
        )

        raise

    # ========================================================
    # 13. SAVE VALIDATION RESULTS
    # ========================================================

    save_validation_outputs(
        model=model,
        candidates=candidates,
        X_val=X_val,
        val_candidates=val_candidates,
        best_threshold=best_threshold,
        best_score=best_score,
        probabilities=probabilities,
        feature_names=feature_names,
        threshold_results=threshold_results,
    )

    # ========================================================
    # 14. FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "ML TRAINING COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "MODEL:"
    )

    print(
        model_path
    )

    print()
    print(
        "BEST THRESHOLD:"
    )

    print(
        f"{best_threshold:.4f}"
    )

    print()
    print(
        "VALIDATION MACRO F0.5:"
    )

    print(
        f"{best_score:.6f}"
    )

    print()
    print(
        "TOP FEATURES:"
    )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": (
                model.feature_importances_
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print(
        importance.head(
            15
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "MODEL DIRECTORY:"
    )

    print(
        MODEL_DIR
    )

    print()
    print(
        "TOTAL RUNTIME:"
    )

    print(
        f"{elapsed_minutes(overall_start):.2f} minutes"
    )

    print()
    print("=" * 70)
    print(
        "READY FOR TEST INFERENCE"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
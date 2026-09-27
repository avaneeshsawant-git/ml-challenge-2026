 # ============================================================
# fix_threshold.py
#
# Re-evaluates the already-trained XGBoost model using the
# CORRECT validation ground truth:
#
# Only GT pairs whose target is inside the exact 200k target
# universe used by the blocker are eligible.
#
# DOES NOT RETRAIN THE MODEL.
# ============================================================

from pathlib import Path
import sys
import time
import gc
import re

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset" / "train"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "model"

S1_PATH = DATASET_DIR / "train_source1.tsv"
S2_PATH = DATASET_DIR / "train_source2.tsv"
S3_PATH = DATASET_DIR / "train_source3.tsv"
GT_PATH = DATASET_DIR / "train_ground_truth.tsv"

S2_CANDIDATES = (
    OUTPUT_DIR /
    "candidate_pairs_S2_TEST.tsv"
)

S3_CANDIDATES = (
    OUTPUT_DIR /
    "candidate_pairs_S3_TEST.tsv"
)

MODEL_PATH = (
    MODEL_DIR /
    "xgboost_matcher.json"
)

FEATURE_NAMES_PATH = (
    MODEL_DIR /
    "feature_names.txt"
)

THRESHOLD_PATH = (
    MODEL_DIR /
    "threshold.txt"
)

THRESHOLD_RESULTS_PATH = (
    MODEL_DIR /
    "threshold_results.tsv"
)

REPORT_PATH = (
    MODEL_DIR /
    "threshold_validation_report.txt"
)


# ============================================================
# SAME UNIVERSE AS TRAINING
# ============================================================

S1_LIMIT = 10_000
TARGET_LIMIT = 200_000

VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42


# ============================================================
# DEPENDENCY
# ============================================================

try:
    from rapidfuzz import fuzz
except ImportError:

    print("Installing rapidfuzz...")

    import subprocess

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "rapidfuzz",
        ]
    )

    from rapidfuzz import fuzz


try:
    from xgboost import XGBClassifier
except ImportError:

    print("Installing xgboost...")

    import subprocess

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "xgboost",
        ]
    )

    from xgboost import XGBClassifier


# ============================================================
# TEXT HELPERS
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
        min(len(a), len(b))
        /
        max(len(a), len(b))
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
# PREPARE
# ============================================================

def prepare(df):

    df = df.copy()

    df["name_norm"] = (
        df["business_name"]
        .map(norm_text)
    )

    df["address_norm"] = (
        df["business_address"]
        .map(norm_text)
    )

    df["name_sorted"] = (
        df["name_norm"]
        .map(sorted_tokens)
    )

    df["country_norm"] = (
        df["country"]
        .map(norm_text)
    )

    return df


# ============================================================
# FEATURES
#
# EXACTLY THE SAME 22 FEATURES USED DURING TRAINING
# ============================================================

def calculate_features(
    candidates,
    s1,
    s2,
    s3,
):

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

    print()
    print("=" * 70)
    print("REBUILDING VALIDATION FEATURES")
    print("=" * 70)

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

    n = len(candidates)

    X = np.empty(
        (
            n,
            22,
        ),
        dtype=np.float32,
    )

    start_time = time.time()

    for i, row in enumerate(
        candidates.itertuples(
            index=False
        )
    ):

        q = s1_map[
            row.source1_entity_id
        ]

        if row.source == "S2":

            t = s2_map[
                row.matched_entity_id
            ]

        else:

            t = s3_map[
                row.matched_entity_id
            ]

        q_name = q["name_norm"]
        t_name = t["name_norm"]

        q_addr = q["address_norm"]
        t_addr = t["address_norm"]

        q_sorted = q["name_sorted"]
        t_sorted = t["name_sorted"]

        q_country = q["country_norm"]
        t_country = t["country_norm"]

        # ----------------------------------------------------
        # NAME
        # ----------------------------------------------------

        name_ratio = (
            fuzz.ratio(
                q_name,
                t_name,
            ) / 100.0
        )

        name_token_sort = (
            fuzz.token_sort_ratio(
                q_name,
                t_name,
            ) / 100.0
        )

        name_token_set = (
            fuzz.token_set_ratio(
                q_name,
                t_name,
            ) / 100.0
        )

        name_partial = (
            fuzz.partial_ratio(
                q_name,
                t_name,
            ) / 100.0
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

        # ----------------------------------------------------
        # ADDRESS
        # ----------------------------------------------------

        address_ratio = (
            fuzz.ratio(
                q_addr,
                t_addr,
            ) / 100.0
        )

        address_token_sort = (
            fuzz.token_sort_ratio(
                q_addr,
                t_addr,
            ) / 100.0
        )

        address_token_set = (
            fuzz.token_set_ratio(
                q_addr,
                t_addr,
            ) / 100.0
        )

        address_partial = (
            fuzz.partial_ratio(
                q_addr,
                t_addr,
            ) / 100.0
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

        # ----------------------------------------------------
        # COMBINED
        # ----------------------------------------------------

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

        X[i] = [
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

            float(len(q_name)),
            float(len(q_addr)),
        ]

        if (
            (i + 1) % 100_000 == 0
            or i + 1 == n
        ):

            print(
                f"processed "
                f"{i + 1:,}/{n:,}"
            )

    print(
        f"Feature time: "
        f"{(time.time() - start_time) / 60:.2f} min"
    )

    return X


# ============================================================
# BUILD FULL GT
# ============================================================

def build_gt(gt):

    print()
    print(
        "Building GT pair set..."
    )

    gt_pairs = set()

    for row in gt.itertuples(
        index=False
    ):

        q = str(
            row.source1_entity_id
        )

        values = str(
            row.matched_entity_ids
        ).strip()

        if not values:
            continue

        for t in values.split(","):

            t = t.strip()

            if t:

                gt_pairs.add(
                    (
                        q,
                        t,
                    )
                )

    print(
        "Unique GT pairs:",
        f"{len(gt_pairs):,}",
    )

    return gt_pairs


# ============================================================
# RECREATE EXACT SAME 80/20 ENTITY SPLIT
# ============================================================

def validation_entities(
    candidates
):

    entities = (
        candidates[
            "source1_entity_id"
        ]
        .unique()
    )

    rng = np.random.default_rng(
        RANDOM_SEED
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

    val_entities = set(
        entities[
            split_index:
        ]
    )

    print()
    print(
        "Train S1:",
        len(train_entities),
    )

    print(
        "Validation S1:",
        len(val_entities),
    )

    return val_entities


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.time()

    print()
    print("=" * 70)
    print(
        "FIXING VALIDATION THRESHOLD"
    )
    print("=" * 70)

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_PATH}"
        )

    print()
    print(
        "Existing model:"
    )

    print(
        MODEL_PATH
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print()
    print(
        "=" * 70
    )
    print(
        "LOADING DATA"
    )
    print(
        "=" * 70
    )

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

    # EXACT SAME UNIVERSE
    s1 = s1.iloc[
        :S1_LIMIT
    ].copy()

    s2 = s2.iloc[
        :TARGET_LIMIT
    ].copy()

    s3 = s3.iloc[
        :TARGET_LIMIT
    ].copy()

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
    # PREPARE
    # ========================================================

    print()
    print(
        "Preparing text..."
    )

    s1 = prepare(s1)
    s2 = prepare(s2)
    s3 = prepare(s3)

    # ========================================================
    # LOAD CANDIDATES
    # ========================================================

    print()
    print(
        "=" * 70
    )
    print(
        "LOADING EXISTING CANDIDATES"
    )
    print(
        "=" * 70
    )

    c2 = pd.read_csv(
        S2_CANDIDATES,
        sep="\t",
        dtype=str,
    )

    c3 = pd.read_csv(
        S3_CANDIDATES,
        sep="\t",
        dtype=str,
    )

    c2["source"] = "S2"
    c3["source"] = "S3"

    c2.columns = [
        "source1_entity_id",
        "matched_entity_id",
        "source",
    ]

    c3.columns = [
        "source1_entity_id",
        "matched_entity_id",
        "source",
    ]

    candidates = pd.concat(
        [
            c2,
            c3,
        ],
        ignore_index=True,
    )

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
        "Candidates:",
        f"{len(candidates):,}",
    )

    # ========================================================
    # EXACT SAME VALIDATION S1 SPLIT
    # ========================================================

    val_entities = (
        validation_entities(
            candidates
        )
    )

    val_mask = (
        candidates[
            "source1_entity_id"
        ]
        .isin(
            val_entities
        )
        .to_numpy()
    )

    val_candidates = (
        candidates.loc[
            val_mask
        ]
        .reset_index(
            drop=True
        )
    )

    print(
        "Validation candidates:",
        f"{len(val_candidates):,}",
    )

    # ========================================================
    # BUILD GT
    # ========================================================

    gt_pairs = build_gt(
        gt
    )

    # ========================================================
    # IMPORTANT FIX
    #
    # ONLY TARGETS PRESENT IN THE 200K BLOCKER UNIVERSE
    # ========================================================

    print()
    print("=" * 70)
    print(
        "FILTERING GT TO ELIGIBLE BLOCKER UNIVERSE"
    )
    print("=" * 70)

    s2_target_ids = set(
        s2[
            "entity_id"
        ]
    )

    s3_target_ids = set(
        s3[
            "entity_id"
        ]
    )

    eligible_gt = set()

    for q_id, t_id in gt_pairs:

        if q_id not in val_entities:
            continue

        if t_id in s2_target_ids:
            eligible_gt.add(
                (
                    q_id,
                    t_id,
                )
            )

        elif t_id in s3_target_ids:
            eligible_gt.add(
                (
                    q_id,
                    t_id,
                )
            )

    print(
        "FULL GT pairs:",
        f"{len(gt_pairs):,}",
    )

    print(
        "ELIGIBLE validation GT:",
        f"{len(eligible_gt):,}",
    )

    # ========================================================
    # CANDIDATE POSITIVE COUNT
    # ========================================================

    candidate_positive_count = 0

    for row in val_candidates.itertuples(
        index=False
    ):

        if (
            row.source1_entity_id,
            row.matched_entity_id,
        ) in eligible_gt:

            candidate_positive_count += 1

    print(
        "Retrieved candidate positives:",
        f"{candidate_positive_count:,}",
    )

    if len(eligible_gt) > 0:

        blocker_recall = (
            candidate_positive_count
            /
            len(eligible_gt)
        )

        print(
            f"Validation blocker recall: "
            f"{blocker_recall * 100:.4f}%"
        )

    # ========================================================
    # BUILD FEATURES
    # ========================================================

    X_val = calculate_features(
        val_candidates,
        s1,
        s2,
        s3,
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("=" * 70)
    print(
        "LOADING SAVED XGBOOST MODEL"
    )
    print("=" * 70)

    model = XGBClassifier()

    model.load_model(
        str(MODEL_PATH)
    )

    print(
        "Model loaded successfully."
    )

    # ========================================================
    # PREDICT
    # ========================================================

    print()
    print(
        "Generating validation probabilities..."
    )

    probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    print(
        "Prediction complete."
    )

    # ========================================================
    # ENTITY INDEX
    # ========================================================

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

    entity_array = (
        val_candidates[
            "source1_entity_id"
        ]
        .unique()
    )

    entity_to_index = {
        entity_id: i
        for i, entity_id
        in enumerate(
            entity_array
        )
    }

    entity_indices = np.fromiter(
        (
            entity_to_index[
                q
            ]
            for q in q_ids
        ),
        dtype=np.int32,
        count=len(q_ids),
    )

    n_entities = len(
        entity_array
    )

    # ========================================================
    # TRUE PAIR MASK
    # ========================================================

    true_mask = np.fromiter(
        (
            (
                q,
                t,
            ) in eligible_gt
            for q, t
            in zip(
                q_ids,
                t_ids,
            )
        ),
        dtype=np.bool_,
        count=len(q_ids),
    )

    print()
    print(
        "Validation candidate positives:",
        f"{int(true_mask.sum()):,}",
    )

    # ========================================================
    # TRUE COUNTS PER ENTITY
    # ========================================================

    true_counts = np.zeros(
        n_entities,
        dtype=np.int32,
    )

    for q_id, t_id in eligible_gt:

        idx = entity_to_index.get(
            q_id
        )

        if idx is not None:

            true_counts[idx] += 1

    # ========================================================
    # THRESHOLD GRID
    # ========================================================

    thresholds = np.concatenate(
        [
            np.arange(
                0.01,
                0.91,
                0.01,
            ),

            np.arange(
                0.91,
                0.991,
                0.005,
            ),
        ]
    )

    print()
    print("=" * 70)
    print(
        "CORRECTED F0.5 THRESHOLD SEARCH"
    )
    print("=" * 70)

    results = []

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
                    & true_mask
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

        scores[
            nonzero
        ] = (
            1.25
            * tp_counts[
                nonzero
            ]
            /
            denominator[
                nonzero
            ]
        )

        macro_f05 = float(
            scores.mean()
        )

        results.append(
            (
                float(threshold),
                macro_f05,
                int(
                    predicted_mask.sum()
                ),
                int(
                    tp_counts.sum()
                ),
                int(
                    fp_counts.sum()
                ),
                int(
                    fn_counts.sum()
                ),
            )
        )

        if (
            i == 1
            or i % 10 == 0
            or i == len(thresholds)
        ):

            print(
                f"tested "
                f"{i}/{len(thresholds)} "
                f"thresholds"
            )

    # ========================================================
    # SORT
    # ========================================================

    results.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    print()
    print("=" * 70)
    print(
        "TOP THRESHOLDS"
    )
    print("=" * 70)

    print(
        "threshold       F0.5       predicted       TP       FP       FN"
    )

    print(
        "-" * 70
    )

    for result in results[:15]:

        (
            threshold,
            f05,
            predicted,
            tp,
            fp,
            fn,
        ) = result

        print(
            f"{threshold:9.3f}   "
            f"{f05:9.6f}   "
            f"{predicted:10,}   "
            f"{tp:7,}   "
            f"{fp:7,}   "
            f"{fn:7,}"
        )

    # ========================================================
    # BEST
    # ========================================================

    (
        best_threshold,
        best_f05,
        best_predictions,
        best_tp,
        best_fp,
        best_fn,
    ) = results[0]

    print()
    print("=" * 70)
    print(
        "BEST CORRECTED THRESHOLD"
    )
    print("=" * 70)

    print(
        f"Threshold : {best_threshold:.4f}"
    )

    print(
        f"Macro F0.5: {best_f05:.6f}"
    )

    print(
        f"Predicted : {best_predictions:,}"
    )

    print(
        f"TP        : {best_tp:,}"
    )

    print(
        f"FP        : {best_fp:,}"
    )

    print(
        f"FN        : {best_fn:,}"
    )

    # ========================================================
    # SAVE THRESHOLD
    # ========================================================

    with open(
        THRESHOLD_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"{best_threshold:.8f}\n"
        )

    # ========================================================
    # SAVE TABLE
    # ========================================================

    results_df = pd.DataFrame(
        results,
        columns=[
            "threshold",
            "macro_f05",
            "predicted",
            "tp",
            "fp",
            "fn",
        ],
    )

    results_df.to_csv(
        THRESHOLD_RESULTS_PATH,
        sep="\t",
        index=False,
    )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "CORRECTED VALIDATION REPORT\n"
        )

        f.write(
            "===========================\n\n"
        )

        f.write(
            f"S1 universe: {len(s1):,}\n"
        )

        f.write(
            f"S2 universe: {len(s2):,}\n"
        )

        f.write(
            f"S3 universe: {len(s3):,}\n"
        )

        f.write(
            f"Validation S1: "
            f"{len(val_entities):,}\n"
        )

        f.write(
            f"Validation candidates: "
            f"{len(val_candidates):,}\n"
        )

        f.write(
            f"Eligible GT pairs: "
            f"{len(eligible_gt):,}\n"
        )

        f.write(
            f"Retrieved GT pairs: "
            f"{candidate_positive_count:,}\n"
        )

        f.write(
            f"Blocker recall: "
            f"{candidate_positive_count / len(eligible_gt) if eligible_gt else 0:.8f}\n"
        )

        f.write(
            f"Best threshold: "
            f"{best_threshold:.8f}\n"
        )

        f.write(
            f"Macro F0.5: "
            f"{best_f05:.8f}\n"
        )

        f.write(
            f"TP: {best_tp}\n"
        )

        f.write(
            f"FP: {best_fp}\n"
        )

        f.write(
            f"FN: {best_fn}\n"
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print(
        "THRESHOLD FIX COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "CORRECT THRESHOLD:"
    )

    print(
        best_threshold
    )

    print()
    print(
        "MODEL:"
    )

    print(
        MODEL_PATH
    )

    print()
    print(
        "THRESHOLD FILE:"
    )

    print(
        THRESHOLD_PATH
    )

    print()
    print(
        "REPORT:"
    )

    print(
        REPORT_PATH
    )

    print()
    print(
        "Runtime:",
        f"{(time.time() - overall_start) / 60:.2f} min",
    )

    print()
    print(
        "READY FOR TEST INFERENCE."
    )


if __name__ == "__main__":
    main()
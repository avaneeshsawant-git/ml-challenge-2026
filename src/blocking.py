# ============================================================
# BLOCKING TEST v2
# Exact Name + Sorted Name + Exact Address
# ============================================================

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent


# ------------------------------------------------------------
# Load small test sample
# ------------------------------------------------------------

print("Loading data...")

s1 = pd.read_csv(
    PROJECT_ROOT / "dataset/train/train_source1.tsv",
    sep="\t"
).sample(10_000, random_state=42)

s2 = pd.read_csv(
    PROJECT_ROOT / "dataset/train/train_source2.tsv",
    sep="\t",
    nrows=200_000
)

s3 = pd.read_csv(
    PROJECT_ROOT / "dataset/train/train_source3.tsv",
    sep="\t",
    nrows=200_000
)


# ------------------------------------------------------------
# Lightweight normalization
# ------------------------------------------------------------

def normalize(series):
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.normalize("NFKC")
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def prepare(df):

    df = df.copy()

    df["name_norm"] = normalize(df["business_name"])

    df["address_norm"] = normalize(df["business_address"])

    df["name_sorted"] = df["name_norm"].apply(
        lambda x: " ".join(sorted(x.split()))
        if x else ""
    )

    df["country_norm"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Blocking keys
    df["block_name"] = (
        df["country_norm"] + "|" + df["name_norm"]
    )

    df["block_name_sorted"] = (
        df["country_norm"] + "|" + df["name_sorted"]
    )

    df["block_address"] = (
        df["country_norm"] + "|" + df["address_norm"]
    )

    return df


print("Preparing...")

s1 = prepare(s1)
s2 = prepare(s2)
s3 = prepare(s3)


# ------------------------------------------------------------
# Blocking
# ------------------------------------------------------------

def generate_candidates(s1, source):

    blocks = []

    # ==========================================
    # 1. Exact normalized name
    # ==========================================

    left = s1[s1["name_norm"] != ""][
        ["entity_id", "block_name"]
    ]

    right = source[source["name_norm"] != ""][
        ["entity_id", "block_name"]
    ]

    x = left.merge(right, on="block_name")

    x = x[
        ["entity_id_x", "entity_id_y"]
    ]

    print("Exact name:", len(x))

    blocks.append(x)


    # ==========================================
    # 2. Token-sorted name
    # ==========================================

    left = s1[s1["name_sorted"] != ""][
        ["entity_id", "block_name_sorted"]
    ]

    right = source[source["name_sorted"] != ""][
        ["entity_id", "block_name_sorted"]
    ]

    x = left.merge(
        right,
        on="block_name_sorted"
    )

    x = x[
        ["entity_id_x", "entity_id_y"]
    ]

    print("Sorted name:", len(x))

    blocks.append(x)


    # ==========================================
    # 3. Exact normalized address
    # ==========================================

    left = s1[s1["address_norm"] != ""][
        ["entity_id", "block_address"]
    ]

    right = source[source["address_norm"] != ""][
        ["entity_id", "block_address"]
    ]

    x = left.merge(
        right,
        on="block_address"
    )

    x = x[
        ["entity_id_x", "entity_id_y"]
    ]

    print("Exact address:", len(x))

    blocks.append(x)


    # ==========================================
    # UNION
    # ==========================================

    candidates = pd.concat(
        blocks,
        ignore_index=True
    )

    candidates.columns = [
        "source1_entity_id",
        "candidate_entity_id"
    ]

    candidates = candidates.drop_duplicates()

    return candidates


# ------------------------------------------------------------
# S1 -> S2
# ------------------------------------------------------------

print("\n==============================")
print("S1 -> S2")
print("==============================")

candidates_s2 = generate_candidates(
    s1,
    s2
)

print(
    "Unique S2 candidates:",
    len(candidates_s2)
)


# ------------------------------------------------------------
# S1 -> S3
# ------------------------------------------------------------

print("\n==============================")
print("S1 -> S3")
print("==============================")

candidates_s3 = generate_candidates(
    s1,
    s3
)

print(
    "Unique S3 candidates:",
    len(candidates_s3)
)


# ------------------------------------------------------------
# Combined
# ------------------------------------------------------------

candidates = pd.concat(
    [candidates_s2, candidates_s3],
    ignore_index=True
).drop_duplicates()


print("\n==============================")
print("FINAL BLOCKING SUMMARY")
print("==============================")

print("S1:", len(s1))
print("S2:", len(s2))
print("S3:", len(s3))

print("Total candidates:", len(candidates))

print(
    "Average candidates per S1:",
    round(len(candidates) / len(s1), 2)
)

display(candidates.head(30))
import re
import unicodedata
import pandas as pd


# ============================================================
# BASIC NORMALIZATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()

    # punctuation -> spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# TOKENS
# ============================================================

def get_tokens(text):
    if not text:
        return []

    return text.split()


def get_token_sorted(text):
    tokens = get_tokens(text)
    return " ".join(sorted(tokens))


# ============================================================
# NUMBERS
# ============================================================

def get_numbers(text):
    if not text:
        return []

    return re.findall(r"\d+", text)


# ============================================================
# LEGAL NAME CANONICALIZATION
# Keep this SEPARATE from normal name.
# ============================================================

LEGAL_SUFFIXES = {
    "private limited": "private",
    "pvt limited": "private",
    "pvt ltd": "private",
    "private ltd": "private",
    "limited": "limited",
    "ltd": "limited",
    "incorporated": "inc",
    "corporation": "corp",
}


def canonicalize_legal_name(text):
    if not text:
        return ""

    result = text

    # longest phrases first
    for old, new in sorted(
        LEGAL_SUFFIXES.items(),
        key=lambda x: len(x[0]),
        reverse=True
    ):
        result = re.sub(
            rf"\b{re.escape(old)}\b",
            new,
            result
        )

    result = re.sub(r"\s+", " ", result).strip()

    return result


# ============================================================
# ADDRESS CANONICALIZATION
# Conservative abbreviations only.
# ============================================================

ADDRESS_ABBREVIATIONS = {
    "road": "rd",
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "highway": "hwy",
    "parkway": "pkwy",
    "place": "pl",
    "court": "ct",
    "circle": "cir",
    "apartment": "apt",
    "suite": "ste",
}


def canonicalize_address(text):
    if not text:
        return ""

    result = text

    for old, new in ADDRESS_ABBREVIATIONS.items():
        result = re.sub(
            rf"\b{old}\b",
            new,
            result
        )

    result = re.sub(r"\s+", " ", result).strip()

    return result


# ============================================================
# SCRIPT DETECTION
# ============================================================

def detect_script(text):
    # Handle missing / NaN values
    if pd.isna(text):
        return "missing"

    text = str(text).strip()

    if not text:
        return "missing"

    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    has_gujarati = bool(re.search(r"[\u0A80-\u0AFF]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))

    if has_devanagari:
        return "devanagari"

    if has_gujarati:
        return "gujarati"

    if has_latin:
        return "latin"

    return "other"


# ============================================================
# DOMAIN / NUMBER INDICATORS
# ============================================================

def contains_domain(text):
    if not text:
        return False

    return bool(
        re.search(
            r"\b[\w.-]+\.(com|in|org|net|co|biz)\b",
            text.lower()
        )
    )


# ============================================================
# MAIN PREPROCESSING
# ============================================================

def preprocess(df):

    df = df.copy()

    # --------------------------------------------------------
    # Basic normalized representations
    # --------------------------------------------------------

    df["business_name_norm"] = (
        df["business_name"]
        .apply(normalize_text)
    )

    df["business_address_norm"] = (
        df["business_address"]
        .apply(normalize_text)
    )

    # --------------------------------------------------------
    # Name features
    # --------------------------------------------------------

    df["name_tokens"] = (
        df["business_name_norm"]
        .apply(get_tokens)
    )

    df["name_token_sorted"] = (
        df["business_name_norm"]
        .apply(get_token_sorted)
    )

    df["business_name_legal_norm"] = (
        df["business_name_norm"]
        .apply(canonicalize_legal_name)
    )

    # --------------------------------------------------------
    # Address features
    # --------------------------------------------------------

    df["address_tokens"] = (
        df["business_address_norm"]
        .apply(get_tokens)
    )

    df["address_token_sorted"] = (
        df["business_address_norm"]
        .apply(get_token_sorted)
    )

    df["address_canonical"] = (
        df["business_address_norm"]
        .apply(canonicalize_address)
    )

    df["address_numbers"] = (
        df["business_address_norm"]
        .apply(get_numbers)
    )

    # --------------------------------------------------------
    # Missingness
    # --------------------------------------------------------

    df["name_missing"] = (
        df["business_name_norm"] == ""
    )

    df["address_missing"] = (
        df["business_address_norm"] == ""
    )

    # --------------------------------------------------------
    # Script
    # --------------------------------------------------------

    df["name_script"] = (
        df["business_name"]
        .apply(detect_script)
    )

    df["address_script"] = (
        df["business_address"]
        .apply(detect_script)
    )

    # --------------------------------------------------------
    # Domain indicator
    # --------------------------------------------------------

    df["name_contains_domain"] = (
        df["business_name_norm"]
        .apply(contains_domain)
    )

    return df
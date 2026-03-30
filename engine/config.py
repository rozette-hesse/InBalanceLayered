from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Final user-facing phases
PHASES = ["Menstrual", "Follicular", "Fertility", "Luteal"]

# Non-menstrual phases predicted by Layer 2 / fusion
NON_MENSTRUAL_PHASES = ["Follicular", "Fertility", "Luteal"]

SUPPORTED_SYMPTOMS = [
    "headaches",
    "cramps",
    "sorebreasts",
    "fatigue",
    "sleepissue",
    "moodswing",
    "stress",
    "foodcravings",
    "indigestion",
    "bloating",
]

MUCUS_OPTIONS = ["dry", "sticky", "creamy", "eggwhite", "watery", "unknown"]

# Tuned from your notebook
LAYER1_WEIGHT = 0.2
LAYER2_WEIGHT = 0.8

# New Layer 2 artifacts
LAYER2_PIPELINE_FILE = "layer2_vnext_pipeline.joblib"
LAYER2_LABEL_ENCODER_FILE = "layer2_vnext_label_encoder.joblib"
LAYER2_FEATURE_COLUMNS_FILE = "layer2_vnext_feature_columns.joblib"
LAYER2_METADATA_FILE = "layer2_vnext_metadata.joblib"

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

PHASES = ["Menstrual", "Follicular", "Fertility", "Luteal"]

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

PCA_EXTRA_INPUTS = [
    "appetite",
    "exerciselevel",
]

ALL_LAYER2_INPUTS = PCA_EXTRA_INPUTS + SUPPORTED_SYMPTOMS

MUCUS_OPTIONS = ["dry", "sticky", "creamy", "eggwhite", "watery", "unknown"]

LAYER1_WEIGHT = 0.5
LAYER2_WEIGHT = 0.5

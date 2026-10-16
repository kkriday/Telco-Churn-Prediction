"""Project-wide settings: paths, random seed, and business assumptions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SEARCH_ITERATIONS = 25

TARGET = "Churn"
ID_COL = "customerID"

# ---------------------------------------------------------------------------
# Retention campaign economics (illustrative assumptions, edit to taste).
# A flagged customer receives a retention offer. The model is judged by the
# net value of that campaign versus doing nothing.
# ---------------------------------------------------------------------------
OFFER_COST = 50.0          # cost of one retention offer, in dollars
SAVE_RATE = 0.30           # share of true churners the offer actually keeps
MONTHS_RETAINED = 12       # months of revenue kept when a churner is saved

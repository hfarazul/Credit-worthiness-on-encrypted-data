"All constants used in the project."

from pathlib import Path
import pandas

# The directory of this project
REPO_DIR = Path(__file__).parent

# Main necessary directories
DEPLOYMENT_PATH = REPO_DIR / "deployment_files"
FHE_KEYS = REPO_DIR / ".fhe_keys"
CLIENT_FILES = REPO_DIR / "client_files"
SERVER_FILES = REPO_DIR / "server_files"

# ALl deployment directories
DEPLOYMENT_PATH = DEPLOYMENT_PATH / "model"

# Path targeting pre-processor saved files
PRE_PROCESSOR_APPLICANT_PATH = DEPLOYMENT_PATH / 'pre_processor_applicant.pkl'
PRE_PROCESSOR_BANK_PATH = DEPLOYMENT_PATH / 'pre_processor_bank.pkl'
PRE_PROCESSOR_CREDIT_BUREAU_PATH = DEPLOYMENT_PATH / 'pre_processor_credit_bureau.pkl'

# Create the necessary directories
FHE_KEYS.mkdir(exist_ok=True)
CLIENT_FILES.mkdir(exist_ok=True)
SERVER_FILES.mkdir(exist_ok=True)

# Store the server's URL
SERVER_URL = "http://localhost:8000/"

# Path to data file
DATA_PATH = "Credit underwriting data.csv"

# Development settings
PROCESSED_INPUT_SHAPE = (1, 39)

CLIENT_TYPES = ["applicant", "bank", "credit_bureau"]
INPUT_INDEXES = {
    "applicant": 0,
    "bank": 1,
    "credit_bureau": 2,
}
INPUT_SLICES = {
    "applicant": slice(0, 12),  # First position: start from 0
    "bank": slice(12, 14),  # Second position: start from n_feature_applicant
    "credit_bureau": slice(14, 27),  # Third position: start from n_feature_applicant + n_feature_bank
}

# Fix column order for pre-processing steps
APPLICANT_COLUMNS = [
'age','education_level','dependents','employment_status','total_employment_years','annual_salary','monthly_salary'
]
BANK_COLUMNS = [
   'last_6_months_avg_balance',	'monthly_expenses'
]
CREDIT_BUREAU_COLUMNS = [
   'existing_loans_count','assets_value','CIBIL_score','total_credit_limit','current_total_balances','credit_utilization_ratio','payment_history','payment_delays','recent_enquiries'
]

_data = pandas.read_csv(DATA_PATH, encoding="utf-8")

def get_min_max(data, column):
    """Get min/max values of a column in order to input them in Gradio's API as key arguments."""
    return {
        "minimum": int(data[column].min()),
        "maximum": int(data[column].max()),
    }

# # App data min and max values
# ACCOUNT_MIN_MAX = get_min_max(_data, "Account_age")
# CHILDREN_MIN_MAX = get_min_max(_data, "Num_children")
# INCOME_MIN_MAX = get_min_max(_data, "Total_income")
# AGE_MIN_MAX = get_min_max(_data, "Age")
# FAMILY_MIN_MAX = get_min_max(_data, "Household_size")

# # Default values
# INCOME_VALUE = 12000
# AGE_VALUE = 30

# # App data choices
# INCOME_TYPES = list(_data["Income_type"].unique())
# OCCUPATION_TYPES = list(_data["Occupation_type"].unique())
# HOUSING_TYPES = list(_data["Housing_type"].unique())
# EDUCATION_TYPES = list(_data["Education_type"].unique())
# FAMILY_STATUS = list(_data["Family_status"].unique())
# YEARS_EMPLOYED_BINS = ['0-2', '2-5', '5-8', '8-11', '11-18', '18+']

# # Years_employed bin order
# YEARS_EMPLOYED_BIN_NAME_TO_INDEX = {bin_name: i for i, bin_name in enumerate(YEARS_EMPLOYED_BINS)}

# assert len(YEARS_EMPLOYED_BINS) == len(list(_data["Years_employed"].unique())), (
#     "Years_employed bins are not matching the expected list"
# )

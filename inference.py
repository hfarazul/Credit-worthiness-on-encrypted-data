import pickle
import numpy as np
import pandas as pd
from settings import (
    DEPLOYMENT_PATH,
    PRE_PROCESSOR_APPLICANT_PATH,
    PRE_PROCESSOR_BANK_PATH,
    PRE_PROCESSOR_CREDIT_BUREAU_PATH,
)
from utils.client_server_interface import MultiInputsFHEModelServer
from backend import (
    keygen_send,
    pre_process_encrypt_send_applicant,
    pre_process_encrypt_send_bank,
    pre_process_encrypt_send_credit_bureau,
    run_fhe,
    get_output_and_decrypt,
    explain_encrypt_run_decrypt,
)

# Load the pre-processors
with (
    PRE_PROCESSOR_APPLICANT_PATH.open('rb') as file_applicant,
    PRE_PROCESSOR_BANK_PATH.open('rb') as file_bank,
    PRE_PROCESSOR_CREDIT_BUREAU_PATH.open('rb') as file_credit_bureau,
):
    pre_processor_applicant = pickle.load(file_applicant)
    pre_processor_bank = pickle.load(file_bank)
    pre_processor_credit_bureau = pickle.load(file_credit_bureau)

# Load the trained model
model_server = MultiInputsFHEModelServer(DEPLOYMENT_PATH)

def preprocess_input(applicant_data, bank_data, credit_bureau_data):
    # Preprocess the input data using the loaded pre-processors
    preprocessed_applicant_data = pre_processor_applicant.transform(applicant_data)
    preprocessed_bank_data = pre_processor_bank.transform(bank_data)
    preprocessed_credit_bureau_data = pre_processor_credit_bureau.transform(credit_bureau_data)

    return preprocessed_applicant_data, preprocessed_bank_data, preprocessed_credit_bureau_data

def make_inference(applicant_data, bank_data, credit_bureau_data):
    # Generate keys and send evaluation key to the server
    client_id, _ = keygen_send()

    # Preprocess, encrypt, and send input data to the server
    _, _ = pre_process_encrypt_send_applicant(client_id, *applicant_data.values.flatten())
    _, _ = pre_process_encrypt_send_bank(client_id, *bank_data.values.flatten())
    _, _ = pre_process_encrypt_send_credit_bureau(client_id, *credit_bureau_data.values.flatten())

    # Run FHE evaluation on the server
    _ = run_fhe(client_id)

    # Receive the encrypted output from the server and decrypt
    prediction, _ = get_output_and_decrypt(client_id)

    return prediction

# Example usage
applicant_data = pd.DataFrame({
    'age': [35],
    'education_level': ["Bachelor's Degree"],
    'dependents': [2],
    'employment_status': ['Salaried'],
    'total_employment_years': [5],
    'annual_salary': [50000],
    'monthly_salary': [4000]
})

bank_data = pd.DataFrame({
    'last_6_months_avg_balance': [10000],
    'monthly_expenses': [2000]
})

credit_bureau_data = pd.DataFrame({
    'existing_loans_count': [1],
    'assets_value': [100000],
    'CIBIL_score': [750],
    'total_credit_limit': [50000],
    'current_total_balances': [20000],
    'credit_utilization_ratio': [0.4],
    'payment_history': [5],
    'payment_delays': [0],
    'recent_enquiries': [2]
})

prediction = make_inference(applicant_data, bank_data, credit_bureau_data)
print("Prediction:", prediction)

"""Backend functions used in the app."""

import os
import shutil
import gradio as gr
import numpy
import requests
import pickle
import pandas
from itertools import chain

from settings import (
    SERVER_URL,
    FHE_KEYS,
    CLIENT_FILES,
    SERVER_FILES,
    DEPLOYMENT_PATH,
    PROCESSED_INPUT_SHAPE,
    INPUT_INDEXES,
    INPUT_SLICES,
    PRE_PROCESSOR_APPLICANT_PATH,
    PRE_PROCESSOR_BANK_PATH,
    PRE_PROCESSOR_CREDIT_BUREAU_PATH,
    CLIENT_TYPES,
    APPLICANT_COLUMNS,
    BANK_COLUMNS,
    CREDIT_BUREAU_COLUMNS,
    #YEARS_EMPLOYED_BINS,
    #YEARS_EMPLOYED_BIN_NAME_TO_INDEX,
)

from utils.client_server_interface import MultiInputsFHEModelClient

# Define the messages associated to the predictions
APPROVED_MESSAGE = "Credit card is likely to be approved ✅"
DENIED_MESSAGE = "Credit card is likely to be denied ❌"

# Load pre-processor instances
with (
    PRE_PROCESSOR_APPLICANT_PATH.open('rb') as file_applicant,
    PRE_PROCESSOR_BANK_PATH.open('rb') as file_bank,
    PRE_PROCESSOR_CREDIT_BUREAU_PATH.open('rb') as file_credit_bureau,
):
    PRE_PROCESSOR_APPLICANT = pickle.load(file_applicant)
    PRE_PROCESSOR_BANK = pickle.load(file_bank)
    PRE_PROCESSOR_CREDIT_BUREAU = pickle.load(file_credit_bureau)


def shorten_bytes_object(bytes_object, limit=500):
    """Shorten the input bytes object to a given length.

    Encrypted data is too large for displaying it in the browser using Gradio. This function
    provides a shorten representation of it.

    Args:
        bytes_object (bytes): The input to shorten
        limit (int): The length to consider. Default to 500.

    Returns:
        str: Hexadecimal string shorten representation of the input byte object.

    """
    # Define a shift for better display
    shift = 100
    return bytes_object[shift : limit + shift].hex()


def clean_temporary_files(n_keys=20):
    """Clean older keys and encrypted files.

    A maximum of n_keys keys and associated temporary files are allowed to be stored. Once this
    limit is reached, the oldest files are deleted.

    Args:
        n_keys (int): The maximum number of keys and associated files to be stored. Default to 20.

    """
    # Get the oldest key files in the key directory
    key_dirs = sorted(FHE_KEYS.iterdir(), key=os.path.getmtime)

    # If more than n_keys keys are found, remove the oldest
    client_ids = []
    if len(key_dirs) > n_keys:
        n_keys_to_delete = len(key_dirs) - n_keys
        for key_dir in key_dirs[:n_keys_to_delete]:
            client_ids.append(key_dir.name)
            shutil.rmtree(key_dir)

    # Delete all files related to the IDs whose keys were deleted
    for directory in chain(CLIENT_FILES.iterdir(), SERVER_FILES.iterdir()):
        for client_id in client_ids:
            if client_id in directory.name:
                shutil.rmtree(directory)


def _get_client(client_id):
    """Get the client instance.

    Args:
        client_id (int): The client ID to consider.

    Returns:
        FHEModelClient: The client instance.
    """
    key_dir = FHE_KEYS / f"{client_id}"

    return MultiInputsFHEModelClient(DEPLOYMENT_PATH, key_dir=key_dir, nb_inputs=len(CLIENT_TYPES))


def _get_client_file_path(name, client_id, client_type=None):
    """Get the file path for the client.

    Args:
        name (str): The desired file name (either 'evaluation_key', 'encrypted_inputs' or
            'encrypted_outputs').
        client_id (int): The client ID to consider.
        client_type (Optional[str]): The type of client to consider (either 'applicant', 'bank',
            'credit_bureau' or None). Default to None, which is used for evaluation key and output.

    Returns:
        pathlib.Path: The file path.
    """
    client_type_suffix = ""
    if client_type is not None:
        client_type_suffix = f"_{client_type}"

    dir_path = CLIENT_FILES / f"{client_id}"
    dir_path.mkdir(exist_ok=True)

    return dir_path / f"{name}{client_type_suffix}"


def _send_to_server(client_id, client_type, file_name):
    """Send the encrypted inputs or the evaluation key to the server.

    Args:
        client_id (int): The client ID to consider.
        client_type (Optional[str]): The type of client to consider (either 'applicant', 'bank',
            'credit_bureau' or None).
        file_name (str): File name to send (either 'evaluation_key' or 'encrypted_inputs').
    """
    # Get the paths to the encrypted inputs
    encrypted_file_path = _get_client_file_path(file_name, client_id, client_type)

    # Define the data and files to post
    data = {
        "client_id": client_id,
        "client_type": client_type,
        "file_name": file_name,
    }

    files = [
        ("files", open(encrypted_file_path, "rb")),
    ]

    # Send the encrypted inputs or evaluation key to the server
    url = SERVER_URL + "send_file"
    with requests.post(
        url=url,
        data=data,
        files=files,
    ) as response:
        return response.ok


def keygen_send():
    """Generate the private and evaluation key, and send the evaluation key to the server.

    Returns:
        client_id (str): The current client ID to consider.
    """
    # Clean temporary files
    clean_temporary_files()

    # Create an ID for the current client to consider
    client_id = numpy.random.randint(0, 2**32)

    # Retrieve the client instance
    client = _get_client(client_id)

    # Generate the private and evaluation keys
    client.generate_private_and_evaluation_keys(force=True)

    # Retrieve the serialized evaluation key
    evaluation_key = client.get_serialized_evaluation_keys()

    file_name = "evaluation_key"

    # Save evaluation key as bytes in a file as it is too large to pass through regular Gradio
    # buttons (see https://github.com/gradio-app/gradio/issues/1877)
    evaluation_key_path = _get_client_file_path(file_name, client_id)

    with evaluation_key_path.open("wb") as evaluation_key_file:
        evaluation_key_file.write(evaluation_key)

    # Send the evaluation key to the server
    _send_to_server(client_id, None, file_name)

    # Create a truncated version of the evaluation key for display
    evaluation_key_short = shorten_bytes_object(evaluation_key)

    return client_id, evaluation_key_short, gr.update(value="Keys are generated and evaluation key is sent ✅")


def _encrypt_send(client_id, inputs, client_type):
    """Encrypt the given inputs for a specific client and send it to the server.

    Args:
        client_id (str): The current client ID to consider.
        inputs (numpy.ndarray): The inputs to encrypt.
        client_type (str): The type of client to consider (either 'applicant', 'bank' or
            'credit_bureau').

    Returns:
        encrypted_inputs_short (str): A short representation of the encrypted input to send in hex.
    """
    if client_id == "":
        raise gr.Error("Please generate the keys first.")

    # Retrieve the client instance
    client = _get_client(client_id)

    # Quantize, encrypt and serialize the inputs
    encrypted_inputs = client.quantize_encrypt_serialize_multi_inputs(
        inputs,
        input_index=INPUT_INDEXES[client_type],
        processed_input_shape=PROCESSED_INPUT_SHAPE,
        input_slice=INPUT_SLICES[client_type],
    )

    file_name = "encrypted_inputs"

    # Save encrypted_inputs to bytes in a file, since too large to pass through regular Gradio
    # buttons, https://github.com/gradio-app/gradio/issues/1877
    encrypted_inputs_path = _get_client_file_path(file_name, client_id, client_type)

    with encrypted_inputs_path.open("wb") as encrypted_inputs_file:
        encrypted_inputs_file.write(encrypted_inputs)

    # Create a truncated version of the encrypted inputs for display
    encrypted_inputs_short = shorten_bytes_object(encrypted_inputs)

    _send_to_server(client_id, client_type, file_name)

    return encrypted_inputs_short, gr.update(value="Inputs are encrypted and sent to server. ✅")


# ... (Previous imports and code remain the same)

def pre_process_encrypt_send_applicant(client_id, *inputs):
    """Pre-process, encrypt and send the applicant inputs for a specific client to the server.

    Args:
        client_id (str): The current client ID to consider.
        *inputs (Tuple[numpy.ndarray]): The inputs to pre-process.

    Returns:
        (str): A short representation of the encrypted input to send in hex.
    """
    age, education_level, dependents, employment_status, total_employment_years, annual_salary, monthly_salary = inputs

    applicant_inputs = pandas.DataFrame({
        "age": [age],
        "education_level": [education_level],
        "dependents": [dependents],
        "employment_status": [employment_status],
        "total_employment_years": [total_employment_years],
        "annual_salary": [annual_salary],
        "monthly_salary": [monthly_salary],
    })

    applicant_inputs = applicant_inputs.reindex(APPLICANT_COLUMNS, axis=1)
    preprocessed_applicant_inputs = PRE_PROCESSOR_APPLICANT.transform(applicant_inputs)

    return _encrypt_send(client_id, preprocessed_applicant_inputs, "applicant")


def pre_process_encrypt_send_bank(client_id, *inputs):
    """Pre-process, encrypt and send the bank inputs for a specific client to the server.

    Args:
        client_id (str): The current client ID to consider.
        *inputs (Tuple[numpy.ndarray]): The inputs to pre-process.

    Returns:
        (str): A short representation of the encrypted input to send in hex.
    """
    last_6_months_avg_balance, monthly_expenses = inputs

    bank_inputs = pandas.DataFrame({
        "last_6_months_avg_balance": [last_6_months_avg_balance],
        "monthly_expenses": [monthly_expenses],
    })

    bank_inputs = bank_inputs.reindex(BANK_COLUMNS, axis=1)
    preprocessed_bank_inputs = PRE_PROCESSOR_BANK.transform(bank_inputs)

    return _encrypt_send(client_id, preprocessed_bank_inputs, "bank")


def pre_process_encrypt_send_credit_bureau(client_id, *inputs):
    """Pre-process, encrypt and send the credit bureau inputs for a specific client to the server.

    Args:
        client_id (str): The current client ID to consider.
        *inputs (Tuple[numpy.ndarray]): The inputs to pre-process.

    Returns:
        (str): A short representation of the encrypted input to send in hex.
    """
    existing_loans_count, assets_value, cibil_score, total_credit_limit, current_total_balances, credit_utilization_ratio, payment_history, payment_delays, recent_enquiries = inputs

    credit_bureau_inputs = pandas.DataFrame({
        "existing_loans_count": [existing_loans_count],
        "assets_value": [assets_value],
        "CIBIL_score": [cibil_score],
        "total_credit_limit": [total_credit_limit],
        "current_total_balances": [current_total_balances],
        "credit_utilization_ratio": [credit_utilization_ratio],
        "payment_history": [payment_history],
        "payment_delays": [payment_delays],
        "recent_enquiries": [recent_enquiries],
    })

    credit_bureau_inputs = credit_bureau_inputs.reindex(CREDIT_BUREAU_COLUMNS, axis=1)
    preprocessed_credit_bureau_inputs = PRE_PROCESSOR_CREDIT_BUREAU.transform(credit_bureau_inputs)

    return _encrypt_send(client_id, preprocessed_credit_bureau_inputs, "credit_bureau")

# ... (Rest of the code remains the same)




# def pre_process_encrypt_send_applicant(client_id, *inputs):
#     """Pre-process, encrypt and send the applicant inputs for a specific client to the server.

#     Args:
#         client_id (str): The current client ID to consider.
#         *inputs (Tuple[numpy.ndarray]): The inputs to pre-process.

#     Returns:
#         (str): A short representation of the encrypted input to send in hex.
#     """
#     bool_inputs, num_children, household_size, total_income, age, income_type, education_type, \
#         family_status, occupation_type, housing_type = inputs

#     # Retrieve boolean values
#     own_car = "Car" in bool_inputs
#     own_property = "Property" in bool_inputs
#     mobile_phone = "Mobile phone" in bool_inputs

#     applicant_inputs = pandas.DataFrame({
#         "Own_car": [own_car],
#         "Own_property": [own_property],
#         "Mobile_phone": [mobile_phone],
#         "Num_children": [num_children],
#         "Household_size": [household_size],
#         "Total_income": [total_income],
#         "Age": [age],
#         "Income_type": [income_type],
#         "Education_type": [education_type],
#         "Family_status": [family_status],
#         "Occupation_type": [occupation_type],
#         "Housing_type": [housing_type],
#     })

#     applicant_inputs = applicant_inputs.reindex(APPLICANT_COLUMNS, axis=1)

#     preprocessed_applicant_inputs = PRE_PROCESSOR_APPLICANT.transform(applicant_inputs)

#     return _encrypt_send(client_id, preprocessed_applicant_inputs, "applicant")


# def pre_process_encrypt_send_bank(client_id, *inputs):
#     """Pre-process, encrypt and send the bank inputs for a specific client to the server.

#     Args:
#         client_id (str): The current client ID to consider.
#         *inputs (Tuple[numpy.ndarray]): The inputs to pre-process.

#     Returns:
#         (str): A short representation of the encrypted input to send in hex.
#     """
#     account_age = inputs[0]

#     bank_inputs = pandas.DataFrame({
#         "Account_age": [account_age],
#     })

#     bank_inputs = bank_inputs.reindex(BANK_COLUMNS, axis=1)

#     preprocessed_bank_inputs = PRE_PROCESSOR_BANK.transform(bank_inputs)

#     return _encrypt_send(client_id, preprocessed_bank_inputs, "bank")


# def pre_process_encrypt_send_credit_bureau(client_id, *inputs):
#     """Pre-process, encrypt and send the credit bureau inputs for a specific client to the server.

#     Args:
#         client_id (str): The current client ID to consider.
#         *inputs (Tuple[numpy.ndarray]): The inputs to pre-process.

#     Returns:
#         (str): A short representation of the encrypted input to send in hex.
#     """
#     years_employed_bin, employed = inputs

#     years_employed = YEARS_EMPLOYED_BIN_NAME_TO_INDEX[years_employed_bin]
#     is_employed = employed == "Yes"

#     credit_bureau_inputs = pandas.DataFrame({
#         "Years_employed": [years_employed],
#         "Employed": [is_employed],
#     })

#     credit_bureau_inputs = credit_bureau_inputs.reindex(CREDIT_BUREAU_COLUMNS, axis=1)
#     preprocessed_credit_bureau_inputs = PRE_PROCESSOR_CREDIT_BUREAU.transform(credit_bureau_inputs)

#     return _encrypt_send(client_id, preprocessed_credit_bureau_inputs, "credit_bureau")


def run_fhe(client_id):
    """Run the model on the encrypted inputs previously sent using FHE.

    Args:
        client_id (str): The current client ID to consider.
    """

    if client_id == "":
        raise gr.Error("Please generate the keys first.")

    data = {
        "client_id": client_id,
    }

    # Trigger the FHE execution on the encrypted inputs previously sent
    url = SERVER_URL + "run_fhe"
    with requests.post(
        url=url,
        data=data,
    ) as response:
        if response.ok:
            return response.json(), gr.update(value="FHE evaluation is done. ✅")
        else:
            raise gr.Error("Please send the inputs from all three parties to the server first.")


def get_output_and_decrypt(client_id):
    """Retrieve the encrypted output.

    Args:
        client_id (str): The current client ID to consider.

    Returns:
        (Tuple[str, bytes]): The output message based on the decrypted prediction as well as
            a byte short representation of the encrypted output.
    """

    if client_id == "":
        raise gr.Error("Please generate the keys first.")

    data = {
        "client_id": client_id,
    }

    # Retrieve the encrypted output
    url = SERVER_URL + "get_output"
    with requests.post(
        url=url,
        data=data,
    ) as response:
        if response.ok:
            encrypted_output_proba = response.content

            # Create a truncated version of the encrypted inputs for display
            encrypted_output_short = shorten_bytes_object(encrypted_output_proba)

            # Retrieve the client API
            client = _get_client(client_id)

            # Deserialize, decrypt and post-process the encrypted output
            output_proba = client.deserialize_decrypt_dequantize(encrypted_output_proba)

            # Determine the predicted class
            output = numpy.argmax(output_proba, axis=1).squeeze()

            return (
                APPROVED_MESSAGE if output == 1 else DENIED_MESSAGE,
                encrypted_output_short,
                gr.update(value="Encrypted outputs have been received from the server. ✅"),
            )

        else:
            raise gr.Error("Please run the FHE execution first and wait for it to be completed.")


def explain_encrypt_run_decrypt(client_id, prediction_output, *inputs):
    """Pre-process and encrypt the inputs, run the prediction in FHE and decrypt the output.

    Args:
        client_id (str): The current client ID to consider.
        prediction_output (str): The initial prediction output. This parameter is only used to
            throw an error in case the prediction was positive.
        *inputs (Tuple[numpy.ndarray]): The inputs to consider.

    Returns:
        (str): A message indicating the number of additional years of employment that could be
            required in order to increase the chance of credit card approval.
    """

    if "approved" in prediction_output:
        raise gr.Error(
            "Explaining the prediction can only be done if the credit card is likely to be denied."
        )

    button_update = gr.update(value="Prediction has been explained. ✅")

    # Retrieve the credit bureau inputs
    years_employed, employed = inputs

    # Years_employed is divided into several ordered bins. Here, we retrieve the index representing
    # the bin from the input
    bin_index = YEARS_EMPLOYED_BIN_NAME_TO_INDEX[years_employed]

    # If the bin is not the last (representing the most years of employment), we run the model in
    # FHE for each bins "older" or equal to the given bin, in order. Then, we retrieve the first
    # bin that changes the model's prediction to "approval" and display it to the applicant.
    if bin_index != len(YEARS_EMPLOYED_BINS) - 1:

        # Loop over the bins starting with "older" or equal to the given bin
        for years_employed_bin in YEARS_EMPLOYED_BINS[bin_index:]:

            # Send the new encrypted input
            pre_process_encrypt_send_credit_bureau(client_id, years_employed_bin, employed)

            # Run the model in FHE
            run_fhe(client_id)

            # Retrieve the new prediction
            output_prediction = get_output_and_decrypt(client_id)

            # If the bin made the model predict an approval, share it to the applicant
            if "approved" in output_prediction[0]:

                # If the approval was made using the given input, that means the applicant most
                # likely tried the bin suggested in a previous explainability run. In that case, we
                # confirm that the credit card is likely to be approved
                if years_employed_bin == years_employed:
                    return APPROVED_MESSAGE, button_update

                # Else, that means the applicant is looking for some explainability. We therefore
                # suggest to try the obtained bin
                return (
                    DENIED_MESSAGE + f" However, having at least {years_employed_bin} years of "
                    "employment would increase your chance of having your credit card approved."
                ), button_update

        # In case no bins made the model predict an approval, explain why
        return (
            DENIED_MESSAGE + " Unfortunately, increasing the number of years of employment up to "
            f"{YEARS_EMPLOYED_BINS[-1]} years does not seem to be enough to get an approval based "
            "on the given inputs. Other inputs like the income or the account's age might have "
            "bigger impact in this particular case."
        ), button_update

    # In case the applicant tried the "oldest" bin (but still got denied), explain why
    return (
        DENIED_MESSAGE + " Unfortunately, you already have the maximum amount of years of "
        f"employment ({years_employed} years). Other inputs like the income or the account's age "
        "might have a bigger impact in this particular case."
    ), button_update

"""Train and compile the model."""

import shutil
import numpy
import pandas
import pickle

from settings import (
    DEPLOYMENT_PATH,
    DATA_PATH,
    INPUT_SLICES,
    PRE_PROCESSOR_APPLICANT_PATH,
    PRE_PROCESSOR_BANK_PATH,
    PRE_PROCESSOR_CREDIT_BUREAU_PATH,
    APPLICANT_COLUMNS,
    BANK_COLUMNS,
    CREDIT_BUREAU_COLUMNS,
)
from utils.client_server_interface import MultiInputsFHEModelDev
from utils.model import MultiInputDecisionTreeClassifier
from utils.pre_processing import get_pre_processors


def get_multi_inputs(data):
    """Get inputs for all three parties from the input data, using fixed slices.

    Args:
        data (numpy.ndarray): The input data to consider.

    Returns:
        (Tuple[numpy.ndarray]): The inputs for all three parties.
    """
    return (
        data[:, INPUT_SLICES["applicant"]],
        data[:, INPUT_SLICES["bank"]],
        data[:, INPUT_SLICES["credit_bureau"]]
    )


print("Load and pre-process the data")

# Load the data
data = pandas.read_csv(DATA_PATH, encoding="utf-8")

# Drop the 'employer_name' column from the input data
data = data.drop(columns=['employer_name'])

# Define input and target data
data_x = data.copy()
data_y = data_x.pop("credit_approval").copy().to_frame()

# Get data from all parties
data_applicant = data_x[APPLICANT_COLUMNS].copy()
data_bank = data_x[BANK_COLUMNS].copy()
data_credit_bureau = data_x[CREDIT_BUREAU_COLUMNS].copy()

# Feature engineer the data
pre_processor_applicant, pre_processor_bank, pre_processor_credit_bureau = get_pre_processors()

preprocessed_data_applicant = pre_processor_applicant.fit_transform(data_applicant)
preprocessed_data_bank = pre_processor_bank.fit_transform(data_bank)
preprocessed_data_credit_bureau = pre_processor_credit_bureau.fit_transform(data_credit_bureau)

# preprocessed_data_applicant = preprocessed_data_applicant.toarray()

# print("Shape of preprocessed_data_applicant:", preprocessed_data_applicant.shape)
# print("Shape of preprocessed_data_bank:", preprocessed_data_bank.shape)
# print("Shape of preprocessed_data_credit_bureau:", preprocessed_data_credit_bureau.shape)

# print("Data type of preprocessed_data_applicant:", type(preprocessed_data_applicant))
# print("Data type of preprocessed_data_bank:", type(preprocessed_data_bank))
# print("Data type of preprocessed_data_credit_bureau:", type(preprocessed_data_credit_bureau))

# preprocessed_data_x = numpy.concatenate((preprocessed_data_applicant, preprocessed_data_bank), axis=1)
# preprocessed_data_x = numpy.concatenate((preprocessed_data_x, preprocessed_data_credit_bureau), axis=1)

preprocessed_data_x = numpy.concatenate((preprocessed_data_applicant, preprocessed_data_bank, preprocessed_data_credit_bureau), axis=1)

# print("Shape of data_applicant:", data_applicant.shape)
# print("Shape of data_bank:", data_bank.shape)
# print("Shape of data_credit_bureau:", data_credit_bureau.shape)

print("\nTrain and compile the model")

model = MultiInputDecisionTreeClassifier()

model, sklearn_model = model.fit_benchmark(preprocessed_data_x, data_y)

multi_inputs_train = get_multi_inputs(preprocessed_data_x)

# for i, input_data in enumerate(multi_inputs_train):
#     print(f"Shape of input {i}: {input_data.shape}")

model.compile(*multi_inputs_train, inputs_encryption_status=["encrypted", "encrypted", "encrypted"])

print("\nSave deployment files")

# Delete the deployment folder and its content if it already exists
if DEPLOYMENT_PATH.is_dir():
    shutil.rmtree(DEPLOYMENT_PATH)

# Save files needed for deployment (and enable cross-platform deployment)
fhe_model_dev = MultiInputsFHEModelDev(DEPLOYMENT_PATH, model)
fhe_model_dev.save(via_mlir=True)

# Save pre-processors
with (
    PRE_PROCESSOR_APPLICANT_PATH.open('wb') as file_applicant,
    PRE_PROCESSOR_BANK_PATH.open('wb') as file_bank,
    PRE_PROCESSOR_CREDIT_BUREAU_PATH.open('wb') as file_credit_bureau,
):
    pickle.dump(pre_processor_applicant, file_applicant)
    pickle.dump(pre_processor_bank, file_bank)
    pickle.dump(pre_processor_credit_bureau, file_credit_bureau)

print("\nDone !")

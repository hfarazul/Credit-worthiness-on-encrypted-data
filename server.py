"""Server that will listen for GET and POST requests from the client."""

import time
from typing import List, Optional
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from settings import DEPLOYMENT_PATH, SERVER_FILES, CLIENT_TYPES
from utils.client_server_interface import MultiInputsFHEModelServer

# Load the server
FHE_SERVER = MultiInputsFHEModelServer(DEPLOYMENT_PATH)


def _get_server_file_path(name, client_id, client_type=None):
    """Get the file path for the server.

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

    dir_path = SERVER_FILES / f"{client_id}"
    dir_path.mkdir(exist_ok=True)

    return dir_path / f"{name}{client_type_suffix}"


# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route
@app.get("/")
def root():
    return {"message": "Welcome to Credit Card Approval Prediction server!"}


@app.post("/send_file")
def send_file(
    client_id: str = Form(),
    client_type: Optional[str] = Form(None),
    file_name: str = Form(),
    files: List[UploadFile] = File(),
):
    """Send the files to the server."""
    # Retrieve the encrypted inputs or evaluation key
    encrypted_file_path = _get_server_file_path(file_name, client_id, client_type)
    
    # Write the file using the above path
    with encrypted_file_path.open("wb") as encrypted_file:
        encrypted_file.write(files[0].file.read())


@app.post("/run_fhe")
def run_fhe(
    client_id: str = Form(),
):
    """Execute the model on the encrypted inputs using FHE."""
    # Get the evaluation key
    evaluation_key_path = _get_server_file_path("evaluation_key", client_id)

    with evaluation_key_path.open("rb") as evaluation_key_file:
        evaluation_key = evaluation_key_file.read()

    # Get the encrypted inputs from all parties
    encrypted_inputs = []
    for client_type in CLIENT_TYPES:
        encrypted_inputs_path = _get_server_file_path("encrypted_inputs", client_id, client_type)

        with encrypted_inputs_path.open("rb") as encrypted_inputs_file:
            encrypted_input = encrypted_inputs_file.read()
            encrypted_inputs.append(encrypted_input)

    # Run the FHE execution
    start = time.time()
    encrypted_output = FHE_SERVER.run(*encrypted_inputs, serialized_evaluation_keys=evaluation_key)
    fhe_execution_time = round(time.time() - start, 2)

    # Retrieve the encrypted output path
    encrypted_output_path = _get_server_file_path("encrypted_output", client_id)

    # Write the file using the above path
    with encrypted_output_path.open("wb") as output_file:
        output_file.write(encrypted_output)

    return JSONResponse(content=fhe_execution_time)


@app.post("/get_output")
def get_output(
    client_id: str = Form(),
):
    """Retrieve the encrypted output."""
    # Retrieve the encrypted output path
    encrypted_output_path = _get_server_file_path("encrypted_output", client_id)

    # Read the file using the above path
    with encrypted_output_path.open("rb") as encrypted_output_file:
        encrypted_output = encrypted_output_file.read()

    return Response(encrypted_output)

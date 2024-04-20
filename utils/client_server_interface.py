"""Modified classes for use for Client-Server interface with multi-inputs circuits."""

import numpy
import copy

from typing import Tuple

from concrete.fhe import Value, EvaluationKeys
from concrete.ml.deployment.fhe_client_server import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import DecisionTreeClassifier


class MultiInputsFHEModelDev(FHEModelDev):

    def __init__(self, *arg, **kwargs):

        super().__init__(*arg, **kwargs)

        # Workaround that enables loading a modified version of a DecisionTreeClassifier model
        model = copy.copy(self.model)
        model.__class__ = DecisionTreeClassifier
        self.model = model


class MultiInputsFHEModelClient(FHEModelClient):

    def __init__(self, *args, nb_inputs=1, **kwargs):
        self.nb_inputs = nb_inputs

        super().__init__(*args, **kwargs)

    def quantize_encrypt_serialize_multi_inputs(
        self,
        x: numpy.ndarray,
        input_index: int,
        processed_input_shape: Tuple[int],
        input_slice: slice,
    ) -> bytes:
        """Quantize, encrypt and serialize inputs for a multi-party model.

        In the following, the 'quantize_input' method called is the one defined in Concrete ML's
        built-in models. Since they don't natively handle inputs for multi-party models, we need
        to use padding along indexing and slicing so that inputs from a specific party are correctly
        associated with input quantizers.

        Args:
            x (numpy.ndarray): The input to consider. Here, the input should only represent a
                single party.
            input_index (int): The index representing the type of model (0: "applicant", 1: "bank",
                2: "credit_bureau")
            processed_input_shape (Tuple[int]): The total input shape (all parties combined) after
                pre-processing.
            input_slice (slice): The slices to consider for the given party.

        """

        x_padded = numpy.zeros(processed_input_shape)

        x_padded[:, input_slice] = x

        q_x_padded = self.model.quantize_input(x_padded)

        q_x = q_x_padded[:, input_slice]

        q_x_inputs = [None for _ in range(self.nb_inputs)]
        q_x_inputs[input_index] = q_x

        # Encrypt the values
        q_x_enc = self.client.encrypt(*q_x_inputs)

        # Serialize the encrypted values to be sent to the server
        q_x_enc_ser = q_x_enc[input_index].serialize()
        return q_x_enc_ser


class MultiInputsFHEModelServer(FHEModelServer):

    def run(
        self,
        *serialized_encrypted_quantized_data: Tuple[bytes],
        serialized_evaluation_keys: bytes,
    ) -> bytes:
        """Run the model on the server over encrypted data for a multi-party model.

        Args:
            serialized_encrypted_quantized_data (Tuple[bytes]): The encrypted, quantized
                and serialized data for a multi-party model.
            serialized_evaluation_keys (bytes): The serialized evaluation key.

        Returns:
            bytes: the result of the model
        """
        assert self.server is not None, "Model has not been loaded."

        deserialized_encrypted_quantized_data = tuple(Value.deserialize(data) for data in serialized_encrypted_quantized_data)

        deserialized_evaluation_keys = EvaluationKeys.deserialize(serialized_evaluation_keys)

        result = self.server.run(
            *deserialized_encrypted_quantized_data, evaluation_keys=deserialized_evaluation_keys
        )
        serialized_result = result.serialize()
        return serialized_result

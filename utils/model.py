"""Modified model class to handles multi-inputs circuit."""

import numpy
import time
from typing import Optional, Sequence, Union

from concrete.fhe.compilation.compiler import Compiler, Configuration, DebugArtifacts, Circuit

from concrete.ml.common.check_inputs import check_array_and_assert
from concrete.ml.common.utils import (
    generate_proxy_function,
    manage_parameters_for_pbs_errors,
    check_there_is_no_p_error_options_in_configuration
)
from concrete.ml.quantization.quantized_module import QuantizedModule, _get_inputset_generator
from concrete.ml.sklearn import DecisionTreeClassifier

class MultiInputModel:

    def quantize_input(self, *X: numpy.ndarray) -> numpy.ndarray:
        self.check_model_is_fitted()
        assert sum(input.shape[1] for input in X) == len(self.input_quantizers)

        base_j = 0
        q_inputs = []
        for i, input in enumerate(X):
            q_input = numpy.zeros_like(input, dtype=numpy.int64)

            for j in range(input.shape[1]):
                quantizer_index = base_j + j
                q_input[:, j] = self.input_quantizers[quantizer_index].quant(input[:, j])

            assert q_input.dtype == numpy.int64, f"Inputs {i} were not quantized to int64 values"

            q_inputs.append(q_input)
            base_j += input.shape[1]

        return tuple(q_inputs) if len(q_inputs) > 1 else q_inputs[0]

    def compile(
        self,
        *inputs,
        configuration: Optional[Configuration] = None,
        artifacts: Optional[DebugArtifacts] = None,
        show_mlir: bool = False,
        p_error: Optional[float] = None,
        global_p_error: Optional[float] = None,
        verbose: bool = False,
        inputs_encryption_status: Optional[Sequence[str]] = None,
    ) -> Circuit:

        # Check that the model is correctly fitted
        self.check_model_is_fitted()

        # Cast pandas, list or torch to numpy
        inputs_as_array = []
        for input in inputs:
            input_as_array = check_array_and_assert(input)
            inputs_as_array.append(input_as_array)

        inputs_as_array = tuple(inputs_as_array)

        # p_error or global_p_error should not be set in both the configuration and direct arguments
        check_there_is_no_p_error_options_in_configuration(configuration)

        # Find the right way to set parameters for compiler, depending on the way we want to default
        p_error, global_p_error = manage_parameters_for_pbs_errors(p_error, global_p_error)

        # Quantize the inputs
        quantized_inputs = self.quantize_input(*inputs_as_array)

        # Generate the compilation input-set with proper dimensions
        inputset = _get_inputset_generator(quantized_inputs)

        # Reset for double compile
        self._is_compiled = False

        # Retrieve the compiler instance
        module_to_compile = self._get_module_to_compile(inputs_encryption_status)

        # Compiling using a QuantizedModule requires different steps and should not be done here
        assert isinstance(module_to_compile, Compiler), (
            "Wrong module to compile. Expected to be of type `Compiler` but got "
            f"{type(module_to_compile)}."
        )

        # Jit compiler is now deprecated and will soon be removed, it is thus forced to False
        # by default
        self.fhe_circuit_ = module_to_compile.compile(
            inputset,
            configuration=configuration,
            artifacts=artifacts,
            show_mlir=show_mlir,
            p_error=p_error,
            global_p_error=global_p_error,
            verbose=verbose,
            single_precision=False,
            fhe_simulation=False,
            fhe_execution=True,
        )

        self._is_compiled = True

        # For mypy
        assert isinstance(self.fhe_circuit, Circuit)

        return self.fhe_circuit

    def _get_module_to_compile(self, inputs_encryption_status) -> Union[Compiler, QuantizedModule]:
        assert self._tree_inference is not None, self._is_not_fitted_error_message()

        if not self._is_compiled:
            tree_inference = self._tree_inference
            self._tree_inference = lambda *args: tree_inference(numpy.concatenate(args, axis=1))

        input_names = [f"input_{i}_encrypted" for i in range(len(inputs_encryption_status))]

        # Generate the proxy function to compile
        _tree_inference_proxy, function_arg_names = generate_proxy_function(
            self._tree_inference, input_names
        )

        inputs_encryption_statuses = {input_name: status for input_name, status in zip(function_arg_names.values(), inputs_encryption_status)}

        # Create the compiler instance
        compiler = Compiler(
            _tree_inference_proxy,
            inputs_encryption_statuses,
        )

        return compiler

class MultiInputDecisionTreeClassifier(MultiInputModel, DecisionTreeClassifier):
    pass

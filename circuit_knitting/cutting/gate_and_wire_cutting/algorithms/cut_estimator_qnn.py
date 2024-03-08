# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Estimator quantum neural network class"""

from __future__ import annotations

import logging
from copy import copy
from typing import Sequence

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseEstimator, Estimator, EstimatorResult
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_algorithms.gradients import (
    BaseEstimatorGradient,
    EstimatorGradientResult,
    ParamShiftEstimatorGradient,
)

from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.exceptions import QiskitMachineLearningError

from qiskit_machine_learning.neural_networks import NeuralNetwork

logger = logging.getLogger(__name__)

# Cutting-specific packages
import re
from circuit_knitting.cutting.gate_and_wire_cutting.frontend import cut_wires_and_gates_to_subcircuits
from circuit_knitting.cutting.gate_and_wire_cutting.frontend import execute_simulation
from circuit_knitting.cutting.cutting_reconstruction import reconstruct_expectation_values

from .cut_param_shift_estimator_gradient import CutParamShiftEstimatorGradient

class CutEstimatorQNN(NeuralNetwork):
    """A neural network implementation based on the Estimator primitive.

    The ``EstimatorQNN`` is a neural network that takes in a parametrized quantum circuit
    with designated parameters for input data and/or weights, an optional observable(s) and outputs
    their expectation value(s). Quite often, a combined quantum circuit is used. Such a circuit is
    built from two circuits: a feature map, it provides input parameters for the network, and an
    ansatz (weight parameters).
    In this case a :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` can be passed as
    circuit to simplify the composition of a feature map and ansatz.
    If a :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` is passed as circuit, the
    input and weight parameters do not have to be provided, because these two properties are taken
    from the :class:`~qiskit_machine_learning.circuit.library.QNNCircuit`.

    Example:

    .. code-block::

        from qiskit import QuantumCircuit
        from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
        from qiskit_machine_learning.circuit.library import QNNCircuit

        from qiskit_machine_learning.neural_networks import EstimatorQNN

        num_qubits = 2

        # Using the QNNCircuit:
        # Create a parameterized 2 qubit circuit composed of the default ZZFeatureMap feature map
        # and RealAmplitudes ansatz.
        qnn_qc = QNNCircuit(num_qubits)

        qnn = EstimatorQNN(
            circuit=qnn_qc
        )

        qnn.forward(input_data=[1, 2], weights=[1, 2, 3, 4, 5, 6, 7, 8])

        # Explicitly specifying the ansatz and feature map:
        feature_map = ZZFeatureMap(feature_dimension=num_qubits)
        ansatz = RealAmplitudes(num_qubits=num_qubits)

        qc = QuantumCircuit(num_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters
        )

        qnn.forward(input_data=[1, 2], weights=[1, 2, 3, 4, 5, 6, 7, 8])


    The following attributes can be set via the constructor but can also be read and
    updated once the EstimatorQNN object has been constructed.

    Attributes:

        estimator (BaseEstimator): The estimator primitive used to compute the neural network's results.
        gradient (BaseEstimatorGradient): The estimator gradient to be used for the backward
            pass.
    """

    def __init__(
        self,
        *,
        circuit: QuantumCircuit,
        estimator: BaseEstimator | None = None,
        observables: Sequence[BaseOperator] | BaseOperator | None = None,
        input_params: Sequence[Parameter] | None = None,
        weight_params: Sequence[Parameter] | None = None,
        gradient: BaseEstimatorGradient | None = None,
        input_gradients: bool = False,
        shots: int | None = None,
        max_subcircuit_width=6,
        max_cuts=9,
        num_subcircuits=None,
        cut_samples=1500,
        model='gurobi'
    ):
        r"""
        Args:
            estimator: The estimator used to compute neural network's results.
                If ``None``, a default instance of the reference estimator,
                :class:`~qiskit.primitives.Estimator`, will be used.
            circuit: The quantum circuit to represent the neural network. If a
                :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` is passed, the
                `input_params` and `weight_params` do not have to be provided, because these two
                properties are taken from the
                :class:`~qiskit_machine_learning.circuit.library.QNNCircuit`.
            observables: The observables for outputs of the neural network. If ``None``,
                use the default :math:`Z^{\otimes num\_qubits}` observable.
            input_params: The parameters that correspond to the input data of the network.
                If ``None``, the input data is not bound to any parameters.
                If a :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` is provided the
                `input_params` value here is ignored. Instead the value is taken from the
                :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` input_parameters.
            weight_params: The parameters that correspond to the trainable weights.
                If ``None``, the weights are not bound to any parameters.
                If a :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` is provided the
                `weight_params` value here is ignored. Instead the value is taken from the
                :class:`~qiskit_machine_learning.circuit.library.QNNCircuit` weight_parameters.
            gradient: The estimator gradient to be used for the backward pass.
                If None, a default instance of the estimator gradient,
                :class:`~qiskit_algorithms.gradients.ParamShiftEstimatorGradient`, will be used.
            input_gradients: Determines whether to compute gradients with respect to input data.
                Note that this parameter is ``False`` by default, and must be explicitly set to
                ``True`` for a proper gradient computation when using
                :class:`~qiskit_machine_learning.connectors.TorchConnector`.

        Raises:
            QiskitMachineLearningError: Invalid parameter values.
        """
        if estimator is None:
            estimator = Estimator()
        self.estimator = estimator
        self._org_circuit = circuit
        if observables is None:
            observables = SparsePauliOp.from_list([("Z" * circuit.num_qubits, 1)])
        if isinstance(observables, BaseOperator):
            observables = (observables,)
        self._observables = observables
        if isinstance(circuit, QNNCircuit):
            self._input_params = list(circuit.input_parameters)
            self._weight_params = list(circuit.weight_parameters)
        else:
            self._input_params = list(input_params) if input_params is not None else []
            self._weight_params = list(weight_params) if weight_params is not None else []
        # Disregard the passed in gradient and use the custom gradient for cutting
        gradient = CutParamShiftEstimatorGradient(self.estimator)
        self.gradient = gradient
        self._input_gradients = input_gradients

        super().__init__(
            num_inputs=len(self._input_params),
            num_weights=len(self._weight_params),
            sparse=False,
            output_shape=len(self._observables),
            input_gradients=input_gradients,
        )

        self._circuit = self._reparameterize_circuit(circuit, input_params, weight_params)

        if num_subcircuits is None:
            num_subcircuits = [2]

        #

        # Split the circuit into subcircuits for running
        self.subcircuits, self.subobservables = cut_wires_and_gates_to_subcircuits(
            circuit=self._circuit,
            observables=[str(observable) for observable in observables[0].paulis],
            method='automatic',
            max_subcircuit_width=max_subcircuit_width,
            max_cuts=max_cuts,
            num_subcircuits=num_subcircuits,
            model=model
        )

        self.shots = shots
        self.cut_samples = cut_samples
        self.max_subcircuit_width = max_subcircuit_width
        self.max_cuts = max_cuts
        self.num_subcircuits = num_subcircuits
        self.cut_samples = cut_samples
        self.model = model


        # Create a dictionary of dictionaries, for each subcircuit, to translate the parameters to the subcircuit parameters
        self.subcircuit_parameter_translation = {}

        # Count the number of total input parameters
        num_inputs = self.circuit.num_qubits

        for subcircuit_key in self.subcircuits.keys():
            subcircuit_param_dict = {}
            # Create a dictionary of the parameters for the subcircuit
            # Check which inputs this subcircuit needs
            for circ_parameter in self.subcircuits[subcircuit_key].parameters:
                if circ_parameter.name.startswith('input'):
                    # Check which number input parameter it needs by using regex for the number between the brackets
                    input_idx = int(re.match(r'.*\[(\d+)\]', circ_parameter.name).group(1))
                    subcircuit_param_dict[circ_parameter] = input_idx
                else:
                    # Check which number weight parameter it needs by using regex for the number between the brackets
                    weight_idx = int(re.match(r'.*\[(\d+)\]', circ_parameter.name).group(1))
                    subcircuit_param_dict[circ_parameter] = weight_idx + num_inputs
            # Add the dictionary for the specific subcircuit to the larger dictionary
            self.subcircuit_parameter_translation[subcircuit_key] = subcircuit_param_dict

        gradient.set_cutting_parameters(self.subcircuits, self.subobservables, self.shots, self.cut_samples,
                                        self.subcircuit_parameter_translation)

    @property
    def circuit(self) -> QuantumCircuit:
        """The quantum circuit representing the neural network."""
        return copy(self._org_circuit)

    @property
    def observables(self) -> Sequence[BaseOperator] | BaseOperator:
        """Returns the underlying observables of this QNN."""
        return copy(self._observables)

    @property
    def input_params(self) -> Sequence[Parameter] | None:
        """The parameters that correspond to the input data of the network."""
        return copy(self._input_params)

    @property
    def weight_params(self) -> Sequence[Parameter] | None:
        """The parameters that correspond to the trainable weights."""
        return copy(self._weight_params)

    @property
    def input_gradients(self) -> bool:
        """Returns whether gradients with respect to input data are computed by this neural network
        in the ``backward`` method or not. By default such gradients are not computed."""
        return self._input_gradients

    @input_gradients.setter
    def input_gradients(self, input_gradients: bool) -> None:
        """Turn on/off computation of gradients with respect to input data."""
        self._input_gradients = input_gradients

    def _cut_and_execute_with_params(self, parameters_list):
        # Create a dictionary with the same keys as self.subcircuits
        subcircuits = dict.fromkeys(self.subcircuits.keys())

        # Create a list of the parameters for the subcircuits
        subcircuits_parameters = dict.fromkeys(self.subcircuits.keys())
        for subcircuit_key in subcircuits.keys():
            subcircuit_parameters_vals = []
            subcircuit_parameters_symbols = self.subcircuits[subcircuit_key].parameters
            for circ_parameter in subcircuit_parameters_symbols:
                subcircuit_parameters_vals.append(parameters_list[self.subcircuit_parameter_translation[subcircuit_key][circ_parameter]])
            subcircuits_parameters[subcircuit_key] = subcircuit_parameters_vals


        # Then, assign the parameters to the subcircuits
        for subcircuit_key in subcircuits.keys():
            subcircuits[subcircuit_key] = self.subcircuits[subcircuit_key].assign_parameters(
                subcircuits_parameters[subcircuit_key], inplace=False)

        # Execute the circuits
        quasi_dists, coefficients = execute_simulation(subcircuits, self.subobservables, shots=self.shots,
                                                       samples=self.cut_samples)

        # Reconstruct the expectation values
        simulated_expvals = reconstruct_expectation_values(quasi_dists, coefficients, self.subobservables)

        return simulated_expvals

    def _forward_postprocess(self, num_samples: int, result: EstimatorResult) -> np.ndarray:
        """Post-processing during forward pass of the network."""
        return np.reshape(result.values, (-1, num_samples)).T

    def _forward(
        self, input_data: np.ndarray | None, weights: np.ndarray | None
    ) -> np.ndarray | None:
        """Forward pass of the neural network."""
        parameter_values_, num_samples = self._preprocess_forward(input_data, weights)

            # job = self.estimator.run(
            #     [self._circuit] * num_samples * self.output_shape[0],
            #     [op for op in self._observables for _ in range(num_samples)],
            #     np.tile(parameter_values_, (self.output_shape[0], 1)),
            # )
        # try:
            # results = job.result()
        # FIXME: the estimator ran many circuits for all the different results. We need to do the same, not one subcircuit.
        results = []
        for parameters in parameter_values_:
            results.append(self._cut_and_execute_with_params(parameters))
            # results = self._cut_and_execute_with_params(parameter_values_)
        # except Exception as exc:
        #     raise QiskitMachineLearningError("Cut and estimate job failed.") from exc

        return self._forward_postprocess(num_samples, results)

    def _backward_postprocess(
        self, num_samples: int, result: EstimatorGradientResult
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Post-processing during backward pass of the network."""
        num_observables = self.output_shape[0]
        if self._input_gradients:
            input_grad = np.zeros((num_samples, num_observables, self._num_inputs))
        else:
            input_grad = None

        weights_grad = np.zeros((num_samples, num_observables, self._num_weights))
        gradients = np.asarray(result.gradients)
        for i in range(num_observables):
            if self._input_gradients:
                input_grad[:, i, :] = gradients[i * num_samples : (i + 1) * num_samples][
                    :, : self._num_inputs
                ]
                weights_grad[:, i, :] = gradients[i * num_samples : (i + 1) * num_samples][
                    :, self._num_inputs :
                ]
            else:
                weights_grad[:, i, :] = gradients[i * num_samples : (i + 1) * num_samples]
        return input_grad, weights_grad

    def _backward(
        self, input_data: np.ndarray | None, weights: np.ndarray | None
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Backward pass of the network."""
        # prepare parameters in the required format
        parameter_values, num_samples = self._preprocess_forward(input_data, weights)

        input_grad, weights_grad = None, None

        if np.prod(parameter_values.shape) > 0:
            num_observables = self.output_shape[0]
            num_circuits = num_samples * num_observables

            circuits = [self._circuit] * num_circuits
            observables = [op for op in self._observables for _ in range(num_samples)]
            param_values = np.tile(parameter_values, (num_observables, 1))

            job = None
            if self._input_gradients:
                job = self.gradient.run(circuits, observables, param_values)
            elif len(parameter_values[0]) > self._num_inputs:
                params = [self._circuit.parameters[self._num_inputs :]] * num_circuits
                job = self.gradient.run(circuits, observables, param_values, parameters=params)

            if job is not None:
                try:
                    results = job.result()
                except Exception as exc:
                    raise QiskitMachineLearningError("Estimator job failed.") from exc

                input_grad, weights_grad = self._backward_postprocess(num_samples, results)

        return input_grad, weights_grad

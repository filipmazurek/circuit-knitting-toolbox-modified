# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Gradient of probabilities with parameter shift
"""

from __future__ import annotations

from collections.abc import Sequence

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit_algorithms.gradients.base.base_estimator_gradient import BaseEstimatorGradient
from qiskit_algorithms.gradients.base.estimator_gradient_result import EstimatorGradientResult
from qiskit_algorithms.gradients.utils import _make_param_shift_parameter_values

from qiskit_algorithms.exceptions import AlgorithmError

# Cutting-specific packages
from circuit_knitting.cutting.gate_and_wire_cutting.frontend import execute_simulation
from circuit_knitting.cutting.cutting_reconstruction import reconstruct_expectation_values


class CutParamShiftEstimatorGradient(BaseEstimatorGradient):
    """
    Compute the gradients of the expectation values by the parameter shift rule [1].

    **Reference:**
    [1] Schuld, M., Bergholm, V., Gogolin, C., Izaac, J., and Killoran, N. Evaluating analytic
    gradients on quantum hardware, `DOI <https://doi.org/10.1103/PhysRevA.99.032331>`_
    """

    SUPPORTED_GATES = [
        "x",
        "y",
        "z",
        "h",
        "rx",
        "ry",
        "rz",
        "p",
        "cx",
        "cy",
        "cz",
        "ryy",
        "rxx",
        "rzz",
        "rzx",
    ]

    def set_cutting_parameters(self, subcircuits, subobservables, shots, num_samples, subcircuit_parameter_translation):
        self.subcircuits = subcircuits
        self.subobservables = subobservables
        self.shots = shots
        self.num_samples = num_samples
        self.subcircuit_parameter_translation = subcircuit_parameter_translation

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
                                                       samples=self.num_samples)

        # Reconstruct the expectation values
        simulated_expvals = reconstruct_expectation_values(quasi_dists, coefficients, self.subobservables)

        return simulated_expvals

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the gradients of the expectation values by the parameter shift rule."""
        g_circuits, g_parameter_values, g_parameters = self._preprocess(
            circuits, parameter_values, parameters, self.SUPPORTED_GATES
        )
        results = self._run_unique(
            g_circuits, observables, g_parameter_values, g_parameters, **options
        )
        return self._postprocess(results, circuits, parameter_values, parameters)

    def _run_unique(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        job_circuits, job_observables, job_param_values, metadata = [], [], [], []
        all_n = []
        for circuit, observable, parameter_values_, parameters_ in zip(
            circuits, observables, parameter_values, parameters
        ):
            metadata.append({"parameters": parameters_})
            # Make parameter values for the parameter shift rule.
            param_shift_parameter_values = _make_param_shift_parameter_values(
                circuit, parameter_values_, parameters_
            )
            # Combine inputs into a single job to reduce overhead.
            n = len(param_shift_parameter_values)
            job_circuits.extend([circuit] * n)
            job_observables.extend([observable] * n)
            job_param_values.extend(param_shift_parameter_values)
            all_n.append(n)

        # Can't batch in our case, so we run each circuit seperately
        results = []
        for job_param_values_in_circ in job_param_values:
            simulated_expvals = self._cut_and_execute_with_params(job_param_values_in_circ)
            results.append(simulated_expvals)

        # Compute the gradients.
        gradients = []
        partial_sum_n = 0
        for n in all_n:
            result = results[partial_sum_n: partial_sum_n + n]
            gradient_ = (result[: n // 2] - result[n // 2 :]) / 2
            gradients.append(gradient_)
            partial_sum_n += n

        opt = self._get_local_options(options)
        return EstimatorGradientResult(gradients=gradients, metadata=metadata, options=opt)

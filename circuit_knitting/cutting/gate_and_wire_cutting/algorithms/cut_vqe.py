# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The variational quantum eigensolver algorithm.
   Modified to immediately work with the gate and wire circuit cutter.
"""

from __future__ import annotations

import logging
from time import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from qiskit.algorithms.gradients import BaseEstimatorGradient
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_aer.primitives import Estimator, Sampler

from qiskit.quantum_info import PauliList

from qiskit.algorithms import eval_observables, AlgorithmError
from qiskit.algorithms.optimizers import SLSQP, Minimizer, Optimizer, OptimizerResult
from qiskit.algorithms.variational_algorithm import VariationalAlgorithm, VariationalResult
from qiskit.algorithms.minimum_eigen_solvers import MinimumEigensolver, MinimumEigensolverResult
from qiskit.algorithms.utils import validate_initial_point, validate_bounds

# private function as we expect this to be updated in the next released
from qiskit.algorithms.utils.set_batching import _set_default_batchsize

from typing import TypeVar, List, Union, Optional, Dict

_T = TypeVar("_T")  # Pylint does not allow single character class names.
ListOrDict = Union[List[Optional[_T]], Dict[str, _T]]

logger = logging.getLogger(__name__)

# Cutting-specific packages
from circuit_knitting.cutting.gate_and_wire_cutting.frontend import cut_wires_and_gates_to_subcircuits
from circuit_knitting.cutting.gate_and_wire_cutting.frontend import execute_simulation
from circuit_knitting.cutting.cutting_reconstruction import reconstruct_expectation_values
from circuit_knitting.cutting.gate_and_wire_cutting.evaluation import azure_queue_experiments
from circuit_knitting.cutting.gate_and_wire_cutting.evaluation import get_experiment_results_from_jobs
from azure.quantum.qiskit.backends.backend import AzureBackendBase


class CutVQE(VariationalAlgorithm, MinimumEigensolver):
    r"""The variational quantum eigensolver (VQE) algorithm.

    VQE is a hybrid quantum-classical algorithm that uses a variational technique to find the
    minimum eigenvalue of a given Hamiltonian operator :math:`H`.

    The ``VQE`` algorithm is executed using an :attr:`estimator` primitive, which computes
    expectation values of operators (observables).

    An instance of ``VQE`` also requires an :attr:`ansatz`, a parameterized
    :class:`.QuantumCircuit`, to prepare the trial state :math:`|\psi(\vec\theta)\rangle`. It also
    needs a classical :attr:`optimizer` which varies the circuit parameters :math:`\vec\theta` such
    that the expectation value of the operator on the corresponding state approaches a minimum,

    .. math::

        \min_{\vec\theta} \langle\psi(\vec\theta)|H|\psi(\vec\theta)\rangle.

    The :attr:`estimator` is used to compute this expectation value for every optimization step.

    The optimizer can either be one of Qiskit's optimizers, such as
    :class:`~qiskit.algorithms.optimizers.SPSA` or a callable with the following signature:

    .. code-block:: python

        from qiskit.algorithms.optimizers import OptimizerResult

        def my_minimizer(fun, x0, jac=None, bounds=None) -> OptimizerResult:
            # Note that the callable *must* have these argument names!
            # Args:
            #     fun (callable): the function to minimize
            #     x0 (np.ndarray): the initial point for the optimization
            #     jac (callable, optional): the gradient of the objective function
            #     bounds (list, optional): a list of tuples specifying the parameter bounds

            result = OptimizerResult()
            result.x = # optimal parameters
            result.fun = # optimal function value
            return result

    The above signature also allows one to use any SciPy minimizer, for instance as

    .. code-block:: python

        from functools import partial
        from scipy.optimize import minimize

        optimizer = partial(minimize, method="L-BFGS-B")

    The following attributes can be set via the initializer but can also be read and updated once
    the VQE object has been constructed.

    Attributes:
        estimator (BaseEstimator): The estimator primitive to compute the expectation value of the
            Hamiltonian operator.
        ansatz (QuantumCircuit): A parameterized quantum circuit to prepare the trial state.
        optimizer (Optimizer | Minimizer): A classical optimizer to find the minimum energy. This
            can either be a Qiskit :class:`.Optimizer` or a callable implementing the
            :class:`.Minimizer` protocol.
        gradient (BaseEstimatorGradient | None): An optional estimator gradient to be used with the
            optimizer.
        callback (Callable[[int, np.ndarray, float, dict[str, Any]], None] | None): A callback that
            can access the intermediate data at each optimization step. These data are: the
            evaluation count, the optimizer parameters for the ansatz, the `evaluate`d mean, and the
            metadata dictionary.

    References:
        [1]: Peruzzo, A., et al, "A variational eigenvalue solver on a quantum processor"
            `arXiv:1304.3061 <https://arxiv.org/abs/1304.3061>`__
    """

    def __init__(
            self,
            estimator: BaseEstimator,
            ansatz: QuantumCircuit,
            optimizer: Optimizer | Minimizer,
            *,
            gradient: BaseEstimatorGradient | None = None,
            initial_point: Sequence[float] | None = None,
            callback: Callable[[int, np.ndarray, float, dict[str, Any]], None] | None = None,
            observables: PauliList | None = None,
            shots: int | None = None,
            max_subcircuit_width=2,
            max_cuts=9,
            num_subcircuits=None,
            num_samples=1500,
            model='gurobi',
            backend=None,
            azure_backend: AzureBackendBase = None
    ) -> None:
        r"""
        Args:
            estimator: The estimator primitive to compute the expectation value of the
                Hamiltonian operator.
            ansatz: A parameterized quantum circuit to prepare the trial state.
            optimizer: A classical optimizer to find the minimum energy. This can either be a
                Qiskit :class:`.Optimizer` or a callable implementing the :class:`.Minimizer`
                protocol.
            gradient: An optional estimator gradient to be used with the optimizer.
            initial_point: An optional initial point (i.e. initial parameter values) for the
                optimizer. The length of the initial point must match the number of :attr:`ansatz`
                parameters. If ``None``, a random point will be generated within certain parameter
                bounds. ``VQE`` will look to the ansatz for these bounds. If the ansatz does not
                specify bounds, bounds of :math:`-2\pi`, :math:`2\pi` will be used.
            callback: A callback that can access the intermediate data at each optimization step.
                These data are: the evaluation count, the optimizer parameters for the ansatz, the
                estimated value, and the metadata dictionary.
        """
        super().__init__()

        if num_subcircuits is None:
            num_subcircuits = [2]

        self.estimator = estimator
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.gradient = gradient
        # this has to go via getters and setters due to the VariationalAlgorithm interface
        self.initial_point = initial_point
        self.callback = callback
        self.observables = observables
        self.shots = shots
        self.num_samples = num_samples

        if backend is None:
            backend = 'simulation'
        elif backend == 'azure':
            if not isinstance(azure_backend, AzureBackendBase):
                raise ValueError("The azure backend must be provided when the backend is 'azure' and it must be of type"
                                 "'AzureBackendBase'")
        else:
            raise ValueError("The backend must be either 'azure' or 'simulation")

        self.backend = backend
        self.azure_backend = azure_backend

        self.subcircuits, self.subobservables, n_wire_cuts, n_gate_cuts = cut_wires_and_gates_to_subcircuits(
                                                circuit=ansatz,
                                                observables=[str(observable) for observable in observables],
                                                method='automatic',
                                                max_subcircuit_width=max_subcircuit_width,
                                                max_cuts=max_cuts,
                                                num_subcircuits=num_subcircuits,
                                                model=model
                                            )

    @property
    def initial_point(self) -> Sequence[float] | None:
        return self._initial_point

    @initial_point.setter
    def initial_point(self, value: Sequence[float] | None) -> None:
        self._initial_point = value

    def compute_minimum_eigenvalue(
            self,
            operator: BaseOperator | PauliSumOp,
            aux_operators: ListOrDict[BaseOperator | PauliSumOp] | None = None,
    ) -> CutVQEResult:
        # self._check_operator_ansatz(operator)

        initial_point = validate_initial_point(self.initial_point, self.ansatz)

        bounds = validate_bounds(self.ansatz)

        start_time = time()

        evaluate_energy = self._get_evaluate_energy(self.ansatz, operator)

        if self.gradient is not None:
            evaluate_gradient = self._get_evaluate_gradient(self.ansatz, operator)
        else:
            evaluate_gradient = None

        # perform optimization
        if callable(self.optimizer):
            optimizer_result = self.optimizer(
                fun=evaluate_energy, x0=initial_point, jac=evaluate_gradient, bounds=bounds
            )
        else:
            # we always want to submit as many estimations per job as possible for minimal
            # overhead on the hardware
            was_updated = _set_default_batchsize(self.optimizer)

            optimizer_result = self.optimizer.minimize(
                fun=evaluate_energy, x0=initial_point, jac=evaluate_gradient, bounds=bounds
            )

            # reset to original value
            if was_updated:
                self.optimizer.set_max_evals_grouped(None)

        optimizer_time = time() - start_time

        logger.info(
            "Optimization complete in %s seconds.\nFound optimal point %s",
            optimizer_time,
            optimizer_result.x,
        )

        if aux_operators is not None:
            raise NotImplementedError("aux_operators not yet supported")
            # aux_operators_evaluated = estimate_observables(
            #     self.estimator, self.ansatz, aux_operators, optimizer_result.x
            # )
        else:
            aux_operators_evaluated = None

        return self._build_vqe_result(
            self.ansatz, optimizer_result, aux_operators_evaluated, optimizer_time
        )

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def _get_evaluate_energy(
            self,
            ansatz: QuantumCircuit,
            operator: BaseOperator | PauliSumOp,
    ) -> Callable[[np.ndarray], np.ndarray | float]:
        """Returns a function handle to evaluate the energy at given parameters for the ansatz.
        This is the objective function to be passed to the optimizer that is used for evaluation.

        Args:
            ansatz: The ansatz preparing the quantum state.
            operator: The operator whose energy to evaluate.

        Returns:
            A callable that computes and returns the energy of the hamiltonian of each parameter.

        Raises:
            AlgorithmError: If the primitive job to evaluate the energy fails.
        """
        num_parameters = ansatz.num_parameters

        # avoid creating an instance variable to remain stateless regarding results
        eval_count = 0

        # def evaluate_energy(parameters: np.ndarray) -> np.ndarray | float:
        #     nonlocal eval_count
        #
        #     # handle broadcasting: ensure parameters is of shape [array, array, ...]
        #     parameters = np.reshape(parameters, (-1, num_parameters)).tolist()
        #     batch_size = len(parameters)
        #
        #     # print(operator)
        #     try:
        #         job = self.estimator.run(batch_size * [ansatz], batch_size * [operator], parameters)
        #         estimator_result = job.result()
        #     except Exception as exc:
        #         raise AlgorithmError("The primitive job to evaluate the energy failed!") from exc
        #
        #     values = estimator_result.values
        #
        #     # print(values)
        #     if self.callback is not None:
        #        metadata = estimator_result.metadata
        #        for params, value, meta in zip(parameters, values, metadata):
        #            eval_count += 1
        #            self.callback(eval_count, params, value, meta)
        #
        #     energy = values[0] if len(values) == 1 else values
        #
        #     return energy

        def evaluate_energy_with_cutting(parameters: np.ndarray) -> np.ndarray | float:
            nonlocal eval_count

            # Transform the numpy array into a list
            parameters_list = list(parameters)

            try:
                # Copy the subcircuits to avoid modifying the original
                # Create a dictionary with the same keys as self.subcircuits
                subcircuits = dict.fromkeys(self.subcircuits.keys())

                # Assign the parameters to the subcircuits
                # First, split the parameters based on how many parameters each subcircuit has
                subcircuit_parameters = []
                for subcircuit_key in subcircuits.keys():
                    subcircuit_parameters.append(parameters_list[0:len(self.subcircuits[subcircuit_key].parameters)])
                    parameters_list = parameters_list[len(self.subcircuits[subcircuit_key].parameters):]

                # Then, assign the parameters to the subcircuits
                for i, subcircuit_key in enumerate(subcircuits.keys()):
                    subcircuits[subcircuit_key] = self.subcircuits[subcircuit_key].assign_parameters(subcircuit_parameters[i], inplace=False)

                # Execute the circuits
                if self.backend == 'simulation':
                    quasi_dists, coefficients = execute_simulation(subcircuits, self.subobservables, shots=self.shots, samples=self.num_samples)
                    # Reconstruct the expectation values
                    simulated_expvals = reconstruct_expectation_values(quasi_dists, coefficients, self.subobservables)
                elif self.backend == 'azure':
                    job_list, qpd_list, coefficients, subexperiments = azure_queue_experiments(
                        circuits=subcircuits,
                        subobservables=self.subobservables,
                        num_samples=8,  # 8 unique samples to get some statistics
                        backend=self.azure_backend,
                        shots=128  # Balance of cost and accuracy
                    )
                    experiment_results = get_experiment_results_from_jobs(job_list, qpd_list, coefficients)
                    simulated_expvals = reconstruct_expectation_values(*experiment_results, self.subobservables)

            except Exception as exc:
                raise AlgorithmError("The primitive job to evaluate the energy failed!") from exc

            energy = 0

            for i in range(0, len(operator)):
                energy += operator[i].coeffs[0].real * simulated_expvals[i]

            if self.callback is not None:
                eval_count += 1
                self.callback(eval_count, parameters, energy, {'': None})

            return energy

        return evaluate_energy_with_cutting

    def _get_evaluate_gradient(
            self,
            ansatz: QuantumCircuit,
            operator: BaseOperator | PauliSumOp,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Get a function handle to evaluate the gradient at given parameters for the ansatz.

        Args:
            ansatz: The ansatz preparing the quantum state.
            operator: The operator whose energy to evaluate.

        Returns:
            A function handle to evaluate the gradient at given parameters for the ansatz.

        Raises:
            AlgorithmError: If the primitive job to evaluate the gradient fails.
        """

        def evaluate_gradient(parameters: np.ndarray) -> np.ndarray:
            # broadcasting not required for the estimator gradients
            try:
                job = self.gradient.run([ansatz], [operator], [parameters])
                gradients = job.result().gradients
            except Exception as exc:
                raise AlgorithmError("The primitive job to evaluate the gradient failed!") from exc

            return gradients[0]

        return evaluate_gradient

    def _check_operator_ansatz(self, operator: BaseOperator | PauliSumOp):
        """Check that the number of qubits of operator and ansatz match and that the ansatz is
        parameterized.
        """
        if operator.num_qubits != self.ansatz.num_qubits:
            try:
                logger.info(
                    "Trying to resize ansatz to match operator on %s qubits.", operator.num_qubits
                )
                self.ansatz.num_qubits = operator.num_qubits
            except AttributeError as error:
                raise AlgorithmError(
                    "The number of qubits of the ansatz does not match the "
                    "operator, and the ansatz does not allow setting the "
                    "number of qubits using `num_qubits`."
                ) from error

        if self.ansatz.num_parameters == 0:
            raise AlgorithmError("The ansatz must be parameterized, but has no free parameters.")

    def _build_vqe_result(
            self,
            ansatz: QuantumCircuit,
            optimizer_result: OptimizerResult,
            aux_operators_evaluated: ListOrDict[tuple[complex, tuple[complex, int]]],
            optimizer_time: float,
    ) -> CutVQEResult:
        result = CutVQEResult()
        result.optimal_circuit = ansatz.copy()
        result.eigenvalue = optimizer_result.fun
        result.cost_function_evals = optimizer_result.nfev
        result.optimal_point = optimizer_result.x
        result.optimal_parameters = dict(zip(self.ansatz.parameters, optimizer_result.x))
        result.optimal_value = optimizer_result.fun
        result.optimizer_time = optimizer_time
        result.aux_operators_evaluated = aux_operators_evaluated
        result.optimizer_result = optimizer_result
        return result


class CutVQEResult(VariationalResult, MinimumEigensolverResult):
    """Variational quantum eigensolver result."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_function_evals: int | None = None

    @property
    def cost_function_evals(self) -> int | None:
        """The number of cost optimizer evaluations."""
        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        self._cost_function_evals = value
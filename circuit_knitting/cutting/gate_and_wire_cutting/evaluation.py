# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions for evaluating circuit cutting experiments."""

from __future__ import annotations

from typing import Any, NamedTuple
from collections.abc import Sequence
from itertools import chain

import numpy as np
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import PauliList
from qiskit.primitives import BaseSampler, Sampler as TerraSampler
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit.result import QuasiDistribution
from azure.quantum.qiskit.job import AzureQuantumJob
from azure.quantum.qiskit.backends.backend import AzureBackendBase

from ...utils.observable_grouping import CommutingObservableGroup, ObservableCollection
from ...utils.iteration import strict_zip
from ..qpd import (
    QPDBasis,
    SingleQubitQPDGate,
    TwoQubitQPDGate,
    generate_qpd_samples,
    decompose_qpd_instructions,
    WeightType,
)
from ..cutting_decomposition import decompose_observables


class CuttingExperimentResults(NamedTuple):
    """Circuit cutting subexperiment results and sampling coefficients."""

    quasi_dists: list[list[list[tuple[QuasiDistribution, int]]]]
    coeffs: Sequence[tuple[float, WeightType]]


def azure_queue_experiments(
    circuits: QuantumCircuit | dict[str | int, QuantumCircuit],
    subobservables: PauliList | dict[str | int, PauliList],
    num_samples: int,
    backend: AzureBackendBase,
    shots: int
) -> [list[list[list[AzureQuantumJob]]], list[list[list[int]]], list[tuple[Any, WeightType]], list[list[list[QuantumCircuit]]]]:
    """
    Run the cutting experiments on the Azure backend.
    Allow for testing syntax and cost using the syntax checker and cost estimator.
    Only usable for Quantinuum backends at the moment."""
    # Returns a list of jobs for each partition (contained in a list)

    # Job list is in the same order as results from the sampler:
    # A 3D list of length-2 tuples holding the quasi-distributions and QPD bit information
    #  for each sub-experiment.
    #  The shape of the list is: (``num_unique_samples``, ``num_partitions``, ``num_commuting_observ_groups``)

    # FIXME: find sensible numbers for num_samples and shots
    usable_backends = ['quantinuum.sim.h1-1sc', 'quantinuum.sim.h1-2sc', 'quantinuum.sim.h1-1e', 'quantinuum.sim.h1-2e', 'quantinuum.qpu.h1-1', 'quantinuum.qpu.h1-2']

    if num_samples <= 0:
        raise ValueError("The number of requested samples must be positive.")

    if isinstance(circuits, dict) and not isinstance(subobservables, dict):
        raise ValueError(
            "If a partition mapping (dict[label, subcircuit]) is passed as the "
            "circuits argument, a partition mapping (dict[label, subobservables]) "
            "is expected as the subobservables argument."
        )

    if isinstance(circuits, QuantumCircuit) and isinstance(subobservables, dict):
        raise ValueError(
            "If a QuantumCircuit is passed as the circuits argument, a PauliList "
            "is expected as the subobservables argument."
        )

    if isinstance(circuits, dict):
        if circuits.keys() != subobservables.keys():
            raise ValueError( "The keys for the circuits and observables dicts should be equivalent.")

    # Get the actual Quantinuum backend from Azure
    quantinuum_api_backend = backend

    # Generate the sub-experiments to run on backend
    # Gives back subexperiments as many as there are num_samples
    (
        subexperiments,
        coefficients,
        sampled_frequencies,
    ) = _generate_cutting_experiments(
        circuits,
        subobservables,
        num_samples,
    )

    # Run each partition's sub-experiments
    num_partitions = len(subexperiments[0])
    job_and_qpd_list_by_partition = [
        _run_experiment_on_azure(
            [sample[i] for sample in subexperiments],
            quantinuum_api_backend,
            shots
        )
        for i in range(num_partitions)
    ]

    # Reformat the jobs and qpd_bits to match the shape of the input before returning
    num_unique_samples = len(subexperiments)
    job_and_qpd_list: list[list[list[tuple[AzureQuantumJob, int]]]] = [
        [] for _ in range(num_unique_samples)
    ]

    for i in range(num_unique_samples):
        for partition in job_and_qpd_list_by_partition:
            job_and_qpd_list[i].append(partition[i])

    # Seperate the jobs and qpd bits, maintaing the same order
    job_list: list[list[list[AzureQuantumJob]]] = []
    qpd_bits_list: list[list[list[int]]] = []

    for i, partition in enumerate(job_and_qpd_list):
        job_list.append([])
        qpd_bits_list.append([])
        for j, sample in enumerate(partition):
            job_list[i].append([])
            qpd_bits_list[i].append([])
            for k, commuting_coeff_job in enumerate(sample):
                job_list[i][j].append(commuting_coeff_job[0])
                qpd_bits_list[i][j].append(commuting_coeff_job[1])

    # Need to also return information about the generated experiments
    return job_list, qpd_bits_list, coefficients, subexperiments


def _run_experiment_on_azure(
        subexperiments,
        backend,
        shots
) -> (list[list[list[AzureQuantumJob]]], list[list[list[int]]]):
    """
    Submit the circuits to the azure backend
    """

    # TODO: why am I still getting included classical registers with 0 qubits in them?

    num_qpd_bits_flat = []

    # Run all the experiments in one big batch
    experiments_flat = list(chain.from_iterable(subexperiments))

    for circ in experiments_flat:
        if (
                len(circ.cregs) > 2
                or sum([reg.size for reg in circ.cregs]) != circ.num_clbits
        ):
            # If the classical bits/registers are in any other format than expected, the user must have
            # input them, so we can just raise this generic error in any case.
            raise ValueError(
                "Circuits input to execute_experiments should contain no classical registers or bits."
            )
        # If the qpd_measurements register was not added (because no qpd measurements in this subexperiment)

        if circ.cregs[0].name != "qpd_measurements":
            num_qpd_bits_flat.append(0)
            if circ.cregs[0].name != "observable_measurements":
                raise ValueError(
                    "Circuits input to execute_experiments should contain no classical registers or bits."
                )
        else:
            num_qpd_bits_flat.append(len(circ.cregs[0]))

    # Run all batched experiments
    jobs_flat = []
    for circ in experiments_flat:
        # Queues the job in Azure
        job = backend.run(circ, shots=shots)
        jobs_flat.append(job)

    # Reshape the output jobs to match the input
    if len(subexperiments) == 1:
        job_list_reshaped = np.array([jobs_flat])
        num_qpd_bits = np.array([num_qpd_bits_flat])
    else:
        # We manually build the shape tuple in second arg because it behaves strangely
        # with QuantumCircuits in some versions. (e.g. passes local pytest but fails in tox env)
        job_list_reshaped = np.reshape(
            jobs_flat, (len(subexperiments), len(subexperiments[0]))
        )
        num_qpd_bits = np.reshape(
            num_qpd_bits_flat, (len(subexperiments), len(subexperiments[0]))
        )

    # Create the counts tuples, which include the number of QPD measurement bits
    job_and_qpd_list: list[list[list[tuple[AzureQuantumJob, int]]]] = [
        [] for _ in range(len(subexperiments))
    ]

    for i, sample in enumerate(job_list_reshaped):
        for j, prob_dict in enumerate(sample):
            job_and_qpd_list[i].append((prob_dict, num_qpd_bits[i][j]))

    return job_and_qpd_list


def get_raw_results_from_jobs(jobs: list[list[list[AzureQuantumJob]]]) -> list[list[list[dict]]]:
    """
    Combines all the results from the AzureQuantumJobs into a single result. Good for checking the syntax checker result
    """
    # This function is useful for checking the syntax checker results
    results: list[list[list[tuple[dict[str, int], int]]]] = []
    for i, partition_jobs in enumerate(jobs):
        results.append([])
        for j, sample_jobs in enumerate(partition_jobs):
            results[i].append([])
            for k, observable_job in enumerate(sample_jobs):
                result = observable_job.result()
                results[i][j].append(result)

    return results


def get_experiment_results_from_jobs(jobs: list[list[list[AzureQuantumJob]]], qpd_bits: list[list[list[int]]], coefficients) -> CuttingExperimentResults:
    # Reformat the counts to match the shape of the input before returning
    num_unique_samples = len(jobs[0])
    quasi_dists: list[list[list[tuple[dict[str, int], int]]]] = []

    # Get all the results from the jobs, preserving order
    result_list: list[list[list[Any]]] = get_raw_results_from_jobs(jobs)

    for i, partition in enumerate(result_list):
        quasi_dists.append([])
        for j, sample in enumerate(partition):
            quasi_dists[i].append([])
            for k, observable_job in enumerate(sample):
                prob_dict = observable_job.results[0].data.probabilities
                new_prob_dict = {}
                for key in prob_dict.keys():
                    new_key = int(key, 2)
                    new_prob_dict[new_key] = prob_dict[key]
                quasi_dists[i][j].append((new_prob_dict, qpd_bits[i][j][k]))

    return CuttingExperimentResults(quasi_dists, coefficients)


def execute_experiments(
    circuits: QuantumCircuit | dict[str | int, QuantumCircuit],
    subobservables: PauliList | dict[str | int, PauliList],
    num_samples: int,
    samplers: BaseSampler | dict[str | int, BaseSampler],
) -> CuttingExperimentResults:
    r"""
    Generate the sampled circuits, append the observables, and run the sub-experiments.

    Args:
        circuits: The circuit(s) resulting from decomposing nonlocal gates
        subobservables: The subobservable(s) corresponding to the circuit(s). If
            a :class:`~qiskit.QuantumCircuit` is submitted for the ``circuits`` argument,
            a :class:`~qiskit.quantum_info.PauliList` is expected; otherwise, a mapping
            from partition label to subobservables is expected.
        num_samples: The number of samples to draw from the quasiprobability distribution
        samplers: Sampler(s) on which to run the sub-experiments.

    Returns:
        - A 3D list of length-2 tuples holding the quasi-distributions and QPD bit information
          for each sub-experiment. The shape of the list is: (``num_unique_samples``, ``num_partitions``, ``num_commuting_observ_groups``)
        - Coefficients corresponding to each unique subexperiment's
          sampling frequency

    Raises:
        ValueError: The number of requested samples must be positive.
        ValueError: The types of ``circuits`` and ``subobservables`` arguments are incompatible.
        ValueError: ``SingleQubitQPDGate``\ s are not supported in unseparable circuits.
        ValueError: The keys for the input dictionaries are not equivalent.
        ValueError: The input circuits may not contain any classical registers or bits.
        ValueError: If multiple samplers are passed, each one must be unique.
    """
    if num_samples <= 0:
        raise ValueError("The number of requested samples must be positive.")

    if isinstance(circuits, dict) and not isinstance(subobservables, dict):
        raise ValueError(
            "If a partition mapping (dict[label, subcircuit]) is passed as the "
            "circuits argument, a partition mapping (dict[label, subobservables]) "
            "is expected as the subobservables argument."
        )

    if isinstance(circuits, QuantumCircuit) and isinstance(subobservables, dict):
        raise ValueError(
            "If a QuantumCircuit is passed as the circuits argument, a PauliList "
            "is expected as the subobservables argument."
        )

    if isinstance(circuits, dict):
        if circuits.keys() != subobservables.keys():
            raise ValueError(
                "The keys for the circuits and observables dicts should be equivalent."
            )
        if isinstance(samplers, dict) and circuits.keys() != samplers.keys():
            raise ValueError(
                "The keys for the circuits and samplers dicts should be equivalent."
            )

    if isinstance(samplers, dict):
        # Ensure that each sampler is unique
        collision_dict: dict[int, str | int] = {}
        for k, v in samplers.items():
            if id(v) in collision_dict:
                raise ValueError(
                    "Currently, if a samplers dict is passed to execute_experiments(), "
                    "then each sampler must be unique; however, subsystems "
                    f"{collision_dict[id(v)]} and {k} were passed the same sampler."
                )
            collision_dict[id(v)] = k

    # Ensure input Samplers can handle mid-circuit measurements
    _validate_samplers(samplers)

    # Generate the sub-experiments to run on backend
    (
        subexperiments,
        coefficients,
        sampled_frequencies,
    ) = _generate_cutting_experiments(
        circuits,
        subobservables,
        num_samples,
    )

    # Create a list of samplers -- one for each qubit partition
    num_partitions = len(subexperiments[0])
    if isinstance(samplers, BaseSampler):
        samplers_by_partition = [samplers] * num_partitions
    else:
        samplers_by_partition = [samplers[key] for key in sorted(samplers.keys())]

    # Run each partition's sub-experiments
    quasi_dists_by_partition = [
        _run_experiments_batch(
            [sample[i] for sample in subexperiments],
            samplers_by_partition[i],
        )
        for i in range(num_partitions)
    ]

    # Reformat the counts to match the shape of the input before returning
    num_unique_samples = len(subexperiments)
    quasi_dists: list[list[list[tuple[dict[str, int], int]]]] = [
        [] for _ in range(num_unique_samples)
    ]
    for i in range(num_unique_samples):
        for partition in quasi_dists_by_partition:
            quasi_dists[i].append(partition[i])

    return CuttingExperimentResults(quasi_dists, coefficients)


def _append_measurement_circuit(
    qc: QuantumCircuit,
    cog: CommutingObservableGroup,
    /,
    *,
    qubit_locations: Sequence[int] | None = None,
    inplace: bool = False,
) -> QuantumCircuit:
    """Append a new classical register and measurement instructions for the given ``CommutingObservableGroup``.

    The new register will be named ``"observable_measurements"`` and will be
    the final register in the returned circuit, i.e. ``retval.cregs[-1]``.

    Args:
        qc: The quantum circuit
        cog: The commuting observable set for
            which to construct measurements
        qubit_locations: A ``Sequence`` whose length is the number of qubits
            in the observables, where each element holds that qubit's corresponding
            index in the circuit.  By default, the circuit and observables are assumed
            to have the same number of qubits, and the identity map
            (i.e., ``range(qc.num_qubits)``) is used.
        inplace: Whether to operate on the circuit in place (default: ``False``)

    Returns:
        The modified circuit
    """
    if qubit_locations is None:
        # By default, the identity map.
        if qc.num_qubits != cog.general_observable.num_qubits:
            raise ValueError(
                f"Quantum circuit qubit count ({qc.num_qubits}) does not match qubit "
                f"count of observable(s) ({cog.general_observable.num_qubits}).  "
                f"Try providing `qubit_locations` explicitly."
            )
        qubit_locations = range(cog.general_observable.num_qubits)
    else:
        if len(qubit_locations) != cog.general_observable.num_qubits:
            raise ValueError(
                f"qubit_locations has {len(qubit_locations)} element(s) but the "
                f"observable(s) have {cog.general_observable.num_qubits} qubit(s)."
            )
    if not inplace:
        qc = qc.copy()

    # Append the appropriate measurements to qc
    obs_creg = ClassicalRegister(len(cog.pauli_indices), name="observable_measurements")
    qc.add_register(obs_creg)
    # Implement the necessary basis rotations and measurements, as
    # in BackendEstimator._measurement_circuit().
    genobs_x = cog.general_observable.x
    genobs_z = cog.general_observable.z
    for clbit, subqubit in enumerate(cog.pauli_indices):
        # subqubit is the index of the qubit in the subsystem.
        # actual_qubit is its index in the system of interest (if different).
        actual_qubit = qubit_locations[subqubit]
        if genobs_x[subqubit]:
            if genobs_z[subqubit]:
                qc.sdg(actual_qubit)
            qc.h(actual_qubit)
        qc.measure(actual_qubit, obs_creg[clbit])

    return qc


def _generate_cutting_experiments(
    circuits: QuantumCircuit | dict[str | int, QuantumCircuit],
    observables: PauliList | dict[str | int, PauliList],
    num_samples: int,
) -> tuple[list[list[list[QuantumCircuit]]], list[tuple[Any, WeightType]], list[float]]:
    """Generate all the experiments to run on the backend and their associated coefficients."""
    # Retrieving the unique bases, QPD gates, and decomposed observables is slightly different
    # depending on the format of the execute_experiments input args, but the 2nd half of this function
    # can be shared between both cases.
    if isinstance(circuits, QuantumCircuit):
        is_separated = False
        subcircuit_list = [circuits]
        subobservables_by_subsystem = decompose_observables(
            observables, "A" * len(observables[0])
        )
        subsystem_observables = {
            label: ObservableCollection(subobservables)
            for label, subobservables in subobservables_by_subsystem.items()
        }
        # Gather the unique bases from the circuit
        bases, qpd_gate_ids = _get_bases(circuits)
        subcirc_qpd_gate_ids = [qpd_gate_ids]

    else:
        is_separated = True
        subcircuit_list = [circuits[key] for key in sorted(circuits.keys())]
        # Gather the unique bases across the subcircuits
        subcirc_qpd_gate_ids, subcirc_map_ids = _get_mapping_ids_by_partition(
            subcircuit_list
        )
        bases = _get_bases_by_partition(subcircuit_list, subcirc_qpd_gate_ids)

        # Create the commuting observable groups
        subsystem_observables = {
            label: ObservableCollection(so) for label, so in observables.items()
        }

    # Sample the joint quasiprobability decomposition
    random_samples = generate_qpd_samples(bases, num_samples=num_samples)

    # Calculate terms in coefficient calculation
    kappa = np.prod([basis.kappa for basis in bases])
    num_samples = sum([value[0] for value in random_samples.values()])  # type: ignore

    # Sort samples in descending order of frequency
    sorted_samples = sorted(random_samples.items(), key=lambda x: x[1][0], reverse=True)

    # Generate the outputs -- sub-experiments, coefficients, and frequencies
    subexperiments: list[list[list[QuantumCircuit]]] = []
    coefficients = []
    sampled_frequencies = []
    for z, (map_ids, (redundancy, weight_type)) in enumerate(sorted_samples):
        subexperiments.append([])
        actual_coeff = np.prod(
            [basis.coeffs[map_id] for basis, map_id in strict_zip(bases, map_ids)]
        )
        sampled_coeff = (redundancy / num_samples) * (kappa * np.sign(actual_coeff))
        coefficients.append((sampled_coeff, weight_type))
        sampled_frequencies.append(redundancy)
        for i, (subcircuit, label) in enumerate(
            strict_zip(subcircuit_list, sorted(subsystem_observables.keys()))
        ):
            map_ids_tmp = map_ids
            if is_separated:
                map_ids_tmp = tuple(map_ids[j] for j in subcirc_map_ids[i])
            decomp_qc = decompose_qpd_instructions(
                subcircuit, subcirc_qpd_gate_ids[i], map_ids_tmp
            )
            subexperiments[-1].append([])
            so = subsystem_observables[label]
            for j, cog in enumerate(so.groups):
                meas_qc = _append_measurement_circuit(decomp_qc, cog)
                subexperiments[-1][-1].append(meas_qc)

    return subexperiments, coefficients, sampled_frequencies


def _run_experiments_batch(
    subexperiments: Sequence[Sequence[QuantumCircuit]],
    sampler: BaseSampler,
) -> list[list[tuple[QuasiDistribution, int]]]:
    """Run subexperiments on the backend."""
    num_qpd_bits_flat = []

    # Run all the experiments in one big batch
    experiments_flat = list(chain.from_iterable(subexperiments))

    for circ in experiments_flat:
        # if (
        #     len(circ.cregs) != 2
        #     or circ.cregs[1].name != "observable_measurements"
        #     or circ.cregs[0].name != "qpd_measurements"
        #     or sum([reg.size for reg in circ.cregs]) != circ.num_clbits
        # ):
        #     # If the classical bits/registers are in any other format than expected, the user must have
        #     # input them, so we can just raise this generic error in any case.
        #     raise ValueError(
        #         "Circuits input to execute_experiments should contain no classical registers or bits."
        #     )

        # Removed the qpd registers if there are no qpd measurements. This is to avoid errors with Azure
        if circ.cregs[0].name != "qpd_measurements":
            num_qpd_bits_flat.append(0)
        else:
            num_qpd_bits_flat.append(len(circ.cregs[0]))

    # Run all batched experiments
    quasi_dists_flat = sampler.run(experiments_flat).result().quasi_dists

    # Reshape the output data to match the input
    if len(subexperiments) == 1:
        quasi_dists_reshaped = np.array([quasi_dists_flat])
        num_qpd_bits = np.array([num_qpd_bits_flat])
    else:
        # We manually build the shape tuple in second arg because it behaves strangely
        # with QuantumCircuits in some versions. (e.g. passes local pytest but fails in tox env)
        quasi_dists_reshaped = np.reshape(
            quasi_dists_flat, (len(subexperiments), len(subexperiments[0]))
        )
        num_qpd_bits = np.reshape(
            num_qpd_bits_flat, (len(subexperiments), len(subexperiments[0]))
        )

    # Create the counts tuples, which include the number of QPD measurement bits
    quasi_dists: list[list[tuple[dict[str, float], int]]] = [
        [] for _ in range(len(subexperiments))
    ]
    for i, sample in enumerate(quasi_dists_reshaped):
        for j, prob_dict in enumerate(sample):
            quasi_dists[i].append((prob_dict, num_qpd_bits[i][j]))

    return quasi_dists


def _get_mapping_ids_by_partition(
    circuits: Sequence[QuantumCircuit],
) -> tuple[list[list[list[int]]], list[list[int]]]:
    """Get indices to the QPD gates in each subcircuit and relevant map ids."""
    # Collect QPDGate id's and relevant map id's for each subcircuit
    subcirc_qpd_gate_ids: list[list[list[int]]] = []
    subcirc_map_ids: list[list[int]] = []
    decomp_ids = set()
    for circ in circuits:
        subcirc_qpd_gate_ids.append([])
        subcirc_map_ids.append([])
        for i, inst in enumerate(circ.data):
            if isinstance(inst.operation, SingleQubitQPDGate):
                decomp_id = int(inst.operation.label.split("_")[-1])
                decomp_ids.add(decomp_id)
                subcirc_qpd_gate_ids[-1].append([i])
                subcirc_map_ids[-1].append(decomp_id)

    return subcirc_qpd_gate_ids, subcirc_map_ids


def _get_bases_by_partition(
    circuits: Sequence[QuantumCircuit], subcirc_qpd_gate_ids: list[list[list[int]]]
) -> list[QPDBasis]:
    """Get a list of each unique QPD basis across the subcircuits."""
    # Collect the bases corresponding to each decomposed operation
    bases_dict = {}
    for i, subcirc in enumerate(subcirc_qpd_gate_ids):
        for basis_id in subcirc:
            decomp_id = int(
                circuits[i].data[basis_id[0]].operation.label.split("_")[-1]
            )
            bases_dict[decomp_id] = circuits[i].data[basis_id[0]].operation.basis
    bases = [bases_dict[key] for key in sorted(bases_dict.keys())]

    return bases


def _get_bases(circuit: QuantumCircuit) -> tuple[list[QPDBasis], list[list[int]]]:
    """Get a list of each unique QPD basis in the circuit and the QPDGate indices."""
    bases = []
    qpd_gate_ids = []
    for i, inst in enumerate(circuit):
        if isinstance(inst.operation, SingleQubitQPDGate):
            raise ValueError(
                "SingleQubitQPDGates are not supported in unseparable circuits."
            )
        if isinstance(inst.operation, TwoQubitQPDGate):
            bases.append(inst.operation.basis)
            qpd_gate_ids.append([i])

    return bases, qpd_gate_ids


def _validate_samplers(samplers: BaseSampler | dict[str | int, BaseSampler]) -> None:
    """Replace unsupported statevector-based Samplers with ExactSampler."""
    if isinstance(samplers, BaseSampler):
        if (
            isinstance(samplers, AerSampler)
            and "shots" in samplers.options
            and samplers.options.shots is None
        ):
            _aer_sampler_error()
        elif isinstance(samplers, TerraSampler):
            _terra_sampler_error()

    elif isinstance(samplers, dict):
        for key, sampler in samplers.items():
            if (
                isinstance(sampler, AerSampler)
                and "shots" in sampler.options
                and sampler.options.shots is None
            ):
                _aer_sampler_error()
            elif isinstance(sampler, TerraSampler):
                _terra_sampler_error()
            elif isinstance(sampler, BaseSampler):
                continue
            else:
                _bad_samplers_error()

    else:
        _bad_samplers_error()


def _aer_sampler_error() -> None:
    raise ValueError(
        "qiskit_aer.primitives.Sampler does not support mid-circuit measurements when shots is None. "
        "Use circuit_knitting.utils.simulation.ExactSampler to generate exact distributions "
        "for each subexperiment.",
    )


def _terra_sampler_error() -> None:
    raise ValueError(
        "qiskit.primitives.Sampler does not support mid-circuit measurements. "
        "Use circuit_knitting.utils.simulation.ExactSampler to generate exact distributions "
        "for each subexperiment."
    )


def _bad_samplers_error() -> None:
    raise ValueError(
        "The samplers input argument must be either an instance of qiskit.primitives.BaseSampler "
        "or a mapping from partition labels to qiskit.primitives.BaseSampler instances."
    )

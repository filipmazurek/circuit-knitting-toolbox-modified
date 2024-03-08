import numpy as np

from qiskit.quantum_info import PauliList
from qiskit_aer.primitives import Estimator, Sampler
from qiskit import QuantumCircuit

from circuit_knitting.cutting import cut_gates, execute_experiments
from circuit_knitting.cutting.qpd import TwoQubitQPDGate
from circuit_knitting.utils.transforms import separate_circuit

from .util import cut_list_to_wire_and_gates, copy_and_add_ancilla, add_I_observables, reorder_observables
from .cutting import cut_circuit_gates_and_wires



def cut_wires_and_gates_to_subcircuits(
    circuit: QuantumCircuit,
    observables: list[str],
    method: str,
    max_subcircuit_width: int,
    max_cuts: int,
    num_subcircuits: list[int],
    model: str = 'cplex',
    verbose: bool = False,
) -> tuple[dict[int, QuantumCircuit], dict[int, PauliList]]:
    """Cut circuit wires using the gate and wire cut algorithm
    """

    cuts_list = cut_circuit_gates_and_wires(
        circuit=circuit,
        method=method,
        max_subcircuit_width=max_subcircuit_width,
        max_cuts=max_cuts,
        num_subcircuits=num_subcircuits,
        model=model,
        verbose=verbose,
    )

    wire_cuts, gate_cuts = cut_list_to_wire_and_gates(cuts_list)

    # Add gates to decompose given the cut locations
    ancilla_circ, cut_indices, qubit_mapping_ancilla = copy_and_add_ancilla(circuit, wire_cuts, gate_cuts)

    # Account for increasing the number of qubits by adding 'I' observables
    new_obs = add_I_observables(observables, qubit_mapping_ancilla, ancilla_circ.num_qubits)

    # Add decompositions to the circuit of increased size
    circ, basis = cut_gates(ancilla_circ, gate_ids=cut_indices)

    # Place numerical labels on the gates to be decomposed
    bases = []
    i = 0
    for inst in circ.data:
        if isinstance(inst.operation, TwoQubitQPDGate):
            bases.append(inst.operation.basis)
            inst.operation.label = inst.operation.label + f"_{i}"
            i += 1

    # Now actually decompose the gates using the decomposition information
    qpd_circuit_dx = circ.decompose(TwoQubitQPDGate)

    # This will fully separate the disconnected circuits into their own circuits
    subcircuits, qubit_mapping = separate_circuit(qpd_circuit_dx)

    # Adjust observables for the new qubit mapping
    newer_obs = reorder_observables(new_obs, qubit_mapping)

    # Make the observables into a dictionary of PauliLists
    subobservables = {}
    for key in newer_obs.keys():
        subobservables[key] = PauliList(newer_obs[key])

    return subcircuits, subobservables


def execute_simulation(subcircuits, subobservables, shots=2**12, samples=1500):
    # Give default values in case of Nones
    if shots is None:
        shots = 2**12
    if samples is None:
        samples = 1500
        
    samplers = {i: Sampler(run_options={"shots": shots}) for i in range(len(subcircuits.keys()))}

    # Run simulations. Have to use Aer Sampler with shot number
    quasi_dists, coefficients = execute_experiments(
        circuits=subcircuits,
        subobservables=subobservables,
        num_samples=samples,
        samplers=samplers,
    )

    return quasi_dists, coefficients


def exact_observables(circuit, observables):
    estimator = Estimator(run_options={"shots": None}, approximation=True)
    exact_expvals = (
        estimator.run([circuit] * len(observables), list(observables)).result().values
    )

    return exact_expvals


def compare_results(experiment_expvals, exact_expvals):
    print(
        f"Simulated expectation values: {[np.round(experiment_expvals[i], 8) for i in range(len(exact_expvals))]}"
    )
    print(
        f"Exact expectation values: {[np.round(exact_expvals[i], 8) for i in range(len(exact_expvals))]}"
    )
    print(
        f"Errors in estimation: {[np.round(experiment_expvals[i] - exact_expvals[i], 8) for i in range(len(exact_expvals))]}"
    )
    print(
        f"Relative errors in estimation: {[np.round((experiment_expvals[i] - exact_expvals[i]) / exact_expvals[i], 8) for i in range(len(exact_expvals))]}"
    )


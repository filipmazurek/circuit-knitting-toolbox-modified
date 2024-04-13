from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
# from qiskit.providers.fake_provider import GenericBackendV2
from circuit_knitting.cutting.gate_and_wire_cutting.frontend import cut_wires_and_gates_to_subcircuits
import numpy as np

a = 1
b = 1


def cut_draper_qft_adder(a, b):
    n_qubits = max(a.bit_length(), b.bit_length()) + 1
    qc = QuantumCircuit(2 * n_qubits - 1)
    a_binary = format(a, f'0{n_qubits}b')
    b_binary = format(b, f'0{n_qubits}b')
    # Encode the binary numbers in the quantum register
    for i, bit in enumerate(reversed(a_binary)):
        if bit == '1':
            qc.x(i + n_qubits)
    for i, bit in enumerate(reversed(b_binary)):
        if bit == '1':
            qc.x(i)
    qc.barrier()
    # Apply QFT to the numbers
    qc.append(QFT(n_qubits, do_swaps=False).to_gate(), range(n_qubits))
    qc.barrier()
    # Apply Draper Adder
    for i in range(n_qubits - 1):
        for j in range(i, n_qubits):
            # qc.cx(i+n_qubits, j)
            # qc.u(0, 0, 2 * np.pi / (2 ** (j - i + 1)), j)
            qc.cp(2 * np.pi / (2 ** (j - i + 1)), i + n_qubits, j)
    qc.barrier()
    # Apply IQFT
    qc.append(QFT(n_qubits, do_swaps=False).inverse().to_gate(), range(n_qubits))
    qc.barrier()
    # qc.measure(range(n_qubits), range(n_qubits))
    # Test when the observables are all 'Z'
    observables = ['YYYY']
    subcircuits, subobservables = cut_wires_and_gates_to_subcircuits(
        circuit=qc.decompose(reps=2),
        observables=observables,
        method='automatic',
        max_subcircuit_width=5,
        max_cuts=7,
        num_subcircuits=[2],
        model='gurobi'
    )
    if len(subcircuits) == 1:
        raise ValueError("The circuit was not cut into subcircuits")

    # Visualize the subcircuits. Note the decomposed 2-qubit gates marked 'cut_cx_0'
    for key in subcircuits.keys():
        print(subcircuits[key])

    from circuit_knitting.cutting.gate_and_wire_cutting.frontend import execute_simulation

    # Execute the subcircuits
    quasi_dists, coefficients = execute_simulation(subcircuits, subobservables)

    from circuit_knitting.cutting.cutting_reconstruction import reconstruct_expectation_values

    simulation_expvals = reconstruct_expectation_values(quasi_dists, coefficients, subobservables)
    print(simulation_expvals)

    # Create ideal results
    from circuit_knitting.cutting.gate_and_wire_cutting.frontend import exact_observables

    ideal_expvals = exact_observables(qc, observables)
    print(ideal_expvals)
    # Compare the error between results

    from circuit_knitting.cutting.gate_and_wire_cutting.frontend import compare_results

    compare_results(simulation_expvals, ideal_expvals)
    # backend = GenericBackendV2(num_qubits=2*n_qubits)
    # transpiled_circuit = transpile(qc, backend)
    # job = backend.run(transpiled_circuit)
    # counts = job.result().get_counts()
    # return counts, qc


cut_draper_qft_adder(a, b)
# counts, qc = cut_draper_qft_adder(a, b)
# print(f"Counts: {counts}")
# _sum = max(counts, key=counts.get)
# print(f"Sum in binary: {_sum}")
# sum_decimal = int(_sum[0:], 2)
# print(f"Sum in decimal: {sum_decimal}")
# qc.decompose(reps=2).draw('mpl')
############
# QPD cutting and adders:
#   Because QPD cutting is only useful for expectation value
#
#
############

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.circuit.library import CDKMRippleCarryAdder

a = 1
b = 3


def cut_ripple_carry_adder(a, b):
    n_qubits = max(a.bit_length(), b.bit_length())
    operand1 = QuantumRegister(n_qubits, 'o1')
    operand2 = QuantumRegister(n_qubits, 'o2')
    anc = QuantumRegister(2, 'a')
    cr = ClassicalRegister(n_qubits + 1)
    qc = QuantumCircuit(operand1, operand2, anc, cr)
    for i in range(n_qubits):
        if (a & (1 << i)):
            qc.x(operand1[i])
        if (b & (1 << i)):
            qc.x(operand2[i])
    adder = CDKMRippleCarryAdder(n_qubits, 'full', 'Full Adder')
    qc.append(adder, [anc[0]] + operand1[:] + operand2[:] + [anc[1]])
    # qc.measure(operand2[:] + [anc[1]], cr)
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

    ideal_expvals = exact_observables(circuit, observables)
    print(ideal_expvals)
    # Compare the error between results

    from circuit_knitting.cutting.gate_and_wire_cutting.frontend import compare_results

    compare_results(simulation_expvals, ideal_expvals)
    # backend = GenericBackendV2(num_qubits=2*n_qubits + 2)
    # transpiled_circuit = transpile(qc, backend)
    # job = backend.run(transpiled_circuit)
    # counts = job.result().get_counts()
    # return counts, qc


cut_ripple_carry_adder(a, b)
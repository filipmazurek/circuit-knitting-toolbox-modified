from qiskit import QuantumCircuit

from circuit_knitting.cutting.gate_and_wire_cutting.frontend import cut_wires_and_gates_to_subcircuits


# Create a quantum circuit to cut. We create a simple ansatz
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.barrier()
circuit.cx(0, 1)
print(circuit)

# Test when the observables are all 'I'
observables = ['II']

subcircuits, subobservables = cut_wires_and_gates_to_subcircuits(
    circuit=circuit,
    observables=observables,
    method='automatic',
    max_subcircuit_width=1,
    max_cuts=9,
    num_subcircuits=[2],
    model='gurobi'
)

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
from qiskit.circuit.random import random_circuit

from circuit_knitting.cutting.gate_and_wire_cutting.frontend import cut_wires_and_gates_to_subcircuits


circuit = random_circuit(8, 5, measure=False).decompose(reps=3)

print(circuit)

# Test when the observables are all 'I'
observables = ['IIIIIIII']

subcircuits, subobservables = cut_wires_and_gates_to_subcircuits(
    circuit=circuit,
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
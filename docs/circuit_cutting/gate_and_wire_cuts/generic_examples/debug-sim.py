from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.circuit.library import EfficientSU2

# Create a quantum circuit to cut. We create a simple ansatz
mapper = JordanWignerMapper()
ansatz = EfficientSU2(3, reps=1)
# Decompose to the actual individual gates
circuit = ansatz.decompose(reps=3)
# Set some arbitrary parameters
circuit.assign_parameters([0.8] * len(circuit.parameters), inplace=True)

print(circuit)

from circuit_knitting.cutting.gate_and_wire_cutting.frontend import cut_wires_and_gates_to_subcircuits

observables = ["ZZI", "IZZ", "IIZ", "XIX", "ZIZ", "IXI"]

subcircuits, subobservables = cut_wires_and_gates_to_subcircuits(
    circuit=circuit,
    observables=observables,
    method='automatic',
    max_subcircuit_width=2,
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
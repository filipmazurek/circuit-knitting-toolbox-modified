from circuit_knitting.cutting.gate_and_wire_cutting.frontend import cut_wires_and_gates_to_subcircuits
from qiskit import QuantumCircuit

num_qubits = 4
observables = ['Z'*num_qubits]

circ = QuantumCircuit(num_qubits)

# Set up a minimal example where the MIP solver finds a solution that does not lead to separable subcircuits

for i in range(2):
    circ.cx(0, 1)

circ.cx(0, 3)

for i in range(2):
    circ.cx(1, 3)

for i in range(4):
    circ.cx(0, 2)

for i in range(2):
    circ.cx(0, 1)

for i in range(2):
    circ.cx(0, 3)

circ.cx(1, 3)

for i in range(3):
    circ.cx(0, 2)


print(circ)

_, _, num_wire_cuts, num_gate_cuts = cut_wires_and_gates_to_subcircuits(
    circuit=circ,
    observables=observables,
    method='automatic',
    max_subcircuit_width=3,
    max_cuts=100,
    num_subcircuits=[2],
    model='gurobi'
)

print(num_wire_cuts, num_gate_cuts)

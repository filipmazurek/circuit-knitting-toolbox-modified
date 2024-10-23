from circuit_knitting.cutting.gate_and_wire_cutting.frontend import cut_wires_and_gates_to_subcircuits
from qiskit.circuit.library import ExcitationPreserving


num_qubits = 5
circ = ExcitationPreserving(num_qubits, flatten=True, reps=2, skip_final_rotation_layer=True).decompose(reps=2)
observables = ['Z'*num_qubits]

# # Remove all single-qubit gates to simplify visually
# gates_to_remove = []
# for i in range(len(circ.data)):
#     if circ.data[i].operation.num_qubits == 1:
#         gates_to_remove.append(i)
#
# # Remove in reverse order to avoid index issues
# for i in reversed(gates_to_remove):
#     circ.data.pop(i)
#
# # Remove the first X gates to simplify the problem
# for _ in range(11):
#     circ.data.pop()
#
# # Remove the last X gates to simplify the problem
# for _ in range(14):
#     circ.data.pop(-1)

print(circ)

_, _, num_wire_cuts, num_gate_cuts = cut_wires_and_gates_to_subcircuits(
    circuit=circ,
    observables=observables,
    method='automatic',
    max_subcircuit_width=4,
    max_cuts=100,
    num_subcircuits=[2],
    model='gurobi'
)

print(num_wire_cuts, num_gate_cuts)

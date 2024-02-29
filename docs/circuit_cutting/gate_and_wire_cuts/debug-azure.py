from azure.quantum import Workspace
from azure.quantum.qiskit import AzureQuantumProvider


workspace = Workspace(
            resource_id = "/subscriptions/4ed1f7fd-7d9e-4c61-9fac-521649937e65/resourceGroups/Cutting/providers/Microsoft.Quantum/Workspaces/Cutting",
            location = "eastus")


provider = AzureQuantumProvider(workspace)

print("This workspace's targets:")
for backend in provider.backends():
    print("- " + backend.name())


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
    num_subcircuits=[2]
)

# Visualize the subcircuits. Note the decomposed 2-qubit gates marked 'cut_cx_0'
for key in subcircuits.keys():
    print(subcircuits[key])
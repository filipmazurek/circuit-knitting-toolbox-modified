from azure.quantum import Workspace
from azure.quantum.qiskit import AzureQuantumProvider
from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA

from circuit_knitting.cutting.gate_and_wire_cutting.algorithms.cut_vqe import CutVQE

H2_op_list = [('II', -1.0523732457728596),
              ('IZ', 0.39793742484317934),
              ('XX', 0.18093119978423144),
              ('ZI', -0.39793742484317934),
              ('ZZ', -0.011280104256235268)]
# Problem specification. Hamiltonian for H2 at 0.735A interatomic distance.
H2_op = SparsePauliOp.from_list(H2_op_list)

# Create ansatz and immediately decompoe to viualize gates
ansatz = EfficientSU2(H2_op.num_qubits, reps=1)
circuit = ansatz.decompose(reps=3)

# Check how many parameters are in the ansatz
num_params = circuit.num_parameters

# Extract the pauli operators from the operations lit
observables = H2_op.paulis

# Set up the azure backend
workspace = Workspace(
    resource_id="/subscriptions/4ed1f7fd-7d9e-4c61-9fac-521649937e65/resourceGroups/Cutting/providers/Microsoft.Quantum/Workspaces/Cutting",
    location="eastus")
provider = AzureQuantumProvider(workspace)
# Using syntax checker for free
backend = provider.get_backend("quantinuum.sim.h1-1sc")

usable_backends = ['quantinuum.sim.h1-1sc', 'quantinuum.sim.h1-2sc', 'quantinuum.sim.h1-1e', 'quantinuum.sim.h1-2e',
                   'quantinuum.qpu.h1-1', 'quantinuum.qpu.h1-2']


# Callback function to check VQE process as it runs
def callback(eval_count, parameters, mean, std):
    print(f"Round num: {eval_count}, energy: {mean}, parameters: {parameters}")


# Define  estimator and optimizer
estimator = Estimator()
optimizer = COBYLA(maxiter=2)  # 2 Iteration for testing

# Set up VQE
vqe = CutVQE(estimator, circuit, optimizer, observables=observables, shots=2 ** 12, max_subcircuit_width=1, max_cuts=9,
             num_subcircuits=[2], model='gurobi', num_samples=1500, callback=callback, backend='azure',
             azure_backend=backend)

with backend.open_session(name="Qiskit Session") as session:
    result = vqe.compute_minimum_eigenvalue(H2_op)

print(result)

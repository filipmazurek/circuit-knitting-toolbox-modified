from qiskit.circuit.library import EfficientSU2, TwoLocal
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


from qiskit.quantum_info import SparsePauliOp

H2_op_list = [('II', -1.0523732457728596),
             ('IZ', 0.39793742484317934),
             ('XX', 0.18093119978423144),
             ('ZI', -0.39793742484317934),
             ('ZZ', -0.011280104256235268)]
# Problem specification. Hamiltonian for H2 at 0.735A interatomic distance.
H2_op = SparsePauliOp.from_list(H2_op_list)

ansatz = EfficientSU2(H2_op.num_qubits, reps=1)
# ansatz = TwoLocal(num_qubits=2, rotation_blocks=["ry", "rz"], entanglement_blocks="cz")

circuit = ansatz.decompose(reps=3)

# See how many parameters are in our ansatz
num_params = circuit.num_parameters

observables = H2_op.paulis

# Cut the circuit using our decomposition technique

# VQE
from qiskit_algorithms.minimum_eigensolvers import VQE, AdaptVQE
from qiskit_algorithms.optimizers import SLSQP, COBYLA
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.primitives import Estimator

def callback(eval_count, parameters, mean, std):
    print(f"Round num: {eval_count}, energy: {mean}, parameters: {parameters}")

# Define our estimator and optimizer
estimator = Estimator()
optimizer = COBYLA(maxiter=2)

from circuit_knitting.cutting.gate_and_wire_cutting.algorithms.cut_vqe import CutVQE
# Run VQE and print our results
vqe = CutVQE(estimator, circuit, optimizer, observables=observables, shots=2**12, max_subcircuit_width=1, max_cuts=9, num_subcircuits=[2], model='gurobi', num_samples=1500, callback=callback)
result = vqe.compute_minimum_eigenvalue(H2_op)
print(result)


print('-------------------------')
print('WITHOUT CUTTING')

# Create a callback to print the current value of the objective function


# Run VQE and print our results
no_cut_vqe = VQE(ansatz=circuit, optimizer=COBYLA(maxiter=80), estimator=Estimator(), callback=callback)
no_cut_result = no_cut_vqe.compute_minimum_eigenvalue(H2_op)
print(no_cut_result)
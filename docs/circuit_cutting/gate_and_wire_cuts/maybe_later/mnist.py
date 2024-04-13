import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize
from sklearn.preprocessing import normalize
from tensorflow.keras.datasets import mnist
from qiskit.providers.fake_provider import GenericBackendV2
from qpie import QPIE

qpie = QPIE()
num_qubits = 4  # Using 4 qubits


def feature_map(image: np.ndarray, num_qubits: int, measurements: bool = False) -> QuantumCircuit:
    """Return a QPIE circuit that encodes the image given as input."""
    # Ensure the image is flattened and normalized
    flattened_image = image.reshape(1, -1).flatten()
    if np.count_nonzero(flattened_image) == 0:
        raise ValueError("The image is an array of zeros, which can't be normalized to a state vector.")

    normalized_img = normalize([flattened_image], norm='l2').flatten()
    if np.count_nonzero(normalized_img) == 0:
        raise ValueError(
            "Normalization resulted in an array of zeros, which is invalid for state vector initialization.")

    # Create a quantum circuit
    qubits = QuantumRegister(num_qubits, name="q")
    qc = QuantumCircuit(qubits)
    # Initialize the quantum circuit with the normalized image
    qc.initialize(normalized_img[:2 ** num_qubits], qubits)  # Use only the first 2^num_qubits elements
    return qc


def variational_form(params, num_qubits):
    """Define a simple variational form"""
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rx(params[i], i)
        if i < num_qubits - 1:
            qc.cry(params[num_qubits + i], i, i + 1)
    return qc


def cost_function(params, feature_map_func, variational_form_func, data, labels, num_qubits):
    """Define the cost function"""
    # print("Step")
    cost = 0
    for x, y in zip(data, labels):
        # Create the circuit
        fm_circuit = feature_map_func(x, num_qubits)
        vf_circuit = variational_form_func(params, num_qubits)
        qc = fm_circuit & vf_circuit
        qc.measure_all()
        # Execute the circuit
        transpiled_circuit = transpile(qc, backend)
        job = backend.run(transpiled_circuit)
        result = job.result().get_counts(qc)
        # Compute the cost
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys()))
        # Assuming the first qubit is the classification qubit
        probabilities = np.array([count if state[-1] == '0' else -1 * count for state, count in zip(states, counts)])
        expectation_value = np.sum(probabilities) / shots
        predicted = 1 if expectation_value > 0 else 0  # Adjust based on your encoding of labels
        if y.any() > 0:
            y = 1
        else:
            y = 0
        cost += (predicted != y) ** 2
    print("Cost:", cost / len(data))
    return cost / len(data)


backend = GenericBackendV2(num_qubits=num_qubits)
shots = 1024

# Load and preprocess the MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_size = 800
test_size = 200

# Simplify the problem to binary classification (0s and 1s for simplicity)
is_01 = (y_train == 0) | (y_train == 1)
x_train, y_train = x_train[is_01][:train_size], y_train[is_01][:train_size]  # Using only 100 samples for demonstration
is_01 = (y_test == 0) | (y_test == 1)
x_test, y_test = x_test[is_01][:test_size], y_test[is_01][:test_size]  # Using only 20 samples for testing

# Reduce the image size
# Here, we resize the image to 4x4 (16 pixels) to match a 4-qubit system.
# You can choose different resizing methods, but make sure to keep the aspect ratio and flatten the image.
# x_train = x_train[:, ::7, ::7].reshape(-1, 16)  # Simple downsampling
# x_test = x_test[:, ::7, ::7].reshape(-1, 16)

img_size = 4
# Resize all images
x_train = qpie.resize_image(x_train, new_size=img_size, multiple_images=True)
x_test = qpie.resize_image(x_test, new_size=img_size, multiple_images=True)
y_train = qpie.resize_image(y_train, new_size=img_size, multiple_images=True)
y_test = qpie.resize_image(y_test, new_size=img_size, multiple_images=True)

# Set up and run the optimizer
theta = np.random.rand(2 ** num_qubits) * 2 * np.pi  # Adjust the parameter vector size
result = minimize(cost_function, theta, args=(feature_map, variational_form, x_train, y_train, num_qubits),
                  method='COBYLA', options={'maxiter': 10})
optimized_theta = result.x

num_correct = 0

# Test the trained model
for x, y in zip(x_test, y_test):
    fm_circuit = feature_map(x, num_qubits)
    vf_circuit = variational_form(optimized_theta, num_qubits)
    qc = fm_circuit & vf_circuit
    qc.measure_all()
    transpiled_circuit = transpile(qc, backend)
    job = backend.run(transpiled_circuit)
    result = job.result().get_counts(qc)
    counts = np.array(list(result.values()))
    states = np.array(list(result.keys()))
    probabilities = np.array([count if state[-1] == '0' else -1 * count for state, count in zip(states, counts)])
    expectation_value = np.sum(probabilities) / 1024
    predicted = 1 if expectation_value > 0 else 0
    if y.any() > 0:
        y = 1
    else:
        y = 0

    if y == predicted:
        num_correct += 1
    print(f"Expected: {y}, Predicted: {predicted}")

print("Final accuracy:", num_correct / test_size)
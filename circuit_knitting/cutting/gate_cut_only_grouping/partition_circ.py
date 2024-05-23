import numpy as np
import pymetis
# Requires PyMetis, https://anaconda.org/conda-forge/pymetis
from qiskit import QuantumCircuit, QuantumRegister


def _graph_to_csr(edge_tuples):
    """Given a list of tuples of the form (vertex1, vertex2, weight), returns the CSR arrays for the graph. This
    method assumes an undirected graph."""

    # Get only the vertices used in the circuit - drop the weights
    vertices_tuples = list(zip(*edge_tuples))[0:2]
    vertices = [v for t in vertices_tuples for v in t]
    # Get the biggest vertex label
    biggest_vertex_label = max(vertices)

    # Prepare arrays - the index of each array is the vertex label and the entries of each array are connected vertices
    connected_vertices = [[] for _ in range(biggest_vertex_label + 1)]
    edge_weights = [[] for _ in range(biggest_vertex_label + 1)]

    # Fill in the array based on edges and their weights. Add edges from both directions.
    for edge_tuple in edge_tuples:
        edge_weight = edge_tuple[2]
        connected_vertices[edge_tuple[0]].append(edge_tuple[1])
        edge_weights[edge_tuple[0]].append(edge_weight)
        connected_vertices[edge_tuple[1]].append(edge_tuple[0])
        edge_weights[edge_tuple[1]].append(edge_weight)

    # Create the xadj, and adjncy, and eweights lists.
    # xadj is the index of the start of each vertex's connected vertices in the adjncy list
    # adjncy is the list of connected vertices
    # eweights is the list of corresponding edge weights for adjncy
    xadj_list = [0]
    adjncy_list = []
    eweights_list = []
    for i in range(biggest_vertex_label + 1):
        adjncy_list += connected_vertices[i]
        eweights_list += edge_weights[i]
        xadj_list.append(len(connected_vertices[i]) + xadj_list[-1])

    return xadj_list, adjncy_list, eweights_list


def _circuit_to_graph(circ: QuantumCircuit):
    """Given a qiskit QuantumCircuit, returns tuples of which vertices are connected via 2-qubit gates.
    The tuples are of the form (qubit1, qubit2, weight)"""
    # Get the 2-qubit gates
    two_qubit_gates = [i for i, gate in enumerate(circ.data) if gate.operation.num_qubits == 2]
    # Get the qubits involved in each 2-qubit gate
    two_qubit_gate_qubits = [(circ.find_bit(circ.data[i][1][0]).index, circ.find_bit(circ.data[i][1][1]).index) for i in two_qubit_gates]
    # Order the two qubits in each tuple so that the first element is the smaller qubit index
    two_qubit_gate_qubits = [tuple(sorted(t)) for t in two_qubit_gate_qubits]
    # Get the number of 2-qubit gates between each pair of qubits
    two_qubit_gate_qubits = [(t[0], t[1], two_qubit_gate_qubits.count(t)) for t in set(two_qubit_gate_qubits)]

    return two_qubit_gate_qubits


def _membership_lists_to_partition(membership_lists):
    """Given a list of lists of membership, return string representation of the partition."""
    partition = ''
    for i, membership_list in enumerate(membership_lists):
        partition += f'{i}'

    return partition

def circuit_to_partitions(circ, num_groups):
    graph = _circuit_to_graph(circ)
    xadj_list, adjncy_list, eweights_list = _graph_to_csr(graph)
    # FIXME: This method does not take into account maximum group size, only the number of groups
    #   Therefore it is useless for cutting in its current state

q = QuantumRegister(4, 'q')
circ = QuantumCircuit(4)
circ.cx(q[0], q[1])
circ.cx(q[0], q[1])
circ.cx(q[0], q[1])
circ.cx(q[1], q[2])
circ.cx(q[1], q[2])
circ.cx(q[2], q[3])
graph = _circuit_to_graph(circ)
# print(graph)
xadj_list, adjncy_list, eweights_list = _graph_to_csr(graph)
# print(f'{xadj_list=}')
# print(f'{adjncy_list=}')
# print(f'{eweights_list=}')
num_groups = 2
n_cuts, membership = pymetis.part_graph(num_groups, xadj=xadj_list, adjncy=adjncy_list, eweights=eweights_list)
# print(n_cuts)
# print(np.argwhere(np.array(membership) == 0).ravel())
# print(np.argwhere(np.array(membership) == 1).ravel())

# Assign membership numbers to each of the vertices
membership_by_vertex = [0] * (len(xadj_list) - 1)
for i in range(num_groups):
    in_group = np.argwhere(np.array(membership) == i).ravel()
    for j in in_group:
        membership_by_vertex[j] = i

# Turn the membership list into a string (NOTE: only works up to 9 partitions)
membership_str = ''.join(str(num) for num in membership_by_vertex)
print(membership_str)

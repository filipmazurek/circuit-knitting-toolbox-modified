from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.circuit import Parameter
import re

def cut_list_to_wire_and_gates(cuts) -> (list[str], list[tuple[str]]):
    """
    Process the output of all cuts into a list of wire cuts and gate cuts.
    """
    wire_cuts = []
    gate_cuts = []
    for cut in cuts:
        loc1 = cut[0]
        loc2 = cut[1]
        # Check if loc1 and loc2 refer to the same qubit. Found by comparing everything before the ]
        if loc1.split(']')[0] == loc2.split(']')[0]:
            # If it is, it's a wire cut. Add the loc with the smaller last number
            if int(loc1.split(']')[1]) < int(loc2.split(']')[1]):
                wire_cuts.append(loc1)
            else:
                wire_cuts.append(loc2)
        else:
            # If it's not, it's a gate cut. Add the cut to the gate cuts list
            gate_cuts.append(cut)

    return wire_cuts, gate_cuts


def copy_and_add_ancilla(original_circ: QuantumCircuit, wire_cut_locations, gate_cut_locations) -> (QuantumCircuit, list[int], list[(tuple[int])]):
    """
    Returns
      1. The new quantum circuit
      2. The gate indices to be decomposed
      3. The mapping of original qubits to new qubits (at the end). This is for adding I's to the observables
    """
    # Find the number of unique qubits in the wire cuts, and which qubits are cut
    wire_cut_dict = {}

    # Keep track of the total gate index
    total_gate_index = -1
    # List of cut gates
    cut_gate_indices = []

    for cut in wire_cut_locations:
        # Get the qubit number from the string
        qubit_indx = int((re.search(r'\[(\d+)\]', cut)).group(1))
        cut_index = int((re.search(r'\](\d+)', cut)).group(1))

        # Add the cut index to the dictionary. If the qubit is already in the dictionary, append the cut index to the list
        if original_circ.qubits[qubit_indx] in wire_cut_dict:
            wire_cut_dict[original_circ.qubits[qubit_indx]].append(cut_index)
        else:
            wire_cut_dict[original_circ.qubits[qubit_indx]] = [cut_index]

    # Process gate cuts by replacing the string with a tuple of the actual qubits
    gate_cut_dict = {}
    for cut in gate_cut_locations:
        # Get the qubit numbers from the string
        qubit_index = int((re.search(r'\[(\d+)\]', cut[0])).group(1))
        cut_index = int((re.search(r'\](\d+)', cut[0])).group(1))

        # Add the cut location to teh dictionary. If the qubit is already in the dictionary, append the cut index to the list
        if original_circ.qubits[qubit_index] in gate_cut_dict:
            gate_cut_dict[original_circ.qubits[qubit_index]].append(cut_index)
        else:
            gate_cut_dict[original_circ.qubits[qubit_index]] = [cut_index]

    # Create a new quantum register with the same number of qubits as the original circuit, plus the number of cut qubits
    q = QuantumRegister(original_circ.num_qubits + len(list(wire_cut_dict.keys())))

    # Create a dictionary of old qubits to new qubits
    qubit_mapping = {}
    for i in range(original_circ.num_qubits):
        qubit_mapping[original_circ.qubits[i]] = q[i]

    # Add the same number of classical bits as the original circuit
    c = ClassicalRegister(original_circ.num_clbits)

    # Create a mapping of classical bits
    bit_mapping = {}
    for i in range(original_circ.num_clbits):
        bit_mapping[original_circ.clbits[i]] = c[i]

    # Replace the old qubit register with the new one
    if original_circ.num_clbits > 0:
        ghz_copy = QuantumCircuit(q, c)
    else:
        ghz_copy = QuantumCircuit(q)

    # Keep track of all the parameters from the original circuit
    ghz_parameters = original_circ.parameters
    new_parameters = {}  # Key: old parameter, Value: new parameter
    for parameter in ghz_parameters:
        new_parameters[parameter] = Parameter(parameter.name)

    # Add a dictionary for what ancilla qubits correspond to which usable qubits, in both directions
    ancilla_mapping = {}
    ancilla_mapping_reverse = {}
    for i, qubit in enumerate(list(wire_cut_dict.keys())):
        # All "extra" qubits are ancilla qubits
        # No need for a +1 because num_qubits is already the total value
        ancilla_mapping[qubit_mapping[qubit]] = q[original_circ.num_qubits + i]
        ancilla_mapping_reverse[q[original_circ.num_qubits + i]] = qubit_mapping[qubit]

    # An "active" tracker to see which qubits are currently being used. This starts off with all the first qubits of the original circuit
    # It will be used to keep track of which qubits are currently being used
    active_qubits = [q[i] for i in range(original_circ.num_qubits)]

    # Dictionary to keep track of the number of two-qubit gates on each qubit
    gate_index_dict = {}
    for i, qubit in enumerate(q):
        if i < original_circ.num_qubits:
            gate_index_dict[qubit] = -1
        else:
            break

    # TODO: another option if this doesn't work: Replace the circuit operation qubits in place in case of wire cuts

    # Add the operations from the original circuit to the new circuit
    for operation in original_circ.data:
        # If it is a barrier, add a barrier
        if operation[0].name == 'barrier':
            ghz_copy.barrier()
            # Barriers count as operations. Increment the total gate index
            total_gate_index += 1
        # If the operation acts on one qubit
        elif len(operation.qubits) == 1:
            # Get the qubit that the operation is acting on
            qubit = operation.qubits[0]
            # Check if the operation is a measurement instruction
            if operation[0].name == 'measure':
                # If it is, add a measurement instruction to the new circuit
                ghz_copy.measure([qubit_mapping[qubit]], [bit_mapping[operation.clbits[0]]])
            else:
                # Add the operation to the new circuit
                ghz_copy.append(operation[0], [qubit_mapping[qubit]])
                # Increment the total gate index
                total_gate_index += 1
        # If the operation is a 2-qubit gate, do the same
        elif len(operation.qubits) == 2:
            qubit1 = operation.qubits[0]
            qubit2 = operation.qubits[1]
            ghz_copy.append(operation[0], [qubit_mapping[qubit1], qubit_mapping[qubit2]])
            # Increment the total gate index
            total_gate_index += 1
            # Add the gate index for the corresponding qubits. But first take into account if the qubit is an ancilla qubit
            if qubit_mapping[qubit1] in gate_index_dict:
                gate_index_dict[qubit_mapping[qubit1]] += 1
            else:
                gate_index_dict[ancilla_mapping_reverse[qubit_mapping[qubit1]]] += 1
            if qubit_mapping[qubit2] in gate_index_dict:
                gate_index_dict[qubit_mapping[qubit2]] += 1
            else:
                gate_index_dict[ancilla_mapping_reverse[qubit_mapping[qubit2]]] += 1
            # Check if the gate is a gate cut. If it is, add the total index to the list of cut gate indices
            if qubit1 in gate_cut_dict and gate_index_dict[qubit_mapping[qubit1]] in gate_cut_dict[qubit1]:
                cut_gate_indices.append(total_gate_index)
            elif qubit2 in gate_cut_dict and gate_cut_dict[qubit2] == gate_index_dict[qubit_mapping[qubit2]]:
                cut_gate_indices.append(total_gate_index)

            # Check if there is a wire cut after this 2-qubit operation (on either, or both, of the 2 qubits)
            for check_qubit in (qubit1, qubit2):
                if check_qubit in wire_cut_dict:
                    # Check what the current gate index is
                    if qubit_mapping[check_qubit] in gate_index_dict:
                        current_gate_index = gate_index_dict[qubit_mapping[check_qubit]]
                    else:
                        current_gate_index = gate_index_dict[ancilla_mapping_reverse[qubit_mapping[check_qubit]]]
                    # If the current gate index is in the wire cut dictionary, add a SWAP operation
                    if current_gate_index in wire_cut_dict[check_qubit]:
                        # Remove the gate index from the wire cut dictionary
                        wire_cut_dict[check_qubit].remove(current_gate_index)
                        # Increment the values of the cut indices for the qubit that was cut (because we are adding two CX)
                        wire_cut_dict[check_qubit] = [x + 2 for x in wire_cut_dict[check_qubit]]
                        # Add a SWAP operation (only need two CX because ancilla should be 0)
                        ghz_copy.cx(ancilla_mapping[qubit_mapping[check_qubit]], qubit_mapping[check_qubit])
                        ghz_copy.cx(qubit_mapping[check_qubit], ancilla_mapping[qubit_mapping[check_qubit]])
                        # Cut both of these gate indices
                        total_gate_index += 1
                        cut_gate_indices.append(total_gate_index)
                        total_gate_index += 1
                        cut_gate_indices.append(total_gate_index)
                        # DO NOT add to the gate_index_dict, as that would confuse which gates to cut
                        # TODO: add a reset operation?
                        # Change the active qubit
                        active_qubits.remove(qubit_mapping[check_qubit])
                        active_qubits.append(ancilla_mapping[qubit_mapping[check_qubit]])
                        # Swap which qubit is the active and which is the ancilla
                        temp = qubit_mapping[check_qubit]
                        qubit_mapping[check_qubit] = ancilla_mapping[qubit_mapping[check_qubit]]
                        ancilla_mapping[qubit_mapping[check_qubit]] = temp
                        # DO NOT mess with the individual gate index dict, as that would lose our spot with the gate cuts

    # Create a mapping of where the original qubits were to where the resulting qubits are. For observable mapping
    qubit_map = []
    for i, qubit in enumerate(original_circ.qubits):
        qubit_map.append((i, qubit_mapping[qubit]._index))
        #qubit_map.append((i, qubit_mapping[qubit].index))

    return ghz_copy, cut_gate_indices, qubit_map

def add_I_observables(observables_list, qubit_map, circuit_size):
    """ Add identity observables to the additional qubits that don't merit measurement at the end anymore.
    """
    new_obs_list = []

    observables_list = [obs[::-1] for obs in observables_list]

    for observable_str in observables_list:
        new_observables = ['I'] * circuit_size

        for original_qubit, mapped_qubit in qubit_map:
            for i, observable in enumerate(observable_str):
                if i == original_qubit:
                    new_observables[mapped_qubit] = observable

        new_observables = new_observables[::-1]
        new_obs_list.append(''.join(new_observables))

    return new_obs_list

def reorder_observables(observables, qubit_map):
    # Determine the maximum subcircuit index to know how many subcircuits there are
    max_subcircuit = max(subcircuit for subcircuit, _ in qubit_map)

    # Initialize a dictionary to hold the new observables for each subcircuit
    new_observables = {i: [] for i in range(max_subcircuit + 1)}

    # Fill each subcircuit's base observable with 'I's corresponding to the maximal qubit index mapped to that subcircuit
    for sc in new_observables.keys():
        length = max(pos for sc_map, pos in qubit_map if sc_map == sc) + 1
        new_observables[sc] = [[['0'] for _ in range(length)] for _ in range(len(observables))]

    # Iterate through each of the observables
    # Note that starting with qubit 0 means the rightmost observable character
    for i, qubit_location in enumerate(qubit_map):
        # Each qubit is mapped to a subcircuit and a position in that subcircuit
        subcircuit, position = qubit_location
        # For every observable, add the observable to the correct subcircuit
        for j, observable in enumerate(observables):
            # Only consider the Pauli for this qubit.
            pauli = observable[len(observable) - i - 1]
            new_observables[subcircuit][j][len(new_observables[subcircuit][j]) - position - 1] = pauli

    # Concatenate the interior lists into strings
    for sc, obs in new_observables.items():
        new_observables[sc] = [''.join(obs[i]) for i in range(len(obs))]

    return new_observables
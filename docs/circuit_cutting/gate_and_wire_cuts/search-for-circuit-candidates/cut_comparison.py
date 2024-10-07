# Compare the number of cuts in CutQC and against gate and wire cutting

from circuit_knitting.cutting.cutqc import cut_circuit_wires
from circuit_knitting.cutting.gate_and_wire_cutting.cutting import cut_circuit_gates_and_wires
from circuit_knitting.cutting.gate_and_wire_cutting.util import copy_and_add_ancilla, cut_list_to_wire_and_gates


def _cut_comparison(circuit, max_cuts, num_subcircuits, max_subcircuit_width):
    # CutQC
    subcircuits_cutqc = cut_circuit_wires(
        circuit=circuit,
        method='automatic',
        max_cuts=max_cuts,
        num_subcircuits=num_subcircuits,
        max_subcircuit_width=max_subcircuit_width
    )

    num_cuts_cutqc = subcircuits_cutqc['num_cuts']

    # Gate and wire
    cuts_gate_and_wire = cut_circuit_gates_and_wires(
        circuit=circuit,
        method='automatic',
        max_cuts=max_cuts,
        num_subcircuits=num_subcircuits,
        max_subcircuit_width=max_subcircuit_width,
    )
    wire_cuts, gate_cuts = cut_list_to_wire_and_gates(cuts_gate_and_wire)
    _, cut_indices, _ = copy_and_add_ancilla(circuit, wire_cuts, gate_cuts)
    num_cuts_gate_wire = len(cut_indices)

    return num_cuts_cutqc, num_cuts_gate_wire


# Create a simple test circuit
def _example_use():
    from qiskit.circuit.library import EfficientSU2

    circuits = []
    for val in range(0.1, 1.0, 0.1):
        # Create a quantum circuit to cut. We create a simple ansatz
        ansatz = EfficientSU2(3, reps=1)
        # Decompose to the actual individual gates
        circuit = ansatz.decompose(reps=3)
        # Set some arbitrary parameters
        circuit.assign_parameters([0.8] * len(circuit.parameters), inplace=True)

    print(_cut_comparison(circuit, 4, [2], 2))


def compare_circuit_cuts(circuits, max_cuts, num_subcircuits, max_subcircuit_width, file_name=None):
    if file_name is None:
        raise ValueError("Please provide a file name to save the results")

    results = []

    # Copmpare the number of cuts in CutQC and against gate and wire cutting
    for circuit in circuits:
        results.append(_cut_comparison(circuit, max_cuts, num_subcircuits, max_subcircuit_width))

    # Save all results to a CSV
    import csv
    with open(file_name, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["CutQC", "Gate and Wire"])
        for result in results:
            writer.writerow(result)


if __name__ == "__main__":
    _example_use()
from qiskit.circuit import QuantumCircuit, ParameterVector
from circuit_knitting.cutting.gate_and_wire_cutting.frontend import cut_wires_and_gates_to_subcircuits
from circuit_knitting.cutting.gate_and_wire_cutting.frontend import execute_simulation
from circuit_knitting.cutting.cutting_reconstruction import reconstruct_expectation_values
from circuit_knitting.cutting.gate_and_wire_cutting.frontend import exact_observables
from circuit_knitting.cutting.gate_and_wire_cutting.frontend import compare_results
import time
from datetime import datetime
import csv

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import CDKMRippleCarryAdder, QFT
from qiskit.circuit.random import random_circuit

import numpy as np
import networkx as nx
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
from qiskit_optimization.applications import Maxcut, Tsp

def benchmark(circuit: QuantumCircuit, observables: list[str], algorithm="N/A", max_subcircuit_width=2, max_cuts=4, num_subcircuits=[2], get_errors=False):
    not_cut = False
    print("Cutting the circuits...")
    # Cut the circuits
    opt_s = time.time()
    
    try:
        subcircuits, subobservables, wire_cuts, gate_cuts = cut_wires_and_gates_to_subcircuits(
            circuit=circuit,
            observables=[str(observable) for observable in observables],
            method='automatic',
            max_subcircuit_width=max_subcircuit_width,
            max_cuts=max_cuts,
            num_subcircuits=num_subcircuits,
            model='gurobi',
        )
    except:
        not_cut = True
    opt_time = time.time() - opt_s
    
    if not not_cut:
        print("Executing the subcircuits and reconstructing...")
        # Execute the subcircuits
        recon_s = time.time()
        quasi_dists, coefficients = execute_simulation(subcircuits, subobservables)
        simulation_expvals = reconstruct_expectation_values(quasi_dists, coefficients, subobservables)
        recon_time = time.time() - recon_s

        if get_errors:
            ideal_expvals = exact_observables(circuit, observables)
            errors, rel_errors = compare_results(simulation_expvals, ideal_expvals)
        else:
            errors=None
            rel_errors=None

        subcircuit_lengths = [s.depth() for s in subcircuits.values()]

        data_dict = {"Algorithm": algorithm,
                      "Original Circuit Depth": circuit.depth(), 
                      "Number of Subcircuits": len(subcircuits), 
                      "Number of Wire Cuts": wire_cuts, 
                      "Number of Gate Cuts": gate_cuts,
                      "Subcircuits' Depths": subcircuit_lengths, 
                      "Time to Find Optimal Solution": opt_time,
                      "Time to Reconstruct": recon_time
                     }
    else:
        data_dict = {"Algorithm": algorithm,
                      "Original Circuit Depth": circuit.depth(), 
                      "Number of Subcircuits": not_cut, 
                      "Number of Wire Cuts": not_cut, 
                      "Number of Gate Cuts": not_cut,
                      "Subcircuits' Depths": not_cut, 
                      "Time to Find Optimal Solution": not_cut,
                      "Time to Reconstruct": not_cut
                     }
    
    return data_dict

def ripple_carry_adder(a, b):
    n_qubits = max(a.bit_length(), b.bit_length())
    operand1 = QuantumRegister(n_qubits, 'o1')
    operand2 = QuantumRegister(n_qubits, 'o2')
    anc = QuantumRegister(2, 'a')
    qc = QuantumCircuit(operand1, operand2, anc)
    for i in range(n_qubits):
        if (a & (1 << i)):
            qc.x(operand1[i])
        if (b & (1 << i)):
            qc.x(operand2[i])
    adder = CDKMRippleCarryAdder(n_qubits, 'full', 'Full Adder')
    qc.append(adder, [anc[0]] + operand1[:] + operand2[:] + [anc[1]])

    return qc.decompose(reps=4)

def qft(num_qubits):
    qc = QuantumCircuit(num_qubits)
    qft = QFT(num_qubits=num_qubits, do_swaps=False)
    qc.compose(qft, inplace=True)
    qc.save_statevector()
    qc = qc.decompose(reps=7)
    return qc

def supremacy(num_qubits):
    return random_circuit(num_qubits, 2, measure=False).decompose(reps=3)

def qaoa(num_qubits):
    assert num_qubits%2 == 0, "Number of qubits/nodes must be even for 3-reg graph."
    
    G = nx.random_regular_graph(3, num_qubits)

    for (u,v) in G.edges():
        G[u][v]['weight'] = 1.0

    pos = nx.spring_layout(G)

    colors = ["r" for node in G.nodes()]
    pos = nx.spring_layout(G)

    # Computing the weight matrix from the random graph
    w = np.zeros([num_qubits, num_qubits])
    for i in range(num_qubits):
        for j in range(num_qubits):
            temp = G.get_edge_data(i, j, default=0)
            if temp != 0:
                w[i, j] = temp["weight"]

    max_cut = Maxcut(w)
    qp = max_cut.to_quadratic_program()
    qubitOp, offset = qp.to_ising()
    
    # Create our ansatz with all parameters assigned for a single cut and execution
    ansatz = QAOAAnsatz(qubitOp, reps=1, flatten=True)
    ansatz = ansatz.decompose(reps=2)

    # Get the number of parameters we need
    n_param = 0
    for d in ansatz.data:
        if len(d.operation.params) > 0 and ('γ' in str(d.operation.params[0]) or 'β' in str(d.operation.params[0])):
            n_param += 1

    beta = ParameterVector("β", n_param)

    # Fix the parameter vectors so we do not have duplicates of parameters
    k = 0
    for d in ansatz.data:
        if len(d.operation.params) > 0 and ('γ' in str(d.operation.params[0]) or 'β' in str(d.operation.params[0])): 
            if 'β' in str(d.operation.params[0]):
                d.operation.params = [np.pi, 0, 0]#[2*beta[k]]
            else:
                d.operation.params = [np.pi, 0, 0]#[beta[k]]
            k+=1

    # Copy to new ansatz so the ansatz.data.parameters only has one parameter vector
    new_ansatz = QuantumCircuit(num_qubits)

    for d in ansatz.data:
        new_ansatz.data.append(d)
    
    return new_ansatz, qubitOp.paulis

def benchmark_run():
    print("Running Ripple Carry Adder Benchmark")
    qc0 = ripple_carry_adder(1,3)
    data_0 = benchmark(qc0, ['Z'*qc0.num_qubits], algorithm="Adder", max_subcircuit_width=3, max_cuts=100, num_subcircuits=[2])

    print("Running QFT Benchmark")
    qc1 = qft(4)
    data_1 = benchmark(qc1, ['Z'*qc1.num_qubits], algorithm="QFT", max_subcircuit_width=2, max_cuts=100, num_subcircuits=[2])
    
    print("Running Supremacy Benchmark")
    qc2 = supremacy(4)
    data_2 = benchmark(qc2, ['Z'*qc2.num_qubits], algorithm="Supremacy", max_subcircuit_width=2, max_cuts=100, num_subcircuits=[2])
    
    print("Running QAOA Benchmark")
    qc3, observables = qaoa(4)
    data_3 = benchmark(qc3, observables, algorithm="QAOA", max_subcircuit_width=2, max_cuts=4, num_subcircuits=[2])
    
    print("Finished, Saving...")
    data_dict = [data_0, data_1, data_2, data_3]
    
    filename = "benchmark_data_" + str(datetime.now().strftime("%m_%d_%Y_%H:%M:%S")) + ".csv"
    
    fields = ["Algorithm", "Number of Subcircuits", "Number of Wire Cuts", "Number of Gate Cuts",
              "Original Circuit Depth", "Subcircuits' Depths", "Time to Find Optimal Solution",
              "Time to Reconstruct"]
    
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data_dict)
        
    print("Completed.")

benchmark_run()
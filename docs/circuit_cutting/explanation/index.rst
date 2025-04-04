.. _circuit cutting explanation:

###################################################
Explanatory material for the circuit cutting module
###################################################

Overview of circuit cutting
---------------------------
Circuit cutting is a technique to increase the size of circuits we can run on quantum hardware at the cost of an additional sampling overhead. A larger quantum circuit can be decomposed by cutting its gates and/or wires, resulting in smaller circuits which can be executed within the constraints of available quantum hardware. The results of these smaller circuits are combined to reconstruct the outcome of the original problem. Circuit cutting can also be used to engineer gates between distant qubits which would otherwise require a large swap overhead.

Circuit cutting as a quasiprobability decomposition (QPD)
---------------------------------------------------------
Quasiprobability decomposition is a technique which can be used to simulate quantum circuit executions that go beyond the actual capabilities of current quantum hardware while using that same hardware.  It forms the basis of many error mitigation techniques, which allow simulating a noise-free quantum computer using a noisy one.  Circuit cutting techniques, which allow simulating a quantum circuit using fewer qubits than would otherwise be necessary, can also be phrased in terms of a quasiprobability decomposition.  No matter the goal, the cost of the quasiprobability decomposition is an exponential overhead in the number of circuit executions which must be performed.  In certain cases, this tradeoff is worth it, because it can allow the estimation of quantities that would otherwise be impossible on today's hardware.

To perform circuit cutting, one must partition (“cut”) the graph representing a quantum circuit into smaller pieces, which are then executed on available hardware.  The results of the original circuit must then be reconstructed during post-processing, resulting in the desired quantity (e.g., the expectation value of a joint observable).

There are two types of cuts: gate cuts and wire cuts.  Gate cuts, also known as "space-like" cuts, exist when the cut goes through a gate operating on two (or more) qubits.  Wire cuts, also known as "time-like" cuts, are direct cuts through a qubit wire, essentially a single-qubit identity gate that has been cut into two pieces.  A wire cut is simulated by introducing a new qubit into the circuit and moving remaining operations after the cut identity gate to the new qubit.

There are three settings to consider for circuit cutting.  The first is where only local operations (LO) [i.e., local *quantum* operations] are available.  The other settings introduce classical communication between the circuit executions, which is known in the quantum information literature as LOCC, for `local operations and classical communication <https://en.wikipedia.org/wiki/LOCC>`__.  The LOCC can be either near-time, one-directional communication between the circuit executions (the second setting), or real-time, bi-directional communication (the third setting).

As mentioned above, the cost of any simulation based on quasiprobability distribution is an exponential sampling overhead. The overhead of a cut gate depends on which gate is cut; see the final appendix of [`1 <https://arxiv.org/abs/2205.00016>`__] for details.  Here, we will focus on the CNOT gate.  If no real-time classical communication is available between qubits of the cut gate or wire, cut CNOT gates incur a sampling overhead of O(:math:`9^n`), and wire cuts incur a sampling overhead of O(:math:`16^n`), where :math:`n` is the total number of cuts. If real-time communication is available (i.e., if the hardware supports “dynamic circuits”), the sampling overhead for both CNOT gate and wire cuts may be reduced to O(:math:`4^n`) [`1 <https://arxiv.org/abs/2205.00016>`__,\ `3 <https://arxiv.org/abs/2302.03366>`__]; however, support for circuit cutting with classical communication (LOCC) is not yet supported in CKT.

For more detailed information on the quasiprobability decomposition technique, refer to the paper, Error mitigation for short-depth quantum circuits [`4 <https://arxiv.org/abs/1612.02058>`__].

Key terms
-----------------
* subcircuits: The set of circuits resulting from cutting gates in a :class:`QuantumCircuit` and then separating the disconnected qubit subsets into smaller circuits. These circuits contain :class:`SingleQubitQPDGate`\ s and will be used to instantiate each unique subexperiment.

* subexperiments: A term used to describe the unique circuit samples associated with a subcircuit. These circuits have had their :class:`BaseQPDGate`\ s decomposed into local Qiskit gates and measurements. Subexperiments are the circuits sent to the backend for execution.

* decompose: We try to honor the Qiskit notion of "decompose" in the documentation and API, which loosely means transforming a gate into a less-abstracted representation. *Occasionally*, we may use the term "decompose" to refer to the act of inserting :class:`BaseQPDGate` instances into quantum circuits as "decomposing" a gate or wire; however, we try to use terms like "partition" and "cut" when referring to this to avoid ambiguity with Qiskit language.

Current limitations
-------------------
* QPD-based wire cutting will be available no sooner than CKT v0.3.0. The `cutqc <../cutqc/index.rst>`__ package may be used for wire cutting in the meantime.
* ``PauliList`` is the only supported observable format until no sooner than CKT v0.3.0.

References
----------

This module is based on the theory described in the
following papers:

[1] Christophe Piveteau, David Sutter, *Circuit knitting with classical communication*,
https://arxiv.org/abs/2205.00016

[2] Kosuke Mitarai, Keisuke Fujii, *Constructing a virtual two-qubit gate by sampling
single-qubit operations*,
https://arxiv.org/abs/1909.07534

[3] Lukas Brenner, Christophe Piveteau, David Sutter, *Optimal wire cutting with
classical communication*,
https://arxiv.org/abs/2302.03366

[4] K. Temme, S. Bravyi, and J. M. Gambetta, *Error mitigation for short-depth quantum circuits*,
https://arxiv.org/abs/1612.02058

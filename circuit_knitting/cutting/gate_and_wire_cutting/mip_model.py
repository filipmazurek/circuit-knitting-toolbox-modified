# This code is a Qiskit project.

# (C) Copyright IBM 2022.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""File containing the tools to find and manage the cuts."""

from __future__ import annotations

from typing import Sequence, Any

import re


class IBMMIPModel(object):
    """
    Class to contain the model that manages the cut MIP.

    This is the modified version of the MIPModel class. This class is used to find both wire and gate cuts
    at the same time

    This class represents circuit cutting as a Mixed Integer Programming (MIP) problem
    that can then be solved (provably) optimally using a MIP solver. This is integrated
    with CPLEX, a fast commercial solver sold by IBM. There are free and open source MIP
    solvers, but they come with substantial slowdowns (often many orders of magnitude).
    By representing the original circuit as a Directed Acyclic Graph (DAG), this class
    can find the optimal wire cuts in the circuit.
    """

    def __init__(
        self,
        n_vertices: int,
        edges: Sequence[tuple[int]],
        vertex_ids: dict[str, int],
        id_vertices: dict[int, str],
        num_subcircuit: int,
        max_subcircuit_width: int,
        num_qubits: int,
        max_cuts: int,
        double_wire_cost: bool = True
    ):
        """
        Initialize member variables.

        Args:
            n_vertices: The number of vertices in the circuit DAG
            edges: The list of edges of the circuit DAG
            n_edges: The number of edges
            vertex_ids: Dictionary mapping vertices (i.e. two qubit gates) to the vertex
                id (i.e. a number)
            id_vertices: The inverse dictionary of vertex_ids, which has keys of vertex ids
                and values of the vertices
            num_subcircuit: The number of subcircuits
            max_subcircuit_width: Maximum number of qubits per subcircuit
            max_subcircuit_cuts: Maximum number of cuts in each subcircuit
            max_subcircuit_size: Maximum number of gates in a subcircuit
            num_qubits: The number of qubits in the circuit
            max_cuts: The maximum total number of cuts
            double_wire_cost: Whether to double the cost of a wire cut. Doubling means using the no classical
                communication method
        Returns:
            None
        """
        self.check_graph(n_vertices, edges)
        self.n_vertices = n_vertices
        self.edges = edges
        self.n_edges = len(edges)
        self.vertex_ids = vertex_ids
        self.id_vertices = id_vertices
        self.num_subcircuit = num_subcircuit
        self.max_subcircuit_width = max_subcircuit_width
        self.num_qubits = num_qubits
        self.max_cuts = max_cuts
        self.double_wire_cost = double_wire_cost

        self.subcircuit_counter: dict[int, dict[str, Any]] = {}

        # Create a dictionary that will give information about the qubit based on the id
        self.id_qubit = {}
        for i in range(n_vertices):
            q_name = self.id_vertices[i].split("]")[0] + "]"
            self.id_qubit[i] = q_name

        # Create a list of qubit names
        self.qubit_names = list(set(self.id_qubit.values()))

        assert len(self.qubit_names) == self.num_qubits

        """
        Count the number of input qubits directly connected to each node
        """
        self.vertex_weight = {}
        """
        Note that these vertex_ids are what was previously called id_node_names, e.g. my_id_node_names {0: 'q0[0]0', 1: 'q0[1]0', 2: 'q0[1]1', 3: 'q0[2]0'}
        Every node is going to have either 1 or 0 inputs
        """
        # Each node is a single qubit involved in a *two* qubit gate. Therefore each one has 1 or 0 inputs
        for node in self.vertex_ids:
            num_in_qubits = 0
            if int(node.split("]")[1]) == 0:
                num_in_qubits += 1
            self.vertex_weight[node] = num_in_qubits

        # Extract qubit and gate information
        self.edge_info = {}
        for edge in self.edges:
            u, v = edge
            u_name, v_name = self.id_vertices[u], self.id_vertices[v]

            # Vertex information is generated as 'q[1]0'. Edges are saved as tuples of the vertices
            u_match = re.match(r'.*\[(\d+)\](\d+)', u_name)
            v_match = re.match(r'.*\[(\d+)\](\d+)', v_name)

            # Match the qubit and gate number
            u_qubit, u_gate = map(int, u_match.groups())
            v_qubit, v_gate = map(int, v_match.groups())

            if u_qubit == v_qubit:
                edge_type = 'wire'
            else:
                edge_type = 'gate'

            # Dictionary of edge information. Used for constraint generation
            self.edge_info[edge] = {'u_qubit': u_qubit, 'u_gate': u_gate, 'v_qubit': v_qubit,
                                    'v_gate': v_gate,
                                    'edge_type': edge_type}

        try:
            from docplex.mp.model import Model
        except ModuleNotFoundError as ex:  # pragma: no cover
            raise ModuleNotFoundError(
                "DOcplex is not installed.  For automatic cut finding to work, both "
                "DOcplex and cplex must be available."
            ) from ex

        self.model = Model("docplex_cutter")
        self.model.log_output = False

        self._add_variables()
        self._add_constraints()

    def _add_variables(self) -> None:
        """Add the necessary variables to the CPLEX model."""

        # Use an indicator variable to check if there is a cut on that particular qubit for that subcircuit
        self.qubit_in_subcircuit = [[] for _ in range(self.num_subcircuit)]
        for subcircuit in range(self.num_subcircuit):
            subcircuit_y = []
            for qubit in range(self.num_qubits):
                var_name = f"bin_wire_cut_on_qubit_sc{subcircuit}_q{qubit}"
                loc_var = self.model.binary_var(name=var_name)
                subcircuit_y.append(loc_var)
            self.qubit_in_subcircuit[subcircuit] = subcircuit_y

        # Helper variable for the above indicator variable. It counts the number of vertices of a particular qubit
        #   within a subcircuit
        self.num_vertices_in_subcircuit_per_qubit = [[] for _ in range(self.num_subcircuit)]
        for subcircuit in range(self.num_subcircuit):
            subcircuit_y = []
            for qubit in range(self.num_qubits):
                var_name = f"num_vertices_per_qubit_sc{subcircuit}_q{qubit}"
                loc_var = self.model.integer_var(name=var_name)
                subcircuit_y.append(loc_var)
            self.num_vertices_in_subcircuit_per_qubit[subcircuit] = subcircuit_y

        # The sum of all edges that are cut that are of a particular type. Assuming edges indices line up with the
        #   edge list
        self.num_wire_cuts = self.model.integer_var(lb=0, ub=self.max_cuts + 0.1, name="total_num_wire_cuts")
        self.num_gate_cuts = self.model.integer_var(lb=0, ub=self.max_cuts + 0.1, name="total_num_gate_cuts")
        # Variable to keep track of the higher cost of wire cuts. Doubles the cost of a wire cut. For when using
        #   SWAP gates instead of classical communication
        self.weighted_total_cuts = self.model.integer_var(lb=0, ub=2 * self.max_cuts + 0.1, name="weighted_total_cuts")

        """
        Indicate if a vertex is in some subcircuit
        """
        self.vertex_var = []
        for i in range(self.num_subcircuit):
            subcircuit_y = []
            for j in range(self.n_vertices):
                varName = "bin_sc_" + str(i) + "_vx_" + str(j)
                loc_var = self.model.binary_var(name=varName)
                subcircuit_y.append(loc_var)
            self.vertex_var.append(subcircuit_y)

        """
        Indicate if an edge has one and only one vertex in some subcircuit
        """
        self.edge_var = []
        for i in range(self.num_subcircuit):
            subcircuit_x = []
            for j in range(self.n_edges):
                varName = "bin_sc_" + str(i) + "_edg_" + str(j)
                loc_var = self.model.binary_var(name=varName)
                subcircuit_x.append(loc_var)
            self.edge_var.append(subcircuit_x)

        """
        Total number of cuts
        add 0.1 for numerical stability
        """
        self.num_cuts = self.model.integer_var(
            lb=0, ub=self.max_cuts + 0.1, name="num_cuts"
        )

        for subcircuit in range(self.num_subcircuit):
            self.subcircuit_counter[subcircuit] = {}

            self.subcircuit_counter[subcircuit]["d"] = self.model.integer_var(
                lb=0.1, ub=self.max_subcircuit_width, name="d_%d" % subcircuit
            )

            if subcircuit > 0:
                lb = 0
                ub = self.num_qubits + 2 * self.max_cuts + 1
                self.subcircuit_counter[subcircuit][
                    "build_cost_exponent"
                ] = self.model.integer_var(
                    lb=lb, ub=ub, name="build_cost_exponent_%d" % subcircuit
                )

    def _add_constraints(self) -> None:
        """Add constraints and objectives to MIP model."""

        """
        each vertex in exactly one subcircuit
        """
        for v in range(self.n_vertices):
            ctName = "cons_vertex_" + str(v)
            self.model.add_constraint(
                self.model.sum(
                    self.vertex_var[i][v] for i in range(self.num_subcircuit)
                )
                == 1,
                ctname=ctName,
            )

        """
        edge_var=1 indicates one and only one vertex of an edge is in subcircuit
        edge_var[subcircuit][edge] = vertex_var[subcircuit][u] XOR vertex_var[subcircuit][v]
        """
        for i in range(self.num_subcircuit):
            for e in range(self.n_edges):
                if len(self.edges[e]) != 2:
                    raise ValueError(
                        "Edges should be length 2 sequences: {self.edges[e]}"
                    )
                u = self.edges[e][0]
                v = self.edges[e][-1]
                u_vertex_var = self.vertex_var[i][u]
                v_vertex_var = self.vertex_var[i][v]
                ctName = "cons_edge_" + str(e)
                self.model.add_constraint(
                    self.edge_var[i][e] - u_vertex_var - v_vertex_var <= 0,
                    ctname=ctName + "_1",
                )
                self.model.add_constraint(
                    self.edge_var[i][e] - u_vertex_var + v_vertex_var >= 0,
                    ctname=ctName + "_2",
                )
                self.model.add_constraint(
                    self.edge_var[i][e] - v_vertex_var + u_vertex_var >= 0,
                    ctname=ctName + "_3",
                )
                self.model.add_constraint(
                    self.edge_var[i][e] + u_vertex_var + v_vertex_var <= 2,
                    ctname=ctName + "_4",
                )

        """
        Symmetry-breaking constraints
        Force small-numbered vertices into small-numbered subcircuits:
            v0: in subcircuit 0
            v1: in subcircuit_0 or subcircuit_1
            v2: in subcircuit_0 or subcircuit_1 or subcircuit_2
            ...
        """
        for vertex in range(self.num_subcircuit):
            ctName = "cons_symm_" + str(vertex)
            self.model.add_constraint(
                self.model.sum(
                    self.vertex_var[subcircuit][vertex]
                    for subcircuit in range(vertex + 1)
                )
                == 1,
                ctname=ctName,
            )

        """
        Compute number of cuts
        """
        # Adds all the edges with a '1' value (a cut) for every subcircuit. Because a cut edge is counted twice (once
        #   in every subcircuit it is in), we divide by 2 to get the total number of cuts
        self.model.add_constraint(
            self.num_cuts
            == self.model.sum(
                [
                    self.edge_var[subcircuit][i]
                    for i in range(self.n_edges)
                    for subcircuit in range(self.num_subcircuit)
                ]
            )
            / 2
        )

        """
        Compute the number of wire cuts and gate cuts
        """
        # Wire cuts is the sum of edges cut that are actually wire edges
        self.model.add_constraint(
            self.num_wire_cuts
            == self.model.sum(
                [
                    self.edge_var[subcircuit][i]
                    for i in range(self.n_edges)
                    for subcircuit in range(self.num_subcircuit)
                    if self.edge_info[self.edges[i]]['edge_type'] == 'wire'
                ]
            )
            / 2
        )

        # Gate cuts is the sum of edges cut that are actually gate edges
        self.model.add_constraint(
            self.num_gate_cuts
            == self.model.sum(
                [
                    self.edge_var[subcircuit][i]
                    for i in range(self.n_edges)
                    for subcircuit in range(self.num_subcircuit)
                    if self.edge_info[self.edges[i]]['edge_type'] == 'gate'
                ]
            )
            / 2
        )

        # Sanity check: the number of wire cuts and gate cuts should add up to the total number of cuts
        self.model.add_constraint(
            self.num_cuts == self.num_wire_cuts + self.num_gate_cuts
        )

        # Number of cuts taking into account the cost of a wire cut SWAP
        self.model.add_constraint(
            self.weighted_total_cuts == 2 * self.num_wire_cuts + self.num_gate_cuts
        )

        num_effective_qubits = []
        for subcircuit in range(self.num_subcircuit):
            """
            Compute number of different types of qubit in a subcircuit
            """

            for i in range(self.n_edges):
                if len(self.edges[i]) != 2:
                    raise ValueError(
                        "Edges should be length 2 sequences: {self.edges[i]}"
                    )

            # Counts the number of vertices in the subcircuit *per qubit*
            for q in range(self.num_qubits):
                self.model.add_constraint(
                    self.num_vertices_in_subcircuit_per_qubit[subcircuit][q]
                    ==
                    # The number of vertices in the subcircuit that correspond to the qubit
                    self.model.sum(
                        self.vertex_var[subcircuit][v]
                        for v in range(self.n_vertices)
                        if self.id_qubit[v] == self.qubit_names[q]
                    )
                )

            # If the number of the number of vertices is at least 1, then that qubit is in the subcircuit
            for j in range(self.num_qubits):
                self.model.add_constraint(
                    (self.num_vertices_in_subcircuit_per_qubit[subcircuit][j] >= 1)
                    == self.qubit_in_subcircuit[subcircuit][j],
                    ctname="qubit_in_subcircuit_sc%d_q%d" % (subcircuit, j),
                )

            # d is the total number of qubits in a subcircuit. It can't be more than the max subcircuit width
            self.model.add_constraint(
                self.subcircuit_counter[subcircuit]["d"]
                - self.model.sum(
                    self.qubit_in_subcircuit[subcircuit][qubit]
                    for qubit in range(self.num_qubits)
                )
                == 0,
                ctname="cons_subcircuit_width_%d" % subcircuit,
            )

        # Constraint to check if wire cuts count as 2 cuts, for when using a SWAP gate for the move operation
        if self.double_wire_cost:
            self.model.set_objective('min', self.weighted_total_cuts)
        else:
            self.model.set_objective("min", self.num_cuts)


    def check_graph(self, n_vertices: int, edges: Sequence[tuple[int]]) -> None:
        """
        Ensure circuit DAG is viable.

        This means that there are no oversized edges, that all edges are from viable nodes,
        and that the graph is otherwise a valid graph.

        Args:
            n_vertices: The number of vertices
            edges: The edge list

        Returns:
            None

        Raises:
            ValueError: The graph is invalid
        """
        # 1. edges must include all vertices
        # 2. all u,v must be ordered and smaller than n_vertices
        vertices = set([i for (i, _) in edges])  # type: ignore
        vertices |= set([i for (_, i) in edges])  # type: ignore
        assert vertices == set(range(n_vertices))
        for edge in edges:
            if len(edge) != 2:
                raise ValueError("Edges should be length 2 sequences: {edge}")
            u = edge[0]
            v = edge[-1]
            if u > v:
                raise ValueError(f"Edge u ({u}) cannot be greater than edge v ({v})")
            if u > n_vertices:
                raise ValueError(
                    f"Edge u ({u}) cannot be greater than number of vertices ({n_vertices})"
                )

    def solve(self, min_postprocessing_cost: float) -> bool:
        """
        Solve the MIP model.

        Args:
            min_post_processing_cost: The predicted minimum post-processing cost,
                often is inf

        Returns:
            Flag denoting if the model found a solution
        """
        from docplex.mp.utils import DOcplexException

        # print(
        #     "Exporting as a LP file to let you check the model that will be solved : ",
        #     min_postprocessing_cost,
        #     str(type(min_postprocessing_cost)),
        # )
        try:
            self.model.export_as_lp(path="./docplex_cutter.lp")
        except RuntimeError:
            print(
                "The LP file export has failed. This is known to happen sometimes "
                "when cplex is not installed. Attempting to continue anyway."
            )
        try:
            self.model.set_time_limit(300)
            if min_postprocessing_cost != float("inf"):
                self.model.parameters.mip.tolerances.uppercutoff(
                    min_postprocessing_cost
                )
            self.model.solve(log_output=False)

        except DOcplexException as e:
            print("Caught: " + e.message)
            raise e

        # from docplex.mp.conflict_refiner import ConflictRefiner
        # crr = ConflictRefiner().refine_conflict(self.model, display=True)
        # crr.display()
        from docplex.mp.relaxer import Relaxer
        rx = Relaxer()
        rs = rx.relax(self.model)
        # Prints all optimization values
        # rx.print_information()
        # rs.display()

        if self.model._has_solution:
            my_solve_details = self.model.solve_details
            self.subcircuits = []
            self.optimal = my_solve_details.status == "optimal"
            self.runtime = my_solve_details.time
            self.node_count = my_solve_details.nb_nodes_processed
            self.mip_gap = my_solve_details.mip_relative_gap
            self.objective = self.model.objective_value

            for i in range(self.num_subcircuit):
                subcircuit = []
                for j in range(self.n_vertices):
                    if abs(self.vertex_var[i][j].solution_value) > 1e-4:
                        subcircuit.append(self.id_vertices[j])
                self.subcircuits.append(subcircuit)
            assert (
                sum([len(subcircuit) for subcircuit in self.subcircuits])
                == self.n_vertices
            )

            cut_edges_idx = []
            cut_edges = []
            for i in range(self.num_subcircuit):
                for j in range(self.n_edges):
                    if (
                        abs(self.edge_var[i][j].solution_value) > 1e-4
                        and j not in cut_edges_idx
                    ):
                        cut_edges_idx.append(j)
                        if len(self.edges[j]) != 2:
                            raise ValueError("Edges should be length-2 sequences.")
                        u = self.edges[j][0]
                        v = self.edges[j][-1]
                        cut_edges.append((self.id_vertices[u], self.id_vertices[v]))
                        # Save the weighted cost for easier use
                        self.weighted_cost = self.weighted_total_cuts.solution_value
            self.cut_edges = cut_edges
            return True
        else:
            return False


class GurobiMIPModel(object):
    """
    Class to contain the model that manages the cut MIP.

    This is the modified version of the MIPModel class. This class is used to find both wire and gate cuts
    at the same time

    This class is modified from the above IBM solver to use the Gurobi Solver
    """

    def __init__(
        self,
        n_vertices: int,
        edges: Sequence[tuple[int]],
        vertex_ids: dict[str, int],
        id_vertices: dict[int, str],
        num_subcircuit: int,
        max_subcircuit_width: int,
        num_qubits: int,
        max_cuts: int,
        double_wire_cost: bool = True
    ):
        """
        Initialize member variables.

        Args:
            n_vertices: The number of vertices in the circuit DAG
            edges: The list of edges of the circuit DAG
            n_edges: The number of edges
            vertex_ids: Dictionary mapping vertices (i.e. two qubit gates) to the vertex
                id (i.e. a number)
            id_vertices: The inverse dictionary of vertex_ids, which has keys of vertex ids
                and values of the vertices
            num_subcircuit: The number of subcircuits
            max_subcircuit_width: Maximum number of qubits per subcircuit
            max_subcircuit_cuts: Maximum number of cuts in each subcircuit
            max_subcircuit_size: Maximum number of gates in a subcircuit
            num_qubits: The number of qubits in the circuit
            max_cuts: The maximum total number of cuts
            double_wire_cost: Whether to double the cost of a wire cut. Doubling means using the no classical
                communication method
        Returns:
            None
        """
        self.check_graph(n_vertices, edges)
        self.n_vertices = n_vertices
        self.edges = edges
        self.n_edges = len(edges)
        self.vertex_ids = vertex_ids
        self.id_vertices = id_vertices
        self.num_subcircuit = num_subcircuit
        self.max_subcircuit_width = max_subcircuit_width
        self.num_qubits = num_qubits
        self.max_cuts = max_cuts
        self.double_wire_cost = double_wire_cost

        self.subcircuit_counter: dict[int, dict[str, Any]] = {}

        # Create a dictionary that will give information about the qubit based on the id
        self.id_qubit = {}
        for i in range(n_vertices):
            q_name = self.id_vertices[i].split("]")[0] + "]"
            self.id_qubit[i] = q_name

        # Create a list of qubit names
        self.qubit_names = list(set(self.id_qubit.values()))

        assert len(self.qubit_names) == self.num_qubits

        """
        Count the number of input qubits directly connected to each node
        """
        self.vertex_weight = {}
        """
        Note that these vertex_ids are what was previously called id_node_names, e.g. my_id_node_names {0: 'q0[0]0', 1: 'q0[1]0', 2: 'q0[1]1', 3: 'q0[2]0'}
        Every node is going to have either 1 or 0 inputs
        """
        # Each node is a single qubit involved in a *two* qubit gate. Therefore each one has 1 or 0 inputs
        for node in self.vertex_ids:
            num_in_qubits = 0
            if int(node.split("]")[1]) == 0:
                num_in_qubits += 1
            self.vertex_weight[node] = num_in_qubits

        # Extract qubit and gate information
        self.edge_info = {}
        for edge in self.edges:
            u, v = edge
            u_name, v_name = self.id_vertices[u], self.id_vertices[v]

            # Vertex information is generated as 'q[1]0'. Edges are saved as tuples of the vertices
            u_match = re.match(r'.*\[(\d+)\](\d+)', u_name)
            v_match = re.match(r'.*\[(\d+)\](\d+)', v_name)

            # Match the qubit and gate number
            u_qubit, u_gate = map(int, u_match.groups())
            v_qubit, v_gate = map(int, v_match.groups())

            if u_qubit == v_qubit:
                edge_type = 'wire'
            else:
                edge_type = 'gate'

            # Dictionary of edge information. Used for constraint generation
            self.edge_info[edge] = {'u_qubit': u_qubit, 'u_gate': u_gate, 'v_qubit': v_qubit,
                                    'v_gate': v_gate,
                                    'edge_type': edge_type}

        try:
            from gurobipy import Model
        except ModuleNotFoundError as ex:  # pragma: no cover
            raise ModuleNotFoundError(
                "gurobipy is not installed. For automatic cut finding to work, gurobipy must be installed"
            ) from ex

        self.model = Model("gurobi_cutter")
        self._add_variables()
        self._add_constraints()

    def _add_variables(self) -> None:
        """Add the necessary variables to the CPLEX model."""
        from gurobipy import GRB

        # Use an indicator variable to check if there is a cut on that particular qubit for that subcircuit
        self.qubit_in_subcircuit = [[] for _ in range(self.num_subcircuit)]
        for subcircuit in range(self.num_subcircuit):
            subcircuit_y = []
            for qubit in range(self.num_qubits):
                var_name = f"bin_wire_cut_on_qubit_sc{subcircuit}_q{qubit}"
                loc_var = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=var_name)
                subcircuit_y.append(loc_var)
            self.qubit_in_subcircuit[subcircuit] = subcircuit_y


        # Helper variable for the above indicator variable. It counts the number of vertices of a particular qubit
        #   within a subcircuit
        self.num_vertices_in_subcircuit_per_qubit = [[] for _ in range(self.num_subcircuit)]
        for subcircuit in range(self.num_subcircuit):
            subcircuit_y = []
            for qubit in range(self.num_qubits):
                var_name = f"num_vertices_per_qubit_sc{subcircuit}_q{qubit}"
                loc_var = self.model.addVar(vtype=GRB.INTEGER, lb=0.0, name=var_name)
                subcircuit_y.append(loc_var)
            self.num_vertices_in_subcircuit_per_qubit[subcircuit] = subcircuit_y

        # The sum of all edges that are cut that are of a particular type. Assuming edges indices line up with the
        #   edge list
        self.num_wire_cuts = self.model.addVar(vtype=GRB.INTEGER, lb=0, ub=self.max_cuts + 0.1, name="total_num_wire_cuts")
        self.num_gate_cuts = self.model.addVar(vtype=GRB.INTEGER, lb=0, ub=self.max_cuts + 0.1, name="total_num_gate_cuts")
        # Variable to keep track of the higher cost of wire cuts. Doubles the cost of a wire cut. For when using
        #   SWAP gates instead of classical communication
        self.weighted_total_cuts = self.model.addVar(vtype=GRB.INTEGER, lb=0, ub=2 * self.max_cuts + 0.1, name="weighted_total_cuts")

        """
        Indicate if a vertex is in some subcircuit
        """
        self.vertex_var = []
        for i in range(self.num_subcircuit):
            subcircuit_y = []
            for j in range(self.n_vertices):
                varName = "bin_sc_" + str(i) + "_vx_" + str(j)
                loc_var = self.model.addVar(vtype=GRB.BINARY, name=varName)
                subcircuit_y.append(loc_var)
            self.vertex_var.append(subcircuit_y)

        """
        Indicate if an edge has one and only one vertex in some subcircuit
        """
        self.edge_var = []
        for i in range(self.num_subcircuit):
            subcircuit_x = []
            for j in range(self.n_edges):
                varName = "bin_sc_" + str(i) + "_edg_" + str(j)
                loc_var = self.model.addVar(vtype=GRB.BINARY, name=varName)
                subcircuit_x.append(loc_var)
            self.edge_var.append(subcircuit_x)

        """
        Total number of cuts
        add 0.1 for numerical stability
        """
        self.num_cuts = self.model.addVar(
            vtype=GRB.INTEGER, lb=0, ub=self.max_cuts + 0.1, name="num_cuts"
        )

        for subcircuit in range(self.num_subcircuit):
            self.subcircuit_counter[subcircuit] = {}

            self.subcircuit_counter[subcircuit]["d"] = self.model.addVar(
                vtype=GRB.INTEGER, lb=0.1, ub=self.max_subcircuit_width, name="d_%d" % subcircuit
            )

            if subcircuit > 0:
                lb = 0
                ub = self.num_qubits + 2 * self.max_cuts + 1
                self.subcircuit_counter[subcircuit][
                    "build_cost_exponent"
                ] = self.model.addVar(
                    vtype=GRB.INTEGER, lb=lb, ub=ub, name="build_cost_exponent_%d" % subcircuit
                )

    def _add_constraints(self) -> None:
        """Add constraints and objectives to MIP model."""
        from gurobipy import GRB, quicksum
        """
        each vertex in exactly one subcircuit
        """
        for v in range(self.n_vertices):
            ctName = "cons_vertex_" + str(v)
            self.model.addConstr(
                quicksum(
                    self.vertex_var[i][v] for i in range(self.num_subcircuit)
                ) == 1,
                name=ctName
            )

        """
        edge_var=1 indicates one and only one vertex of an edge is in subcircuit
        edge_var[subcircuit][edge] = vertex_var[subcircuit][u] XOR vertex_var[subcircuit][v]
        """
        for i in range(self.num_subcircuit):
            for e in range(self.n_edges):
                if len(self.edges[e]) != 2:
                    raise ValueError(
                        "Edges should be length 2 sequences: {self.edges[e]}"
                    )
                u = self.edges[e][0]
                v = self.edges[e][-1]
                u_vertex_var = self.vertex_var[i][u]
                v_vertex_var = self.vertex_var[i][v]
                ctName = "cons_edge_" + str(e)
                self.model.addConstr(
                    self.edge_var[i][e] - u_vertex_var - v_vertex_var <= 0,
                    name=ctName + "_1",
                )
                self.model.addConstr(
                    self.edge_var[i][e] - u_vertex_var + v_vertex_var >= 0,
                    name=ctName + "_2",
                )
                self.model.addConstr(
                    self.edge_var[i][e] - v_vertex_var + u_vertex_var >= 0,
                    name=ctName + "_3",
                )
                self.model.addConstr(
                    self.edge_var[i][e] + u_vertex_var + v_vertex_var <= 2,
                    name=ctName + "_4",
                )

        """
        Symmetry-breaking constraints
        Force small-numbered vertices into small-numbered subcircuits:
            v0: in subcircuit 0
            v1: in subcircuit_0 or subcircuit_1
            v2: in subcircuit_0 or subcircuit_1 or subcircuit_2
            ...
        """
        for vertex in range(self.num_subcircuit):
            ctName = "cons_symm_" + str(vertex)
            self.model.addConstr(
                quicksum(
                    self.vertex_var[subcircuit][vertex]
                    for subcircuit in range(vertex + 1)
                )
                == 1,
                name=ctName,
            )

        """
        Compute number of cuts
        """
        # Adds all the edges with a '1' value (a cut) for every subcircuit. Because a cut edge is counted twice (once
        #   in every subcircuit it is in), we divide by 2 to get the total number of cuts
        self.model.addConstr(
            self.num_cuts
            == quicksum(
                [
                    self.edge_var[subcircuit][i]
                    for i in range(self.n_edges)
                    for subcircuit in range(self.num_subcircuit)
                ]
            )
            / 2
        )

        """
        Compute the number of wire cuts and gate cuts
        """
        # Wire cuts is the sum of edges cut that are actually wire edges
        self.model.addConstr(
            self.num_wire_cuts
            == quicksum(
                [
                    self.edge_var[subcircuit][i]
                    for i in range(self.n_edges)
                    for subcircuit in range(self.num_subcircuit)
                    if self.edge_info[self.edges[i]]['edge_type'] == 'wire'
                ]
            )
            / 2
        )

        # Gate cuts is the sum of edges cut that are actually gate edges
        self.model.addConstr(
            self.num_gate_cuts
            == quicksum(
                [
                    self.edge_var[subcircuit][i]
                    for i in range(self.n_edges)
                    for subcircuit in range(self.num_subcircuit)
                    if self.edge_info[self.edges[i]]['edge_type'] == 'gate'
                ]
            )
            / 2
        )

        # Sanity check: the number of wire cuts and gate cuts should add up to the total number of cuts
        self.model.addConstr(
            self.num_cuts == self.num_wire_cuts + self.num_gate_cuts
        )

        # Number of cuts taking into account the cost of a wire cut SWAP
        self.model.addConstr(
            self.weighted_total_cuts == 2 * self.num_wire_cuts + self.num_gate_cuts
        )

        num_effective_qubits = []
        for subcircuit in range(self.num_subcircuit):
            """
            Compute number of different types of qubit in a subcircuit
            """

            for i in range(self.n_edges):
                if len(self.edges[i]) != 2:
                    raise ValueError(
                        "Edges should be length 2 sequences: {self.edges[i]}"
                    )

            # Counts the number of vertices in the subcircuit *per qubit*
            for q in range(self.num_qubits):
                self.model.addConstr(
                    self.num_vertices_in_subcircuit_per_qubit[subcircuit][q]
                    ==
                    # The number of vertices in the subcircuit that correspond to the qubit
                    quicksum(
                        self.vertex_var[subcircuit][v]
                        for v in range(self.n_vertices)
                        if self.id_qubit[v] == self.qubit_names[q]
                    )
                )

            # If the number of the number of vertices is at least 1, then that qubit is in the subcircuit
            for j in range(self.num_qubits):
                # self.model.addConstr(
                #     (self.num_vertices_in_subcircuit_per_qubit[subcircuit][j] >= 1),
                #     GRB.EQUAL,
                #     self.qubit_in_subcircuit[subcircuit][j],
                #     name="qubit_in_subcircuit_sc%d_q%d" % (subcircuit, j),
                # )
                M = 100  # Big-M, should be sufficiently large
                self.model.addConstr(
                    self.qubit_in_subcircuit[subcircuit][j] >= self.num_vertices_in_subcircuit_per_qubit[subcircuit][
                        j] / M,
                    name="qubit_in_subcircuit_sc%d_q%d_lb" % (subcircuit, j)
                )
                self.model.addConstr(
                    self.qubit_in_subcircuit[subcircuit][j] <= self.num_vertices_in_subcircuit_per_qubit[subcircuit][j],
                    name="qubit_in_subcircuit_sc%d_q%d_ub" % (subcircuit, j)
                )

            # d is the total number of qubits in a subcircuit. It can't be more than the max subcircuit width
            self.model.addConstr(
                self.subcircuit_counter[subcircuit]["d"]
                - quicksum(
                    self.qubit_in_subcircuit[subcircuit][qubit]
                    for qubit in range(self.num_qubits)
                )
                == 0,
                name="cons_subcircuit_width_%d" % subcircuit,
            )

        # Constraint to check if wire cuts count as 2 cuts, for when using a SWAP gate for the move operation
        if self.double_wire_cost:
            self.model.setObjective( self.weighted_total_cuts, GRB.MINIMIZE)
        else:
            self.model.setObjective( self.num_cuts, GRB.MINIMIZE)


    def check_graph(self, n_vertices: int, edges: Sequence[tuple[int]]) -> None:
        """
        Ensure circuit DAG is viable.

        This means that there are no oversized edges, that all edges are from viable nodes,
        and that the graph is otherwise a valid graph.

        Args:
            n_vertices: The number of vertices
            edges: The edge list

        Returns:
            None

        Raises:
            ValueError: The graph is invalid
        """
        # 1. edges must include all vertices
        # 2. all u,v must be ordered and smaller than n_vertices
        vertices = set([i for (i, _) in edges])  # type: ignore
        vertices |= set([i for (_, i) in edges])  # type: ignore
        assert vertices == set(range(n_vertices))
        for edge in edges:
            if len(edge) != 2:
                raise ValueError("Edges should be length 2 sequences: {edge}")
            u = edge[0]
            v = edge[-1]
            if u > v:
                raise ValueError(f"Edge u ({u}) cannot be greater than edge v ({v})")
            if u > n_vertices:
                raise ValueError(
                    f"Edge u ({u}) cannot be greater than number of vertices ({n_vertices})"
                )

    def solve(self, min_postprocessing_cost: float) -> bool:
        """
        Solve the MIP model.

        Args:
            min_post_processing_cost: The predicted minimum post-processing cost,
                often is inf

        Returns:
            Flag denoting if the model found a solution
        """

        from gurobipy import GurobiError, GRB

        # print('solving for %d subcircuits'%self.num_subcircuit)
        # print('model has %d variables, %d linear constraints,%d quadratic constraints, %d general constraints'
        # % (self.model.NumVars,self.model.NumConstrs, self.model.NumQConstrs, self.model.NumGenConstrs))
        try:
            self.model.Params.TimeLimit = 300
            self.model.Params.cutoff = min_postprocessing_cost
            self.model.optimize()
        except (GurobiError, AttributeError, Exception) as e:
            print('Caught: ' + e.message)

        if self.model.solcount > 0:
            self.objective = None
            self.subcircuits_vertices = []
            self.optimal = (self.model.Status == GRB.OPTIMAL)
            self.runtime = self.model.Runtime
            self.node_count = self.model.nodecount
            self.mip_gap = self.model.mipgap
            self.objective = self.model.ObjVal

            for i in range(self.num_subcircuit):
                subcircuit_vertices = []
                for j in range(self.n_vertices):
                    if abs(self.vertex_var[i][j].x) > 1e-4:
                        subcircuit_vertices.append(self.id_vertices[j])
                self.subcircuits_vertices.append(subcircuit_vertices)
            assert sum([len(x) for x in self.subcircuits_vertices]) == self.n_vertices

            cut_edges_idx = []
            cut_edges = []
            for i in range(self.num_subcircuit):
                for j in range(self.n_edges):
                    if abs(self.edge_var[i][j].x) > 1e-4 and j not in cut_edges_idx:
                        cut_edges_idx.append(j)
                        u = self.edges[j][0]
                        v = self.edges[j][1]
                        cut_edges.append((self.id_vertices[u], self.id_vertices[v]))
            self.cut_edges = cut_edges
            self.weighted_cost = self.weighted_total_cuts.x
            return True
        else:
            # print('Infeasible')
            return False

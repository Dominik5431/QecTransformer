import networkx as nx
import numpy as np
import stim
from matplotlib import pyplot as plt
from tqdm import tqdm


class SurfaceCode():
    def __init__(self, distance, noise, noise_model='depolarizing'):
        self.noise_model = noise_model
        self.noise = noise
        self.distance = distance


    def measure_all_z(self, coord_to_index, index_to_coordinate, list_z_ancillas_index):
        circuit = stim.Circuit()
        list_pairs = [[+1, -1], [-1, -1], [-1, +1], [+1, +1]]

        for xi, yi in list_pairs:
            for ancilla_qubit_idx in list_z_ancillas_index:
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]

                if "({},{})".format(coord_x + xi, coord_y + yi) in coord_to_index:
                    data_qubit_idx = coord_to_index["({},{})".format(coord_x + xi, coord_y + yi)]
                    circuit.append("CNOT", [data_qubit_idx, ancilla_qubit_idx])

        circuit.append("TICK")

        return circuit

    def measure_all_x(self, coord_to_index, index_to_coordinate, list_x_ancillas_index):
        circuit = stim.Circuit()

        circuit.append("H", list_x_ancillas_index)
        circuit.append("TICK")

        list_pairs = [[+1, -1], [-1, -1], [-1, +1], [+1, +1]]

        for xi, yi in list_pairs:
            for ancilla_qubit_idx in list_x_ancillas_index:
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]

                if "({},{})".format(coord_x + xi, coord_y + yi) in coord_to_index:
                    data_qubit_idx = coord_to_index["({},{})".format(coord_x + xi, coord_y + yi)]
                    circuit.append("CNOT", [ancilla_qubit_idx, data_qubit_idx])

        circuit.append("TICK")

        circuit.append("H", list_x_ancillas_index)
        circuit.append("TICK")

        return circuit

    def measure_bell_stabilizers(self, coord_to_index, reference_qubit_index, reference_ancillas_index, Ly, Lx):
        circuit = stim.Circuit()

        # Z_R Z_L stabilizer
        circuit.append("CNOT", [reference_qubit_index, reference_ancillas_index[0]])
        for i in range(Ly):  # Ly):
            for xi in range(Lx):
                x = 2 * xi + 1
                circuit.append("CNOT", [coord_to_index["({},{})".format(x, 2 * i + 1)], reference_ancillas_index[0]])
        circuit.append("TICK")

        # X_R X_L stabilizer
        circuit.append("H", reference_ancillas_index[1])  # 1 instead of Ly + 1
        circuit.append("CNOT", [reference_ancillas_index[1], reference_qubit_index])
        for i in range(Lx):  # Lx):
            for yi in range(Ly):
                y = 2 * yi + 1
                circuit.append("CNOT", [reference_ancillas_index[1], coord_to_index["({},{})".format(2 * i + 1, y)]])
        circuit.append("H", reference_ancillas_index[1])
        circuit.append("TICK")

        return circuit

    def create_code_instance(self):
        circuit = stim.Circuit()

        Lx, Ly = self.distance, self.distance
        Lx_ancilla, Ly_ancilla = 2 * Lx + 1, 2 * Ly + 1

        coord_to_index = {}
        index_to_coordinate = []

        # data qubit coordinates
        qubit_idx = 0
        for yi in range(Ly):
            y = 2 * yi + 1
            for xi in range(Lx):
                x = 2 * xi + 1
                circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x, y) + " {}".format(qubit_idx))
                coord_to_index.update({"({},{})".format(x, y): qubit_idx})
                index_to_coordinate.append([x, y])

                qubit_idx += 1

        # ancilla qubit coordinates

        list_z_ancillas_index = []
        list_x_ancillas_index = []
        list_data_index = []

        for i in range(Lx * Ly):
            list_data_index.append(i)

        for x in range(2, Lx_ancilla - 1, 4):
            circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x, 0) + " {}".format(qubit_idx))
            coord_to_index.update({"({},{})".format(x, 0): qubit_idx})
            index_to_coordinate.append([x, 0])

            list_x_ancillas_index.append(qubit_idx)

            qubit_idx += 1

        for y in range(2, Ly_ancilla - 1, 2):
            yi = y % 4
            xs = range(yi, 2 * Lx + yi // 2, 2)
            for idx, x in enumerate(xs):
                circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x, y) + " {}".format(qubit_idx))
                coord_to_index.update({"({},{})".format(x, y): qubit_idx})
                index_to_coordinate.append([x, y])

                if idx % 2 == 0:
                    list_z_ancillas_index.append(qubit_idx)
                elif idx % 2 == 1:
                    list_x_ancillas_index.append(qubit_idx)

                qubit_idx += 1

        for x in range(4, Lx_ancilla, 4):
            circuit.append_from_stim_program_text(
                "QUBIT_COORDS({},{})".format(x, Ly_ancilla - 1) + " {}".format(qubit_idx))
            coord_to_index.update({"({},{})".format(x, Ly_ancilla - 1): qubit_idx})
            index_to_coordinate.append([x, Ly_ancilla - 1])
            list_x_ancillas_index.append(qubit_idx)

            qubit_idx += 1

        # reference qubit coordinates
        reference_index = qubit_idx
        circuit.append_from_stim_program_text(
            "QUBIT_COORDS({},{})".format(Lx_ancilla - 1, Ly_ancilla - 1) + " {}".format(qubit_idx))
        coord_to_index.update({"({},{})".format(Lx_ancilla - 1, Ly_ancilla - 1): qubit_idx})
        index_to_coordinate.append([Lx_ancilla - 1, Ly_ancilla - 1])
        qubit_idx += 1

        reference_ancillas = []
        # logical z reference qubit
        for i in range(1):  # Ly):
            circuit.append_from_stim_program_text(
                "QUBIT_COORDS({},{})".format(Lx_ancilla + i, Ly_ancilla - 1) + " {}".format(qubit_idx))
            coord_to_index.update({"({},{})".format(Lx_ancilla + i, Ly_ancilla - 1): qubit_idx})
            index_to_coordinate.append([Lx_ancilla + i, Ly_ancilla - 1])
            reference_ancillas.append(qubit_idx)
            qubit_idx += 1

        # logical x reference qubit
        for i in range(1):  # Lx):
            circuit.append_from_stim_program_text(
                "QUBIT_COORDS({},{})".format(Lx_ancilla - 1, Ly_ancilla + i) + " {}".format(qubit_idx))
            coord_to_index.update({"({},{})".format(Lx_ancilla - 1, Ly_ancilla + i): qubit_idx})
            index_to_coordinate.append([Lx_ancilla - 1, Ly_ancilla + i])
            reference_ancillas.append(qubit_idx)
            qubit_idx += 1

        measure_z = self.measure_all_z(coord_to_index, index_to_coordinate, list_z_ancillas_index)
        measure_x = self.measure_all_x(coord_to_index, index_to_coordinate, list_x_ancillas_index)
        measure_bell = self.measure_bell_stabilizers(coord_to_index, reference_index, reference_ancillas, Ly, Lx)

        circuit.append("R", range(2 * Lx * Ly - 1))
        circuit.append("TICK")

        circuit += measure_z
        circuit += measure_x
        circuit += measure_bell

        circuit.append("MR", list_z_ancillas_index)
        circuit.append("MR", list_x_ancillas_index)
        circuit.append("MR", reference_ancillas)
        circuit.append("TICK")

        # errors
        if self.noise_model == 'depolarizing':
            circuit.append("DEPOLARIZE1", list_data_index, self.noise)
        elif self.noise_model == 'bitflip':
            circuit.append("X_ERROR", list_data_index, self.noise)
        else:
            raise ValueError("Unknown noise_model {}".format(self.noise_model))
        circuit.append("TICK")

        circuit += measure_z
        circuit += measure_x
        circuit += measure_bell

        circuit.append("M", list_z_ancillas_index)
        circuit.append("M", list_x_ancillas_index)
        circuit.append("M", reference_ancillas)

        offset = (Lx * Ly - 1) // 2
        r_offset = len(reference_ancillas)

        for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            circuit.append_from_stim_program_text(
                "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx + offset + r_offset,
                                                                                         1 + idx + 3 * offset + 2 * r_offset))

        for idx, ancilla_qubit_idx in enumerate(list_x_ancillas_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            circuit.append_from_stim_program_text(
                "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx + r_offset,
                                                                                         1 + idx + 2 * offset + 2 * r_offset))

        for idx, ancilla_qubit_idx in enumerate(reference_ancillas[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            circuit.append_from_stim_program_text(
                "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx,
                                                                                         1 + idx + r_offset + 2 * offset))

        return circuit



class DEM_Graph:
    def __init__(self, distance, noise, noise_model):
        circuit = SurfaceCode(distance, noise, noise_model).create_code_instance()
        dem = circuit.detector_error_model()
        print(dem)
        dem_text = str(dem)
        self.graph = nx.Graph()
        p1, px, py, pz = 0, 0, 0, 0
        for line in dem_text.splitlines():
            if line.startswith("error"):
                parts = line.split()
                weight = float(parts[0].split('(')[1].rstrip(')'))  # Extract error probability
                error_nodes = [p for p in parts if p.startswith("D")]
                if 'D8' in error_nodes and 'D9' in error_nodes:
                    py += weight
                elif 'D8' in error_nodes:
                    pz += weight
                elif 'D9' in error_nodes:
                    px += weight
                else:
                    p1 += weight
                if len(error_nodes) == 1:
                    # Add a self-loop with the weight
                    self.graph.add_edge(error_nodes[0], error_nodes[0], weight=weight)
                else:
                    # Add weighted edges between detectors
                    for i in range(len(error_nodes)):
                        for j in range(i + 1, len(error_nodes)):
                            if self.graph.has_edge(error_nodes[i], error_nodes[j]):
                                # If the edge already exists, add the weight to the existing one
                                self.graph[error_nodes[i]][error_nodes[j]]['weight'] += weight
                            else:
                                self.graph.add_edge(error_nodes[i], error_nodes[j], weight=weight)

        self.pr = p1**2 + px**2 + py**2 + pz**2
        # print(graph.edges())

    def get_graph(self):
        return self.graph


if __name__ == "__main__":
    noises = np.arange(0, 0.4, 0.02)
    pr = np.zeros(len(noises))
    for i, noise in enumerate(tqdm(noises)):
        graph = DEM_Graph(distance=3, noise=noise, noise_model='depolarizing')
        pr[i] = graph.pr
    plt.plot(noises, pr)
    '''
    graph = DEM_Graph(3, 0.03, 'depolarizing').get_graph()

    # Visualize the graph with weights
    pos = nx.spring_layout(graph)  # Compute layout positions
    edges = graph.edges(data=True)

    # Draw nodes and edges
    nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", font_weight="bold")
    edge_labels = {(u, v): f"{data['weight']:.3f}" for u, v, data in edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color="red")
    '''
    plt.show()
from abc import ABC, abstractmethod
import stim
import numpy as np
from pathlib import Path
from random import random


class QECCode(ABC):
    def __init__(self, distance, noise):
        self.distance = distance
        self.noise = noise
        if distance % 2 == 0:
            raise ValueError("Not optimal distance.")
        self.circuit = self.create_code_instance()

    def circuit_to_png(self):
        diagram = self.circuit.diagram('timeline-svg')
        with open("diagram_testing.svg", 'w') as f:
            f.write(diagram.__str__())

    @abstractmethod
    def create_code_instance(self):
        raise NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def get_syndromes(self, n):
        raise NotImplementedError("Subclasses should implement this!")


class SurfaceCode(QECCode):
    def __init__(self, distance, noise, noise_model='depolarizing'):
        self.noise_model = noise_model
        super().__init__(distance, noise)


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

    def get_syndromes(self, n, only_syndromes: bool = False):
        sampler = self.circuit.compile_detector_sampler()
        samples = sampler.sample(shots=n)
        samples = np.array(list(map(lambda y: np.where(y, 1, 0), samples)))
        syndromes = samples
        if only_syndromes:
            return syndromes[:, :-2]  #  * self.distance]
        return syndromes


class RepetitionCode(QECCode):
    def __init__(self, distance, noise):
        super().__init__(distance, noise)

    def measure_all_z(self, coord_to_index, index_to_coordinate, list_z_ancillas_index):
        circuit = stim.Circuit()
        list_pairs = [[+1, 0], [-1, 0]]

        for xi, yi in list_pairs:
            for ancilla_qubit_idx in list_z_ancillas_index:
                coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]

                if "({},{})".format(coord_x + xi, coord_y + yi) in coord_to_index:
                    data_qubit_idx = coord_to_index["({},{})".format(coord_x + xi, coord_y + yi)]
                    circuit.append("CNOT", [data_qubit_idx, ancilla_qubit_idx])

        circuit.append("TICK")

        return circuit

    def measure_all(self, coord_to_index, index_to_coordinate, list_data_index, list_log_ancillas_index):
        circuit = stim.Circuit()

        for ai in list_log_ancillas_index:
            for xi in list_data_index:
                circuit.append("CNOT", [xi, ai])

        circuit.append("TICK")

        return circuit

    def create_code_instance(self):
        circuit = stim.Circuit()

        L = self.distance
        L_ancilla = 2 * L

        coord_to_index = {}
        index_to_coordinate = []

        # data qubit coordinates
        qubit_idx = 0
        for i in range(L):
            j = 2 * i
            circuit.append_from_stim_program_text("QUBIT_COORDS({},0)".format(j) + " {}".format(qubit_idx))
            coord_to_index.update({"({},0)".format(j): qubit_idx})
            index_to_coordinate.append([j, 0])

            qubit_idx += 1

        # ancilla qubit coordinates
        list_z_ancillas_index = []
        list_data_index = []

        for i in range(L):
            list_data_index.append(i)

        for i in range(1, L + 1, 2):
            circuit.append_from_stim_program_text("QUBIT_COORDS({},0)".format(i) + " {}".format(qubit_idx))
            coord_to_index.update({"({},{})".format(i, 0): qubit_idx})
            index_to_coordinate.append([i, 0])

            list_z_ancillas_index.append(qubit_idx)

            qubit_idx += 1

        # logical qubit coordinate
        list_log_ancilla_index = [qubit_idx]
        circuit.append_from_stim_program_text(
            "QUBIT_COORDS({},{})".format(L_ancilla, 0) + " {}".format(qubit_idx))
        coord_to_index.update({"({},{})".format(L_ancilla, 0): qubit_idx})
        index_to_coordinate.append([L_ancilla, 0])
        qubit_idx += 1

        measure_z = self.measure_all_z(coord_to_index, index_to_coordinate, list_z_ancillas_index)
        measure_logical = self.measure_all(coord_to_index, index_to_coordinate, list_data_index, list_log_ancilla_index)

        circuit.append("R", range(2 * L))
        circuit.append("TICK")

        circuit += measure_z
        circuit += measure_logical

        circuit.append("MR", list_z_ancillas_index)
        circuit.append("MR", list_log_ancilla_index)
        circuit.append("TICK")

        # errors
        circuit.append("X_ERROR", list_data_index, self.noise)
        circuit.append("TICK")

        circuit += measure_z
        circuit += measure_logical

        circuit.append("M", list_z_ancillas_index)
        circuit.append("M", list_log_ancilla_index)

        r_offset = len(list_log_ancilla_index)

        for idx, ancilla_qubit_idx in enumerate(list_z_ancillas_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_qubit_idx]
            circuit.append_from_stim_program_text(
                "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx + r_offset,
                                                                                         1 + idx + L - 1 + 2 * r_offset))

        for idx, ancilla_log_index in enumerate(list_log_ancilla_index[::-1]):
            coord_x, coord_y = index_to_coordinate[ancilla_log_index]
            circuit.append_from_stim_program_text(
                "DETECTOR({},{})".format(coord_x, coord_y) + " rec[-{}] rec[-{}]".format(1 + idx,
                                                                                         1 + idx + L - 1 + r_offset))

        return circuit

    def get_syndromes(self, n, only_syndromes: bool = False):
        sampler = self.circuit.compile_detector_sampler()
        samples = sampler.sample(shots=n)
        samples = np.array(list(map(lambda y: np.where(y, 1, 0), samples)))
        syndromes = samples
        if only_syndromes:
            return syndromes[:, :-1]  #  * self.distance]
        return syndromes


if __name__ == '__main__':
    s = RepetitionCode(3, 0.4)
    print(s.get_syndromes(3))
    s.circuit_to_png()
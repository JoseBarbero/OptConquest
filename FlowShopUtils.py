import numpy as np


def read_file(file):
    """
    Crea una matriz con los datos del problema de Flowshop permutacional a partir de un fichero.
    """
    f_input = open(file, 'r')
    lines = f_input.readlines()
    f_input.close()
    rows, columns = lines[0].split()
    content = np.empty((int(rows), int(columns)), dtype=int)
    i = 0
    for line in lines:
        content[i - 1] = np.array(line.split()[1::2])
        i += 1
    return content


def f(solution, data):
    """
    Crea la matriz de costes (f) de cierta solución dada.
    """
    cost_matrix = np.zeros_like(data)
    cost_matrix[0] = np.add.accumulate(data[solution[0]])

    # No se sigue el orden inicial de la matriz para evitar complicaciones de indexado,
    # las filas van en el orden de la solución respecto a la matriz inicial
    for i in range(1, len(solution)):
        for j in range(len(data[0])):
            cost_matrix[i][j] = max(cost_matrix[i - 1][j], cost_matrix[i][j - 1]) + data[solution[i]][j]
    return cost_matrix


def fmed(solution, data):
    """
    Calcula fmed a la vez que crea la matriz f (más eficiente).
    """
    cost_matrix = np.zeros_like(data)
    cost_matrix[0] = np.add.accumulate(data[solution[0]])

    # No se sigue el orden inicial de la matriz para evitar complicaciones de indexado,
    # las filas van en el orden de la solución respecto a la matriz inicial
    sum_last_column = 0
    n_rows = len(solution)
    n_cols = len(data[0])
    for i in range(1, n_rows):
        for j in range(n_cols):
            cost_matrix[i][j] = max(cost_matrix[i - 1][j], cost_matrix[i][j - 1]) + data[solution[i]][j]
            if j == n_cols-1:
                sum_last_column += cost_matrix[i][j]
    return sum_last_column/n_rows


def old_fmed(cost_matrix):
    """
    Calcula el valor fmed de una matriz de costes.
    """
    # Redondeo porque excesivos decimales a veces dan problemas al hacer comparaciones
    return np.mean(cost_matrix[:, -1])


def fmax(cost_matrix):
    """
    Calcula el valor fmax de una matriz de costes.
    """
    return cost_matrix[-1, -1]


def print_pop(pop):
    """
    Imprime una población con saltos de línea.
    """
    for s in pop:
        print(*s)


def fmed_mean(population, data):
    """
    Calcula el fmed medio de la población.
    """
    return np.mean([fmed(f(solution, data)) for solution in population])


def evaluate_algorithm(algorithm, params):
    """
    Ejecuto el algoritmo 5 veces con un dataset del tamaño del real para sacar el fmed medio.
    """
    results = [algorithm(*params)[0] for _ in range(5)]
    return sum(results)/len(results), min(results)






import numpy as np
cimport numpy as np
cimport cpython

def opt_fmed(solution, data):

    cdef np.ndarray c_cost_matrix = np.zeros([len(data), len(data[0]), dtype=np.double])
    c_cost_matrix[0] = np.add.accumulate(data[solution[0]])

    sum_last_column = c_cost_matrix[0, -1]
    n_rows = len(solution)
    n_cols = len(data[0])
    for i in range(1, n_rows):
        for j in range(n_cols):
            c_cost_matrix[i][j] = max(c_cost_matrix[i - 1][j], c_cost_matrix[i][j - 1]) + data[solution[i]][j]
            if j == n_cols-1:
                sum_last_column += c_cost_matrix[i][j]

    return sum_last_column/n_rows
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef opt_fmed(solution, data):
    cdef int n_rows = len(solution)
    cdef int n_cols = len(data[0])
    
    cdef np.ndarray c_cost_matrix = np.zeros([n_rows, n_cols])
    c_cost_matrix[0] = np.add.accumulate(data[solution[0]])

    cdef float sum_last_column = c_cost_matrix[0, -1]
    
    for i in range(1, n_rows):
        for j in range(n_cols):
            c_cost_matrix[i, j] = max(c_cost_matrix[i - 1, j], c_cost_matrix[i, j - 1]) + data[solution[i], j]

    return c_cost_matrix[:, -1].mean()
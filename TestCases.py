from FlowShopUtils import *


def reorder(cost_matrix, sol):
    """
    Reordena una matriz de costes (f) para comparar mi solución con las de los casos de prueba.
    """
    ordered_mat = np.zeros_like(cost_matrix)
    i = 0
    for step in sol:
        ordered_mat[step] = cost_matrix[i]
        i += 1
    return ordered_mat

# timeit.timeit('read_file("ProblemasFlowShopPermutacional/Doc11.txt")', setup="from __main__ import read_file", number=100000)

sol_doc1_mat = np.loadtxt("Tests/SolDoc1_Mat.txt", usecols=range(5), dtype=int)
sol_doc1_vec = np.loadtxt("Tests/SolDoc1_Vec.txt", usecols=range(11), dtype=int)

sol_doc2_mat = np.loadtxt("Tests/SolDoc2_Mat.txt", usecols=range(4), dtype=int)
sol_doc2_vec = np.loadtxt("Tests/SolDoc2_Vec.txt", usecols=range(13), dtype=int)

sol_doc1_vec = [a-1 for a in sol_doc1_vec]
sol_doc2_vec = [a-1 for a in sol_doc2_vec]

doc1 = read_file("ProblemasFlowShopPermutacional/Doc1.txt")
doc2 = read_file("ProblemasFlowShopPermutacional/Doc2.txt")

print(reorder(f(sol_doc1_vec, doc1), sol_doc1_vec) == sol_doc1_mat)
print(reorder(f(sol_doc2_vec, doc2), sol_doc2_vec) == sol_doc2_mat)

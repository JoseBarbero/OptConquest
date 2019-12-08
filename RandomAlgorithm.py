from FlowShopUtils import *


def create_random_solution(size):
    return np.random.permutation(range(size))


def random_algorithm(data, n_iter):
    """
    Algoritmo que crea soluciones aleatorias y se queda con la mejor durante n_iter iteraciones.
    """
    best = []
    best_fmed = np.inf
    for i in range(n_iter):

        print('\r' + str(i + 1) + "/" + str(n_iter), end='')

        sol = create_random_solution(len(data))
        sol_fmed = fmed(f(sol, data))
        if sol_fmed < best_fmed:
            best_fmed = sol_fmed
            best = sol
    return best_fmed, best

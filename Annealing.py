from Evolutive import *


def select_neighbours(n, solution, t, data):
    """
    Genera n vecinos y sobre la marcha decide si se aceptan o no.
    """
    # TODO los vecinos pueden generarse mutando mucho o poco
    for _ in range(n):
        vecino = swap_mutation(solution, 100)
        solution = accept(solution, vecino, t, data)

    return solution


def accept(ini_solution, new_solution, t, data):
    # TODO Igual se puede evitar calcular fmed algunas veces
    pre = fmed(ini_solution, data)
    post = fmed(new_solution, data)

    p = np.exp((-(pre - post) / t))

    if p < 1:
        return new_solution
    else:
        return ini_solution


def simulated_annealing(t, alpha, solution, time_, n_neighbours, data):
    # Bucle de 60 segundos
    t_end = time.time() + time_ - 0.5
    while time.time() < t_end:
        solution = select_neighbours(n_neighbours, solution, t, data)
        if t - alpha > 0:
            t -= alpha  # TODO esto puede ir variando y hay que vigilar que no baje de 0
    return fmed(solution, data)


from Evolutive import *


def select_neighbours(n, solution, solution_fmed, t, data):
    """
    Genera n vecinos y sobre la marcha decide si se aceptan o no.
    """
    # TODO los vecinos pueden generarse mutando mucho o poco
    for _ in range(n):
        vecino = swap_mutation(solution, 100)
        solution, solution_fmed = accept(solution, solution_fmed, vecino, t, data)

    return solution, solution_fmed


def accept(ini_solution, ini_solution_fmed, new_solution, t, data):
    pre = ini_solution_fmed
    post = fmed(new_solution, data)

    p = np.exp((-(post - pre) / t))

    if post < pre:
        #print(post, file=open("fmed_evolution.txt", "a"))
        return new_solution, post
    elif np.random.random() < p:
        #print(post, file=open("fmed_evolution.txt", "a"))
        return new_solution, post
    else:
        #print(pre, file=open("fmed_evolution.txt", "a"))
        return ini_solution, pre


def simulated_annealing(ini_t, alpha, solution, time_, n_neighbours, data):
    # Bucle de 60 segundos
    t_end = time.time() + time_ - 0.5
    t = ini_t
    solution_fmed = fmed(solution, data)
    while time.time() < t_end:
        solution, solution_fmed = select_neighbours(n_neighbours, solution, solution_fmed, t, data)
        step = t*alpha/100

        if t - step > 0.1:
            t -= step
    return solution, solution_fmed

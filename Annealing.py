import copy
from Evolutive import *


def get_neighbour(solution):
    sol = copy.deepcopy(solution)
    swap_indexes = np.random.choice(len(sol), 2, replace=False)
    sol[swap_indexes[0]], sol[swap_indexes[1]] = solution[swap_indexes[1]], solution[swap_indexes[0]]
    return sol


def select_neighbours(n, solution, solution_fmed, t, data):
    """
    Genera n vecinos y sobre la marcha decide si se aceptan o no.
    """

    for _ in range(n):
        vecino = get_neighbour(solution)
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


def simulated_annealing(tries, t_ini_factor, alpha, solution, time_, data):

    # Bucle de 60 segundos
    t_end = time.time() + time_ - 0.1
    solution_fmed = fmed(solution, data)
    t = solution_fmed * t_ini_factor

    while time.time() < t_end:
        for i in range(tries): # Tries before getting t down
            vecino = get_neighbour(solution)
            solution, solution_fmed = accept(solution, solution_fmed, vecino, t, data)

        step = t*alpha/100  # Baja la temperatura un alpha%

        if t - step > 0.1:
            t -= step

    return solution, solution_fmed


def local_best_search(solution, solution_fmed, data, time_limit):
    """
    Búsqueda local del mejor.
    """
    best_fmed = solution_fmed
    best_solution = solution
    t_end = time.time() + time_limit
    for neighbour in generate_neighbours(solution):
        if time.time() < t_end:
            neighbour_fmed = fmed(neighbour, data)
            if neighbour_fmed < best_fmed:
                best_solution, best_fmed = neighbour, neighbour_fmed
        else:
            print("TIME'S UP")
            return best_solution, best_fmed
    return best_solution, best_fmed


def local_best_first_search(solution, solution_fmed, data, time_limit):
    """
    Búsqueda local del primer mejor.
    """
    t_end = time.time() + time_limit
    best_solution = solution
    best_fmed = solution_fmed

    while time.time() < t_end:
        target_fmed = best_fmed
        for neighbour in generate_neighbours(best_solution):
            neighbour_fmed = fmed(neighbour, data)
            if neighbour_fmed < best_fmed:
                best_solution, best_fmed = neighbour, neighbour_fmed
        if target_fmed == best_fmed:
            break
    return best_solution, best_fmed

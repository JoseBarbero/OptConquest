import time
from multiprocessing import Pool
from Annealing import *
from Evolutive import *
from RandomAlgorithm import *
from operator import itemgetter

def show_results(n_runs, cores, p_e):
    res = []
    for _ in range(n_runs):
        res.append(p_e(cores, 25))

    print(f"\tMean: {sum(res)/len(res)}")
    print(f"\tBest: {min(res)}")


def evolutive_worker(mutation):
    pop = Population(75, 50, read_file("Datasets/Doc11.txt"))

    return evolutive_algorithm(read_file("Datasets/Doc11.txt"), pop,
                          time_=60, elite_size=1, mut_ratio=mutation[1], diversify_size=0, not_improving_limit=False,
                          sel_f=median_selection,
                          elite_f=get_elite,
                          rep_f=ox_reproduction,
                          mut_f=mutate)[0]


def parallel_evolutive(n_processes, mutation):
    p = Pool(processes=n_processes)
    results = p.map(evolutive_worker, [(i, mutation) for i in range(n_processes)])
    p.close()
    return results


def annealing_worker(params):
    t, alpha, solution0, n_neighbours = params[1]
    return simulated_annealing(t, alpha, solution0, 60, n_neighbours, read_file("Datasets/Doc11.txt"))


def parallel_annealing(n_processes, t, alpha, n_neighbours):
    p = Pool(processes=n_processes)

    solution0 = create_random_solution(75)
    params = [t, alpha, solution0, n_neighbours]
    results_ = p.map(annealing_worker, [(i, params) for i in range(n_processes)])

    solutions, fmeds = zip(*results_)
    p.close()
    idx_best = min(enumerate(fmeds), key=itemgetter(1))[0]

    print(fmeds, "--->", min(fmeds))

    return results_[idx_best]


def test_params():
    t_list = [100, 200, 250, 350]
    alpha_list = [0.5, 1, 2, 3, 5]
    n_neighbours_list = [1, 3, 5, 10]

    for t, alpha, n_neighbours in list(itertools.product(t_list, alpha_list, n_neighbours_list)):
        print("t, alpha, n_neighbours:", t, alpha, n_neighbours, file=open("output.txt", "a"))
        print(t, alpha, n_neighbours, file=open("means.txt", "a"))
        parallel_annealing(4, t, alpha, n_neighbours)


if __name__ == '__main__':
    # print(parallel_evolutive(4, 10))
    results = []
    for _ in range(5):
        results.append(parallel_annealing(4, 250, 3, 5)[1])
    print("Mean fmed:", np.mean(results))
    print("Best fmed:", np.min(results))

#TODO Parametrizar bien y probar otros métodos para bajar la temperatura
#Todo Probar con diferentes niveles de mutación
#Todo Probar si merece la pena empezar con una buena solución de un genético
#Todo Probar si merece la pena acabar con una buena solución de un genético
#Todo Búsqueda local al final
#Todo 30 segundos y otros 30 empezando con el mejor (maybe busqueda local por medio)
#Todo guardar los últimos 10 fmeds en un diccionario para evitar recalcular lo mismo
#Todo mutacion loca en un nucleo
#Todo Igual merece la pena hacer otro algoritmo cuando este se atasque
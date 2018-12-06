import time
from multiprocessing import Pool
from Annealing import *
from Evolutive import *
from RandomAlgorithm import *


def worker(mutation):
    pop = generate_random_population(75, 100)
    return evolutive_algorithm(read_file("Datasets/Doc11.txt"), pop, pop_size=3,
                          time_=60, elite_size=1, mut_ratio=mutation[1], diversify_size=0, not_improving_limit=False,
                          sel_f=median_selection,
                          elite_f=get_elite,
                          rep_f=ox_reproduction,
                          mut_f=mutate)[0][0]


def worker_2(mutation):
    pop = generate_random_population(75, 100)
    return evolutive_algorithm(read_file("Datasets/Doc11.txt"), pop, pop_size=100,
                          time_=10, elite_size=1, mut_ratio=mutation[1], diversify_size=0, not_improving_limit=False,
                          sel_f=median_selection,
                          elite_f=get_elite,
                          rep_f=ox_reproduction,
                          mut_f=mutate)


def worker_3(args):
    _, (mutation, pop) = args
    # TODO devolver la solución en sí
    return evolutive_algorithm(read_file("Datasets/Doc11.txt"), pop, pop_size=100,
                          time_=50, elite_size=1, mut_ratio=mutation, diversify_size=0, not_improving_limit=False,
                          sel_f=median_selection,
                          elite_f=get_elite,
                          rep_f=ox_reproduction,
                          mut_f=mutate)[0][0]


def parallel_evolutive(n_processes, mutation):
    p = Pool(processes=n_processes)
    results = p.map(worker, [(i, mutation) for i in range(n_processes)])
    p.close()
    return min(results)


def parallel_evolutive_v2(n_processes, mutation):
    p = Pool(processes=n_processes)
    results = p.map(worker_2, [(i, mutation) for i in range(n_processes)])
    p.close()

    best_pop = min(results, key=operator.itemgetter(0))[1]

    p2 = Pool(processes=n_processes)
    results2 = p2.map(worker_3, [(i, (mutation, best_pop)) for i in range(n_processes)])
    p2.close()

    return min(results2)


def show_results(n_runs, cores, p_e):
    res = []
    for _ in range(n_runs):
        res.append(p_e(cores, 25))

    print(f"\tMean: {sum(res)/len(res)}")
    print(f"\tBest: {min(res)}")


def annealing_worker(params):
    t, alpha, n_neighbours = params[1]
    solution0 = create_random_solution(75)
    return simulated_annealing(t, alpha, solution0, 60, n_neighbours, read_file("Datasets/Doc11.txt"))

# 500. 0.1 10


def parallel_annealing(n_processes, t, alpha, n_neighbours):
    p = Pool(processes=n_processes)
    params = [t, alpha, n_neighbours]
    results = p.map(annealing_worker, [(i, params) for i in range(n_processes)])
    p.close()
    print(results, file=open("output.txt", "a"))
    print(np.mean(results), file=open("means.txt", "a"))
    return min(results)


def test_params():
    t_list = [100, 500, 1000, 5000]
    alpha_list = [0.1, 0.5, 1, 5, 10]
    n_neighbours_list = [1, 3, 5, 10, 50]

    for t, alpha, n_neighbours in list(itertools.product(t_list, alpha_list, n_neighbours_list)):
        print("t, alpha, n_neighbours:", t, alpha, n_neighbours, file=open("output.txt", "a"))
        print(t, alpha, n_neighbours, file=open("means.txt", "a"))
        parallel_annealing(4, t, alpha, n_neighbours)


if __name__ == '__main__':
    #parallel_annealing(4, 500, 0.1, 10)
    test_params()

#Todo Probar si merece la pena empezar con una buena solución de un genético
#Todo Búsqueda local al final
#Todo 30 segundos y otros 30 empezando con el mejor (maybe busqueda local por medio)
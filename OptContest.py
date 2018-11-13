import time
from multiprocessing import Pool
from Annealing import *
from Evolutive import *
from RandomAlgorithm import *

def worker(mutation):
    pop = generate_random_population(75, 100)
    return evolutive_algorithm(read_file("Datasets/Doc11.txt"), pop, pop_size=100,
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


if __name__ == '__main__':
    solution0 = create_random_solution(75)
    print(simulated_annealing(5000, 1, solution0, 60, 5, read_file("Datasets/Doc11.txt")))

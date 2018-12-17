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
    results_ = p.map(evolutive_worker, [(i, mutation) for i in range(n_processes)])
    p.close()
    return results_


def annealing_worker(params):
    tries, alpha, solution0, n_neighbours, time_ = params[1]
    return simulated_annealing(tries, alpha, solution0, time_, n_neighbours, read_file("Datasets/Doc11.txt"))


def parallel_annealing(n_processes, tries, alpha, n_neighbours, time_):
    p = Pool(processes=n_processes)

    solution0 = create_random_solution(75)
    params = [tries, alpha, solution0, n_neighbours, time_]
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
    for _ in range(1):
        time_ini = time.time()
        annealing_results = parallel_annealing(2, 5, 5, 5, 50)
        results.append(local_best_first_search(*annealing_results, read_file("Datasets/Doc11.txt"), 10))
        time_fin = time.time()
        print(results)
        print(time_fin-time_ini)

    #print("Mean fmed:", np.mean(results))
    #print("Best fmed:", np.min(results))

#TODO Parametrizar bien y probar otros métodos para bajar la temperatura
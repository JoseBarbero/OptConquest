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
    tries, t_ini_factor, alpha, solution0, n_neighbours, time_ = params[1]
    return simulated_annealing(tries, t_ini_factor, alpha, solution0, time_, n_neighbours, read_file("Datasets/Doc11.txt"))


def parallel_annealing(n_processes, tries, t_ini_factor, alpha, n_neighbours, time_, solution0):
    p = Pool(processes=n_processes)


    params = [tries, t_ini_factor, alpha, solution0, n_neighbours, time_]
    results_ = p.map(annealing_worker, [(i, params) for i in range(n_processes)])

    solutions, fmeds = zip(*results_)
    p.close()
    idx_best = min(enumerate(fmeds), key=itemgetter(1))[0]

    #print(fmeds, "--->", min(fmeds))

    return results_[idx_best]


def test_params():
    t_factor = [0.1, 0.5, 0.9]
    alpha_list = [1, 10, 20]
    n_neighbours_list = [3, 5, 10]
    n_tries_list = [5, 50]
    times = [(60, 0), (50, 10), (30, 30)]
    searches_list = [local_best_search, local_best_first_search]

    for t_factor, alpha, n_neighbours, n_tries, times_, search_method in list(itertools.product(t_factor, alpha_list, n_neighbours_list, n_tries_list, times, searches_list)):
        print("t_factor, alpha, n_neighbours, n_tries, times_, search_method: ", t_factor, alpha, n_neighbours, n_tries, times_, search_method, file=open("output.txt", "a"))
        annealing_results = parallel_annealing(3, n_tries, t_factor, alpha, n_neighbours, times_[0])
        searcb_result = local_best_first_search(*annealing_results, read_file("Datasets/Doc11.txt"), times_[1])
        print(annealing_results, file=open("output.txt", "a"))
        print(searcb_result, file=open("output.txt", "a"))
        print("", file=open("output.txt", "a"))


if __name__ == '__main__':



    results = []

    for _ in range(5):
        #time_ini = time.time()
        solution0 = create_random_solution(75)
        solution1, solution1_fmed = parallel_annealing(3, 5, 0.9, 20, 3, 60, solution0)
        solution2, solution2_fmed = parallel_annealing(3, 1, 0.01, 50, 1, 0, solution1)
        if solution1_fmed < solution2_fmed:
            results.append(solution1_fmed)
        else:
            results.append(solution2_fmed)
        print(solution1_fmed, solution2_fmed)
        #time_fin = time.time()
        #print(time_fin-time_ini)
    print(results)
    print("Mean fmed:", np.mean(results))
    print("Best fmed:", np.min(results))

#Todo añadir en el recocido la condición para que se mantenga el mejor
import sys
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


def evolutive_worker(params):
    mutation, ini_sol, time_, data_ = params[1]
    pop_ = Population(len(ini_sol[0]), 50, data_, ini_sol)
    return evolutive_algorithm(data_, pop_,
                          time_=time_, elite_size=1, mut_ratio=mutation, diversify_size=0, not_improving_limit=False,
                          sel_f=median_selection,
                          elite_f=get_elite,
                          rep_f=ox_reproduction,
                          mut_f=mutate)


def parallel_evolutive(n_processes, mutation, ini_sol, time_, data_):
    p = Pool(processes=n_processes)
    params = mutation, ini_sol, time_, data_
    results_ = p.map(evolutive_worker, [(i, params) for i in range(n_processes)])

    solutions, fmeds = zip(*results_)
    p.close()
    idx_best = min(enumerate(fmeds), key=itemgetter(1))[0]

    return results_[idx_best]


def annealing_worker(params):
    tries, t_ini_factor, alpha, solution0, time_, data_ = params[1]
    return simulated_annealing(tries, t_ini_factor, alpha, solution0, time_, data_)


def parallel_annealing(n_processes, tries, t_ini_factor, alpha, time_, solution0, data_):
    p = Pool(processes=n_processes)

    params = [tries, t_ini_factor, alpha, solution0, time_, data_]
    results_ = p.map(annealing_worker, [(i, params) for i in range(n_processes)])

    solutions, fmeds = zip(*results_)
    p.close()
    idx_best = min(enumerate(fmeds), key=itemgetter(1))[0]

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
    time_ini = time.time()
    file_path = sys.argv[1]
    data = read_file(file_path)

    solution0 = create_random_solution(len(data))
    solution, solution_fmed = parallel_annealing(4, 10, 0.6, 5, 60, solution0, data)

    time_fin = time.time()

    print("Best fmed:", solution_fmed)
    print("Execution time:", time_fin-time_ini)
    print("Best solution:", solution)
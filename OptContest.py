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


# def annealing_worker(params):
#     tries, t_ini_factor, alpha, initial_solution, time_, data_ = params
#     return simulated_annealing(tries, t_ini_factor, alpha, initial_solution, time_, data_)

def annealing_worker(params):
    tries, t_ini_factor, alpha, initial_solution, time_, data_ = params
    presolution, presolution_fmed = simulated_annealing(tries, t_ini_factor, alpha, initial_solution, time_, data_)
    solution, solution_fmed = local_best_search(presolution, presolution_fmed, data, 60-time_) # El último parámetro es el tiempo de búsqueda local
    print("Prelocal: ", presolution_fmed, "Postlocal:", solution_fmed)
    return solution, solution_fmed


def parallel_annealing(n_processes, tries, t_ini_factor, alpha, time_, data_):
    p = Pool(processes=n_processes)

    params = []
    for process in range(n_processes):
        params.append([tries, t_ini_factor, alpha, create_random_solution(len(data_)), time_, data_])

    results_ = p.map(annealing_worker, params)

    solutions, fmeds = zip(*results_)

    # print(fmeds, file=open("output.txt", "a"))
    # print(np.mean(fmeds), file=open("output.txt", "a"))
    # print(tries, t_ini_factor, alpha, file=open("output.txt", "a"))
    # print("", file=open("output.txt", "a"))
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


# if __name__ == '__main__':
#     time_ini = time.time()
#     file_path = sys.argv[1]
#     data = read_file(file_path)

#     # solution, solution_fmed = parallel_annealing(4, 10, 0.6, 5, 54, data)
#     # solution, solution_fmed = parallel_annealing(4, 5, 0.1, 1, 54, data) 
#     # solution, solution_fmed = parallel_annealing(4, 5, 0.2, 0.5, 54, data) 
#     solution, solution_fmed = parallel_annealing(4, 5, 0.5, 0.5, 53, data) 

#     time_fin = time.time()

#     print("Best fmed:", solution_fmed)
#     print("Execution time:", time_fin-time_ini)
#     print("Best solution:", solution)


if __name__ == '__main__':
    time_ini = time.time()
    file_path = sys.argv[1]
    data = read_file(file_path)

    # solution, solution_fmed = parallel_annealing(4, 10, 0.6, 5, 53, data)
    solution, solution_fmed = parallel_annealing(4, 100, 0.6, 5, 53, data)
    time_fin = time.time()
    print("Params: 10, 0.5, 5")
    print("Best fmed:", solution_fmed)
    print("Execution time:", time_fin-time_ini)
    print("")


# if __name__ == '__main__':
#     time_ini = time.time()
#     file_path = sys.argv[1]
#     data = read_file(file_path)
    
#     tries_list = [1, 5, 10, 15, 20]
#     t_ini_list = [0.1, 0.2, 0.5, 1, 5, 10]
#     alpha_list = [0.1, 0.5, 1, 5, 10]

#     for tries in tries_list:
#         for t_ini in t_ini_list:
#             for alpha in alpha_list:
#                 params = [4, tries, t_ini, alpha, 54, data]
#                 solution, solution_fmed = parallel_annealing(*params)

#     time_fin = time.time()
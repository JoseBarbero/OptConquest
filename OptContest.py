import time
from multiprocessing import Pool
from Evolutive import *
from RandomAlgorithm import *


def worker(mutation):
    # TODO devolver la solución en sí
    return evolutive_algorithm(read_file("Datasets/Doc11.txt"), pop_size=100,
                          time_=60, elite_size=1, mut_ratio=mutation[1], diversify_size=0, not_improving_limit=0,
                          sel_f=median_selection,
                          elite_f=get_elite,
                          rep_f=ox_reproduction,
                          mut_f=mutate)[0]


def parallel_evolutive(n_processes, mutation):
    p = Pool(processes=n_processes)
    results = p.map(worker, [(i, mutation) for i in range(n_processes)])
    p.close()
    return min(results)


if __name__ == '__main__':
    res = []
    for _ in range(5):
        res.append(parallel_evolutive(4, 25))
    print(f"\tMean: {sum(res)/len(res)}")
    print(f"\tBest: {min(res)}")

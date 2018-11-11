from Evolutive import *
from RandomAlgorithm import *
from multiprocessing import Process

results = []

for mut_perc in [50]:
    print(f"% Mutación: {mut_perc} ")
    for _ in range(5):
        results.append(evolutive_algorithm(read_file("Datasets/Doc11.txt"), pop_size=100,
                              time_=60, elite_size=1, mut_ratio=mut_perc, diversify_size=0, not_improving_limit=0,
                              sel_f=median_selection,
                              elite_f=get_elite,
                              rep_f=ox_reproduction,
                              mut_f=mutate)[0])
    print(f"\tMedia: {sum(results)/len(results)}")
    print(f"\tMínimo: {min(results)}")
    print("\n")

"""
if __name__ == 'main':
    p1 = Process(target=evolutive_algorithm(read_file("Datasets/Doc11.txt"), pop_size=100,
                              time_=60, elite_size=1, mut_ratio=mut_perc, diversify_size=0, not_improving_limit=0,
                              sel_f=median_selection,
                              elite_f=get_elite,
                              rep_f=ox_reproduction,
                              mut_f=mutate))

    p2 = Process(target=evolutive_algorithm(read_file("Datasets/Doc11.txt"), pop_size=100,
                                            time_=60, elite_size=1, mut_ratio=mut_perc, diversify_size=0,
                                            not_improving_limit=0,
                                            sel_f=median_selection,
                                            elite_f=get_elite,
                                            rep_f=ox_reproduction,
                                            mut_f=mutate))

    p3 = Process(target=evolutive_algorithm(read_file("Datasets/Doc11.txt"), pop_size=100,
                                            time_=60, elite_size=1, mut_ratio=mut_perc, diversify_size=0,
                                            not_improving_limit=0,
                                            sel_f=median_selection,
                                            elite_f=get_elite,
                                            rep_f=ox_reproduction,
                                            mut_f=mutate))

    p4 = Process(target=evolutive_algorithm(read_file("Datasets/Doc11.txt"), pop_size=100,
                                            time_=60, elite_size=1, mut_ratio=mut_perc, diversify_size=0,
                                            not_improving_limit=0,
                                            sel_f=median_selection,
                                            elite_f=get_elite,
                                            rep_f=ox_reproduction,
                                            mut_f=mutate))
                                            """
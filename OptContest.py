from Evolutive import *
from RandomAlgorithm import *


# ini = time.time()
# print(evaluate_algorithm(evolutive_algorithm, [read_file("Datasets/Doc11.txt"), 100]))
print(evolutive_algorithm(read_file("Datasets/Doc11.txt"), pop_size=150,
                          time_=60, elite_size=5, mut_ratio=0.05, diversify_size=0, not_improving_limit=3,
                          sel_f=median_selection,
                          elite_f=get_elite,
                          rep_f=ox_reproduction,
                          mut_f=mutate))
# fin = time.time()
# print("Time spend:", fin-ini)

# print(find_best_params(read_file("Datasets/Doc11.txt")))

# print(random_algorithm(read_file("Datasets/Doc11.txt"), 10000))
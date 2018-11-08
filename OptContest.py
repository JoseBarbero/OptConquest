from Evolutive import *
from RandomAlgorithm import *


# ini = time.time()
# print(evaluate_algorithm(evolutive_algorithm, [read_file("Datasets/Doc11.txt"), 100]))
# print(evolutive_algorithm(read_file("Datasets/Doc11.txt"), 50, elite_size=1, mut_ratio=10))
# fin = time.time()
# print("Time spend:", fin-ini)

print(find_best_params(read_file("Datasets/Doc11.txt")))

# print(random_algorithm(read_file("Datasets/Doc11.txt"), 10000))
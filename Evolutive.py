import operator
import time
import itertools
import pandas as pd
from RandomAlgorithm import *
from FlowShopUtils import *


def generate_random_population(sol_size, pop_size):
    """
    Genera una población de soluciones aleatorias.
    """
    return [create_random_solution(sol_size) for _ in range(pop_size)]


def mean_selection(population, data):
    """
    Nos quedamos con los que son mejores que la media de los fmed de toda la población.
    """
    pop_fmed = [fmed(f(solution, data)) for solution in population]
    test = [sol for sol, f_med in zip(population, pop_fmed) if f_med <= np.median(pop_fmed)]
    return test


def basic_reproduction(population, pop_size):
    """

    """
    # Idea: reproducir más veces a los que más fmed tengan
    # De momento se rellenan al azar con copias de lo que ya hay
    # Deberían mezclarse fragmentos de una solución con fragmentos de otra
    test = [population[i] for i in np.random.choice(len(population), pop_size - len(population))]
    population.extend(test)
    return population


def ox_reproduction(population, target_len, elite_size):
    """
    Genera una nueva población con hijos de los seleccionados
    """
    pop_len = len(population)  # Se calcula antes porque la población va creciendo
    new_pop = []
    for i in range(elite_size, target_len):
        indexes = np.random.choice(pop_len, 2)
        new_pop.append(cruce_pseudo_ox(population[indexes[0]], population[indexes[1]]))
    return new_pop


def cruce_pseudo_ox(parent1, parent2):
    # Cojo dos índices aleatorios entre los cuales se va a mantener el contenido del parent1
    index_ini, index_fin = sorted(np.random.choice(range(len(parent1)), 2, replace=False))
    p1_aux = parent1[index_ini:index_fin]

    # Máscara para sacar los elementos del parent2 que no han sido ya seleccionados del 1
    p2_aux = parent2[np.logical_not(np.isin(parent2, p1_aux))]

    return np.array([*p2_aux[:index_ini], *p1_aux, *p2_aux[index_ini:]])

    # PROBAR CON MÁSCARAS. LA IDEA ES METER LOS DEL PRIMERO Y LOS QUE NO SEAN ESOS DEL SEGUNDO


def get_best_solution(pop, data):
    """
    Obtiene la mejor solución de una población.
    """
    return min([(fmed(f(solution, data)), solution) for solution in pop], key=operator.itemgetter(0))


def get_n_best_solutions(pop, data, n_top):
    elite = sorted([(fmed(f(solution, data)), solution) for solution in pop], key=operator.itemgetter(0))[:n_top]
    return [x[1] for x in elite]


def mutate(pop, data, ratio):
    """
    Produce intercambios entre dos posiciones aleatorias de cada solución menos la mejor.
    """
    best = get_best_solution(pop, data)[1].copy()  # Si no copiamos se modifica después
    for i in range(len(pop)):
        if np.random.randint(100) < ratio:  # Mutan solo un 5%
            swap_indexes = np.random.choice(pop[i], 2)
            pop[i][swap_indexes[0]], pop[i][swap_indexes[1]] = pop[i][swap_indexes[1]], pop[i][swap_indexes[0]]
    # Me cargo la primera solución para meter la mejor
    # porque calcular el índice de la mejor puede resultar demasiado lento
    pop[0] = best
    return pop


def get_elite(pop, data, size):
    if size == 1:
        return [get_best_solution(pop, data)[1]]
    else:
        return get_n_best_solutions(pop, data, size)


def evolutive_algorithm(data, pop_size, time_=60, elite_size=5, mut_ratio=10):
    # 1. Crear población
    pop = generate_random_population(len(data), pop_size)

    # 2. Bucle de evolución (mientras no se alcance la condición de parada)

    t_end = time.time() + time_ - 0.5
    # for i in range(generations):
    while time.time() < t_end:

        pop = evolutive_generation(data, pop, pop_size, elite_size, mut_ratio)

        # print('\r' + str(i + 1) + "/" + str(generations), end='')
        # print(get_best_solution(pop, data))

    # 3. Retornar mejor solución
    return get_best_solution(pop, data)


def evolutive_generation(data, pop, pop_size, elite_size, mut_ratio):
    # 2.1. Selección
    pop = mean_selection(pop, data)

    # 2.2. Elitismo
    elite = get_elite(pop, data, elite_size)

    # 2.2. Reproducción
    pop = ox_reproduction(pop, pop_size, elite_size)

    # 2.4. Mutación
    pop = mutate(pop, data, mut_ratio)

    # 2.5. Combinar élite con el resto
    pop.extend(elite)

    return pop


def find_best_params(dataset):
    """
    Realiza ejecuciones con diferentes parámetros para encontrar los más óptimos para cierto dataset.
    """
    pop_size_list = range(100, 301, 50)
    elite_size = [0, 1, 3, 5, 10]
    mut_ratio = [0.01, 0.05, 0.1, 1, 3]
    best_params = []
    lowest_fmed = np.inf

    col_names = ['Population Size', 'Elite Size', 'Mutation %', 'Fmed']
    results_df = pd.DataFrame(columns=col_names)

    for p_size, n_elite, mut_perc in list(itertools.product(pop_size_list, elite_size, mut_ratio)):
        temp_fmed, temp_min_fmed = evaluate_algorithm(evolutive_algorithm, (dataset, p_size, 60, n_elite, mut_perc))
        results_df.loc[len(results_df)] = [p_size, n_elite, mut_perc, temp_fmed]
        print(f"Pop. Size: {p_size}, "
              f"N. Elite:{n_elite}, "
              f"Mut %: {mut_perc}, "
              f"Mean fmed: {temp_fmed}, "
              f"Best fmed {temp_min_fmed}")
        if temp_fmed < lowest_fmed:
            best_params = [p_size, n_elite, mut_perc]
        results_df.to_csv("fmedByParams.csv", sep='\t', encoding='utf-8')
    print("Best params:", best_params)
    return best_params

# Mejoras
# . Evitar que la mejor solución se mantenga igual (pasa en varias generaciones, sobretodo al final)
# . Función para evaluar cómo de buena es una solución
# . Calcular los parámetros tamaño/generaciones más óptimos para ese tamaño de problema
# . Crear clase para una población
# . Correr en paralelo
# . Hacer una especie de elitismo inverso en el que todas las generaciones se meta algo aleatorio nuevo
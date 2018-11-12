import operator
import time
import itertools
import pandas as pd
import random as rnd
from RandomAlgorithm import *
from FlowShopUtils import *


def generate_random_population(sol_size, pop_size):
    """
    Genera una población de soluciones aleatorias.
    """
    return [create_random_solution(sol_size) for _ in range(pop_size)]


def median_selection(population, data, alpha=1):
    """
    Nos quedamos con los que son mejores que la mediana*alpha de los fmed de toda la población.
    """
    pop_fmed = [fmed(f(solution, data)) for solution in population]
    median_fmed = np.median(pop_fmed)*alpha
    pop = []
    best_fmed = np.inf  # TODO permitir más de uno en la élite
    for solution, solution_fmed in zip(population, pop_fmed):
        if solution_fmed <= median_fmed:
            pop.append(solution)
            if solution_fmed < best_fmed:
                best_fmed = solution_fmed
                best_solution = solution
    return pop, best_solution


def old_median_selection(population, data):
    """
    Nos quedamos con los que son mejores que la mediana*alpha de los fmed de toda la población.
    """
    pop_fmed = [fmed(f(solution, data)) for solution in population]
    pop = [sol for sol, f_med in zip(population, pop_fmed) if f_med <= np.median(pop_fmed)]
    return pop



def tournament_selection(population, data, wanted_size=30, p=2):
    """
    Selección por torneo determinista.
    """
    pop = []
    best_fmed = np.inf
    for _ in range(wanted_size):
        winner_fmed, winner = get_best_solution(rnd.sample(population, p), data)
        pop.append(winner)
        if winner_fmed <= best_fmed:
            best_solution = winner
    return pop, best_solution


def random_selection(population, _, wanted_size=50):
    return rnd.sample(population, wanted_size)


def basic_reproduction(population, pop_size):
    test = [population[i] for i in np.random.choice(len(population), pop_size - len(population))]
    population.extend(test)
    return population


def ox_reproduction(population, target_len, elite_size):
    """
    Genera una nueva población con hijos de los seleccionados
    """
    pop_len = len(population)
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


def get_best_solution(pop, data):
    """
    Obtiene la mejor solución de una población.
    """
    #TODO esto es lo más costoso de todo el algoritmo, hay que optimizar
    return min([(fmed(f(solution, data)), solution) for solution in pop], key=operator.itemgetter(0))


def get_n_best_solutions(pop, data, n_top):
    elite = sorted([(fmed(f(solution, data)), solution) for solution in pop], key=operator.itemgetter(0))[:n_top]
    return [x[1] for x in elite]


def mutate(pop, ratio):
    """
    Produce intercambios entre dos posiciones aleatorias de cada solución.
    """
    #TODO mutar dentro del cruce para evitar otro bucle
    #TODO otros métodos de mutación
    for i in range(len(pop)):
        if np.random.randint(100) < ratio:
            swap_indexes = np.random.choice(pop[i], 2)
            pop[i][swap_indexes[0]], pop[i][swap_indexes[1]] = pop[i][swap_indexes[1]], pop[i][swap_indexes[0]]
    return pop


def get_elite(pop, data, size):
    if size == 1:  # Más eficiente que ordenar toda la población
        return [get_best_solution(pop, data)[1]]
    elif size > 1:
        return get_n_best_solutions(pop, data, size)
    else:
        return []


def get_elite_with_rep(pop, data, size):
    """
    Seleccionamos la élite y la reproducimos para tener asegurados hijos de la élite en la siguiente.
    """
    elite = get_elite(pop, data, size)
    if size < 2:
        elite.extend(elite)
    elite.extend(ox_reproduction(pop, size * 2, size))
    return elite


def evolutive_algorithm(data, pop_size, time_=60, elite_size=5, mut_ratio=10, diversify_size=0, not_improving_limit=5,
                        sel_f=median_selection, elite_f=get_elite, rep_f=ox_reproduction, mut_f=mutate):
    # 1. Crear población
    pop = generate_random_population(len(data), pop_size)

    # 2. Bucle de evolución (mientras no se alcance la condición de parada)

    t_end = time.time() + time_ - 0.5
    best = np.inf
    while time.time() < t_end:

        pop = evolutive_generation(data, pop, pop_size, elite_size, mut_ratio, diversify_size,
                                   sel_f, elite_f, rep_f, mut_f)
        if not_improving_limit:
            current_fmed = get_best_solution(pop, data)[0]
            if current_fmed < best:
                best = current_fmed
                not_improving = 0
            else:
                if not_improving < not_improving_limit:
                    not_improving += 1
                else:   # Si no mejoramos en x generaciones metemos parámetros más agresivos
                    # Los cálculos controlan que no se pase del tamaño de la población
                    mut_ratio += (100-mut_ratio)/10
                    diversify_size += int((len(pop)-diversify_size)/10)
                    not_improving = 0

        # print('\r' + str(i + 1) + "/" + str(generations), end='')
        # print(get_best_solution(pop, data))

    # 3. Retornar mejor solución
    return get_best_solution(pop, data)


def diversify(pop, target_size):
    """
    Añade población random para evitar mínimos locales.
    """
    pop.extend(generate_random_population(len(pop[0]), target_size-len(pop)))
    return pop


def evolutive_generation(data, pop, pop_size, elite_size, mut_ratio, diversify_size, sel_f, elite_f, rep_f, mut_f):
    #step0 = time.time()
    # 2.1. Selección
    pop, elite = sel_f(pop, data)

    #step1 = time.time()

    # 2.2. Elitismo
    # elite = elite_f(pop, data, elite_size)

    #step2 = time.time()

    # 2.2. Reproducción
    pop = rep_f(pop, pop_size-diversify_size, elite_size)

    #step3 = time.time()

    # 2.4. Mutación
    pop = mut_f(pop, mut_ratio)

    #step4 = time.time()

    # 2.5. Diversificación
    pop = diversify(pop, pop_size)

    #step5 = time.time()

    # 2.5. Combinar élite con el resto
    pop.append(elite)

    #step6 = time.time()

    #print(f"%Selección: {(step1-step0)/(step6-step0)*100} \t "
    #      f"%Elitismo: {(step2-step1)/(step6-step0)*100} \t "
    #      f"%Reproducción: {(step3-step2)/(step6-step0)*100} \t "
    #      f"%Mutación: {(step4-step3)/(step6-step0)*100} \t "
    #      f"%Diversificación: {(step5-step4)/(step6-step0)*100} \t "
    #      f"%Combinar: {(step6-step5)/(step6-step0)*100}")

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
            lowest_fmed = temp_fmed
        results_df.to_csv("fmedByParams.csv", sep='\t', encoding='utf-8')
    print("Best params:", best_params)
    return best_params

# Mejoras
# . Crear clase para una población
# . Probar cambiando arrays/lists
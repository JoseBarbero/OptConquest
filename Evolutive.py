import operator
import time
import itertools
import pandas as pd
import random as rnd
from RandomAlgorithm import *
from FlowShopUtils import *
from multiprocessing import Pool


class Population:
    def __init__(self, solution_size, population_size, problem_data, ini_sol=False):
        """
        Utilizo una biblioteca de fmeds para no tener que volver a calcularlos de nuevo (que es de lo más costoso)
        """
        self.len = population_size
        self.fmed_library = {}
        self.best_solution = None
        self.best_fmed = np.inf
        self.fmed_sum = 0
        self.problem_data = problem_data
        # Separo en dos listas porque para varios cálculos tendría que separar los fmeds (los idx son los mismos)
        self.population, self.population_fmeds = self.initialize_population(solution_size, population_size, ini_sol)

    def add_to_library(self, solution):
        """
        Añade el fmed de una solución a la biblioteca y lo devuelve (para tenerlo directamente en get_fmed sin buscarlo)
        """
        fmed_value = fmed(solution, self.problem_data)
        if fmed_value < self.best_fmed:
            self.best_fmed = fmed_value
            self.best_solution = solution
        # Uso .toBytes porque no se puede usar un array como key de diccionario
        # (se podría usar str() pero no sé cómo es de eficiente)
        self.fmed_library[solution.tobytes()] = fmed_value
        return fmed_value

    def get_fmed(self, solution):
        """
        Saca el fmed de cierta solución de la biblioteca. Si no lo encuentra, lo mete en la biblioteca y lo devuelve.
        """
        #return self.fmed_library.get(solution.tobytes(), self.add_to_library(solution))
        return fmed(solution, self.problem_data)

    def initialize_population(self, sol_size, pop_size, ini_sol=False):
        """
        Inicializa la población de forma aleatoria y calcula los fmeds para la biblioteca.
        """
        population = []
        population_fmeds = []
        if ini_sol:
            population.append(ini_sol[0])
            population_fmeds.append(ini_sol[1])
        for _ in range(pop_size):
            solution = create_random_solution(sol_size)
            current_fmed = self.get_fmed(solution)
            population.append(solution)
            population_fmeds.append(current_fmed)
            self.fmed_sum += current_fmed
            if current_fmed < self.best_fmed:
                self.best_fmed = current_fmed
                self.best_solution = solution
        return population, population_fmeds

    def get_mean(self):
        return self.fmed_sum / self.size

    def get_median(self):
        # Todo Buscar manera de hacer esto más eficiente manteniendo siempre la mediana sin tener que recalcular
        return np.median(self.population_fmeds)

    def clear_population(self):
        self.population = []
        self.population_fmeds = []
        self.size = 0
        self.best_solution = None
        self.best_fmed = np.inf
        self.fmed_sum = 0

    def replace_full_population(self, new_pop, new_pop_fmeds, new_size, new_best_solution, new_best_fmed, new_fmed_sum):
        self.population = new_pop
        self.population_fmeds = new_pop_fmeds
        self.size = new_size
        self.best_solution = new_best_solution
        self.best_fmed = new_best_fmed
        self.fmed_sum = new_fmed_sum

    def replace_population(self, new_pop):
        self.clear_population()
        self.population = new_pop
        for solution in new_pop:
            current_fmed = self.get_fmed(solution)
            self.population_fmeds.append(current_fmed)
            self.fmed_sum += current_fmed
            if current_fmed < self.best_fmed:
                self.best_fmed = current_fmed
                self.best_solution = solution


def generate_random_population(sol_size, pop_size):
    return [create_random_solution(sol_size) for _ in range(pop_size)]


def median_selection(population):
    """
    Nos quedamos con los que son mejores que la mediana*alpha de los fmed de toda la población.
    """
    pop_fmed = population.population_fmeds
    median_fmed = population.get_median()

    new_pop = []
    new_pop_fmeds = []
    new_fmed_sum = 0

    for solution, solution_fmed in zip(population.population, pop_fmed):
        if solution_fmed <= median_fmed:
            new_pop.append(solution)
            new_pop_fmeds.append(solution_fmed)
            new_fmed_sum += solution_fmed

    population.replace_full_population(new_pop, new_pop_fmeds, len(new_pop),
                                       population.best_solution, population.best_fmed, new_fmed_sum)
    return population


def tournament_selection(population, data, wanted_size=30, p=2, ):
    """
    Selección por torneo determinista.
    """
    # Todo cambiar al nuevo sistema de población
    pop = []
    best_fmed = np.inf
    for _ in range(wanted_size):
        winner_fmed, winner = get_best_solution(rnd.sample(population, p), data)
        pop.append(winner)
        if winner_fmed <= best_fmed:
            best_solution = winner
    return pop, best_solution


def random_selection(population, _, wanted_size=50):
    # Todo cambiar al nuevo sistema de población
    return rnd.sample(population, wanted_size)


def basic_reproduction(population, pop_size):
    # Todo cambiar al nuevo sistema de población
    test = [population[i] for i in np.random.choice(len(population), pop_size - len(population))]
    population.extend(test)
    return population


def ox_reproduction(population, target_len, elite_size, mut_prob):
    """
    Genera una nueva población con hijos de los seleccionados
    """
    pop_len = population.size
    new_pop = []
    for i in range(elite_size, target_len):
        indexes = np.random.choice(pop_len, 2)
        son = cruce_pseudo_ox(population.population[indexes[0]], population.population[indexes[1]])

        # Se muta aquí para evitar otro bucle en la mutación
        new_pop.append(swap_mutation(son, mut_prob))
    return new_pop


def cruce_pseudo_ox(parent1, parent2):
    # Todo cambiar al nuevo sistema de población
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
    # Todo cambiar al nuevo sistema de población
    #TODO esto es lo más costoso de todo el algoritmo, hay que optimizar
    return min([(fmed(solution, data), solution) for solution in pop], key=operator.itemgetter(0))


def get_n_best_solutions(pop, data, n_top):
    # Todo cambiar al nuevo sistema de población
    elite = sorted([(fmed(solution, data), solution) for solution in pop], key=operator.itemgetter(0))[:n_top]
    return [x[1] for x in elite]


def swap_mutation(solution, ratio, number_of_swaps=1):
    # Todo cambiar al nuevo sistema de población
    solution = solution.copy()
    for _ in range(number_of_swaps):
        if np.random.randint(100) < ratio:
            swap_indexes = np.random.choice(len(solution), 2, replace=False)
            solution[swap_indexes[0]], solution[swap_indexes[1]] = solution[swap_indexes[1]], solution[swap_indexes[0]]
    return solution


def mutate(pop, ratio):
    """
    Produce intercambios entre dos posiciones aleatorias de cada solución.
    """
    # Todo cambiar al nuevo sistema de población
    #TODO mutar dentro del cruce para evitar otro bucle
    #TODO otros métodos de mutación
    for i in range(len(pop)):
        if np.random.randint(100) < ratio:
            swap_indexes = np.random.choice(pop[i], 2)
            pop[i][swap_indexes[0]], pop[i][swap_indexes[1]] = pop[i][swap_indexes[1]], pop[i][swap_indexes[0]]
    return pop


def get_elite(pop, data, size):
    if size == 1:  # Más eficiente que ordenar toda la población
        return pop.best_solution
    elif size > 1:
        return get_n_best_solutions(pop, data, size)
    else:
        return []


def get_elite_with_rep(pop, data, size):
    """
    Seleccionamos la élite y la reproducimos para tener asegurados hijos de la élite en la siguiente.
    """
    # Todo cambiar al nuevo sistema de población
    elite = get_elite(pop, data, size)
    if size < 2:
        elite.extend(elite)
    elite.extend(ox_reproduction(pop, size * 2, size))
    return elite


def evolutive_algorithm(data, pop, time_=60, elite_size=5, mut_ratio=10, diversify_size=0, not_improving_limit=5,
                        sel_f=median_selection, elite_f=get_elite, rep_f=ox_reproduction, mut_f=mutate):

    # 2. Bucle de evolución (mientras no se alcance la condición de parada)

    t_end = time.time() + time_ - 0.5
    #best = np.inf
    while time.time() < t_end:
        pop, elite = evolutive_generation(pop, elite_size, mut_ratio, diversify_size,
                                   sel_f, elite_f, rep_f, mut_f)
        """
        if not_improving_limit:
            current_fmed = pop.best_fmed
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

        #print(get_best_solution(pop, data))
        """

    # 3. Retornar mejor solución
    return elite


def evolutive_generation(pop, elite_size, mut_ratio, diversify_size, sel_f, elite_f, rep_f, mut_f):
    # Elitismo
    elite = pop.best_solution

    #Selección
    pop = sel_f(pop)

    #step1 = time.time()

    # 2.2. Elitismo
    # elite = elite_f(pop, data, elite_size)

    #step2 = time.time()

    # 2.2. Reproducción
    new_solutions = rep_f(pop, pop.len-diversify_size, elite_size, mut_ratio)
    #step3 = time.time()

    # 2.4. Mutación
    #pop = mut_f(pop, mut_ratio)

    #step4 = time.time()

    # 2.5. Diversificación
    #pop = diversify(pop, diversify_size)
    #step5 = time.time()

    # 2.5. Combinar élite con el resto
    new_solutions.append(elite)

    pop.replace_population(new_solutions)

    #step6 = time.time()

    #print(f"%Selección: {(step1-step0)/(step6-step0)*100} \t "
    #      f"%Elitismo: {(step2-step1)/(step6-step0)*100} \t "
    #      f"%Reproducción: {(step3-step2)/(step6-step0)*100} \t "
    #      f"%Mutación: {(step4-step3)/(step6-step0)*100} \t "
    #      f"%Diversificación: {(step5-step4)/(step6-step0)*100} \t "
    #      f"%Combinar: {(step6-step5)/(step6-step0)*100}")
    print(pop.best_fmed, file=open("evolutive_fmeds.txt", "a"))
    return pop, (elite, pop.best_fmed)


#def diversify(pop, div_size):
#    """
#    Añade población random para evitar mínimos locales.
#    """
#    pop.extend(generate_random_population(len(pop[0]), div_size))
#    return pop


def parallel_median_selection(population, data):
    # TODO hacer esto más limpio
    chunks = [population[:25], population[25:50], population[50:75], population[75:]]
    p = Pool(processes=len(chunks))
    selected = []
    elite = []
    results = p.map(median_selection, [(chunk, data) for chunk in chunks])
    p.close()
    for result in results:
        selected.extend(result[0])
        elite.append(result[1])
    best = min(elite, key=operator.itemgetter(0))[1]
    return selected, best


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

# Todo buscar el mejor tamaño de la población
# Todo Aumentar el tamaño de la población a medida que pasan las iteraciones
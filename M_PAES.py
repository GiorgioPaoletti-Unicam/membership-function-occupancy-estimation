import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from FCM import ConcreteFCM
from FLCGenerator import ConcreteFLC
from FMParamsCalculator import *
from FRSGenerator import *
from FVTGenerator import *

# Input universes
"""input_universes = {
    'Temperature': np.arange(18, 30, 0.01),
    'Humidity': np.arange(14, 45, 0.01),
    'Light': np.arange(0, 1800, 0.01),
    'CO2': np.arange(300, 2200, 0.01),
    'HumidityRatio': np.arange(0.01, 0.1, 0.001)
}"""
input_universes = {
    'Temp': np.arange(18, 30, 0.01),
    'Light': np.arange(0, 1800, 0.01),
    'Sound': np.arange(0.0, 4.0, 0.01),
    # 'PIR': np.arange(0, 1, 1), # 0 or 1 used directly in the rule set
    'CO2': np.arange(300, 2200, 0.01),
}

# Output universe
output_universe = np.arange(0, 101, 1)

# Maximum occupancy of the room
max_occupancy = 3

# FMC parameters
C = 3  # Number of clusters
F = len(input_universes)  # Number of input variables

# Constants for your problem
GAMMA_MAX = 5  # Upper bound for γ
DELTA_MAX = 3  # Upper bound for δ

# F_MAX = 5  # Maximum number of input variables (Fmax)
F_MAX = F  # Maximum number of input variables (Fmax)

# setted to cover the entire range of possible rules
# AssertionError: Total area is zero in defuzzification!
# ValueError: Crisp output cannot be calculated, likely because the system is too sparse.
# Check to make sure this set of input values will activate at least one connected Term in each Antecedent
# via the current set of Rules.
M_MIN = 10  # Minimum number of rules (Mmin)
M_MAX = 50  # Maximum number of rules (Mmax)
M_INIT = 25  # Initial number of rules (Minit)

ARCHIVE_SIZE = 50  # Size of the archive

total_generations = 1000  # Number of generations


class Chromosome:

    # def __init__(self, mmax, m_init, v):
    def __init__(self):
        # costructor for the chromosome class

        # J: define the RB
        # the RB of an MFRBS can be completely described by a matrix J ∈ R(Mx(F+1)), where the generic element
        # indicates that fuzzy set A_f,j_m,f has been selected for variable X_f in rule Rm.
        self._J = np.random.randint(1, C + 1, size=(M_INIT, F + 1))

        # Eliminate duplicate rules immediately after generation
        # We convert the array to a set of tuples (which automatically removes duplicates) and then back to an array
        self._J = np.unique(self._J, axis=0)

        # objectives: define the objectives
        # Each chromosome is associated with an objective vector of I values, where each value typically expresses
        # the fulfillment degree of a different objective. In our case, we use a bidimensional vector.
        # The first element of the vector measures the complexity as the number of genes, corresponding to the
        # antecedents, which differ from 0, that is, the sum of the input variables actually used in each of the
        # M rules.
        # The second element expresses the accuracy as the root mean squared error (RMSE) between the output of
        # the Mamdani system and the expected output.
        self._objectives = [0, 0.0]

    def get_j(self):
        return self._J

    def set_j(self, J):
        self._J = J

    def set_j_element(self, i, j, value):
        self._J[i, j] = value

    def get_objectives(self):
        return self._objectives

    def set_objectives(self, objectives):
        self._objectives = objectives


# Crossover operator
# The one-point crossover operator cuts the chromosomes c1 and c2 at some chosen common gene and
# swaps the resulting sub-chromosomes. The common gene is chosen by extracting randomly a number
# in [M_min , ρ_min ], where M_min is the minimum number of rules, which must be present in a rule base,
# and ρ_min is the minimum number of rules in c1 and c2, and multiplying this number by (F+1).
def crossover(ind1, ind2):
    rho_min = min(ind1.get_j().shape[0], ind2.get_j().shape[0])
    common_gene = random.randint(M_MIN, rho_min) * (F + 1)

    temp = np.copy(ind1.get_j())
    ind1.set_j(np.concatenate((ind1.get_j()[:common_gene], ind2.get_j()[common_gene:])))
    ind2.set_j(np.concatenate((ind2.get_j()[:common_gene], temp[common_gene:])))

    return ind1, ind2


# First mutation operator
# The first mutation operator adds γ rules to the rule base, where γ is randomly chosen in [1, γ_max].
# The upper bound γ_max is fixed by the user. If γ + M > M_max, then γ=M_max−M.
# For each rule m added to the chromosome, we generate a random number t ∈ [1,F_max],
# which indicates the number of input variables used in the antecedent of the rule.
# Then, we generate t natural random numbers between 1 and F to determine the input variables
# which compose the antecedent part of the rule. Finally, for each selected input variable f,
# we generate a random natural number j_m,f between 1 and T_f , which determines the fuzzy set A_f,j_m,f
# to be used in the antecedent of rule m. To select the consequent fuzzy set A_(F+1),j_m,(F+1) ,
# a random number between 1 and T_(F+1) is generated.
def mutate1(ind):
    gamma = random.randint(1, GAMMA_MAX)
    if gamma + ind.get_j().shape[0] > M_MAX:
        gamma = M_MAX - ind.get_j().shape[0]

    for _ in range(gamma):
        t = random.randint(1, F_MAX)
        selected_input_vars = random.sample(range(1, F + 1), t)
        # Generate the antecedent parts of the rule for selected variables
        antecedent_part = [random.randint(1, C) if f in selected_input_vars else 0 for f in range(1, F + 1)]
        # Generate the consequent part of the rule separately
        consequent_part = random.randint(1, C)  # Assuming the consequent part should also be between 1 and C
        # Combine antecedent and consequent parts
        new_rule = antecedent_part + [consequent_part]
        ind.set_j(np.vstack([ind.get_j(), new_rule]))

    return ind,


# Second mutation operator
# The second mutation operator randomly changes δ elements of matrix J.
# The number δ is randomly generated in [1,δ_max]. The upper bound δ_max is fixed by the user.
# For each element to be modified, a number is randomly generated in [0, T_f],
# where f is the input variable corresponding to the selected matrix element.
# The element is modified only if the constraint on the maximum number of input variables
# for each rule is satisfied; otherwise, the element maintains its original value.
def mutate2(ind):
    delta = random.randint(1, DELTA_MAX)
    for _ in range(delta):
        # Since Python indexing is zero-based, we need to subtract 1 to include the last row
        # in the range of possible indices.
        row = random.randint(0, ind.get_j().shape[0] - 1)
        # Again, subtracting 1 is necessary to include the last column in the range of possible indices.
        col = random.randint(0, ind.get_j().shape[1] - 1)
        # new_value = random.randint(0, C)
        if col < ind.get_j().shape[1] - 1:  # If it's an antecedent
            new_value = random.randint(0, C)  # '0' represents "don't care"
        else:  # If it's the consequent
            new_value = random.randint(1, C)  # Ensure consequent is never '0'

        ind.set_j_element(row, col, new_value)

    # if the antecedent of a rule results to be composed only of “don’t care” conditions, the rule is removed.
    ind.set_j(np.array([rule for rule in ind.get_j() if not np.all(rule[:-1] == 0)]))

    # Further, the second mutation operator can generate a rule equal to another rule. Since two equal
    # rules contain the same information, one of the two rules is eliminated.
    unique_rules = np.unique(ind.get_j(), axis=0)
    ind.set_j(unique_rules)

    return ind,


# Evaluation function (placeholder - customize for your specific problem)
def evaluate(individual, u_matrix, cntr, x_val, y_val, fvt_generator):
    # controller = get_fuzzy_controller(individual.get_j(), u_matrix, cntr, input_universes, output_universe, 'triangular')
    controller = get_fuzzy_controller(individual.get_j(), fvt_generator)
    y_pred = []

    not_computed = 0
    computed = 0

    for record in x_val:
        # y_pred.append(controller.compute(x))
        # Input values into the controller
        for feature in record:
            controller.input[feature] = record[feature]

        # Compute the result
        # TODO: Si ferma qui perche non riesce a computare l'output
        #  potrei salatre quelli che danno questo errore e contare quanti record input salta
        # controller.compute()

        try:
            # Try to compute the result
            controller.compute()
        except ValueError:
            # AssertionError: Total area is zero in defuzzification!
            # ValueError: Crisp output cannot be calculated, likely because the system is too sparse.
            # Check to make sure this set of input values will activate at least one connected Term in each Antecedent
            # via the current set of Rules.
            not_computed += 1
            y_pred.append(0)
        except Exception as e:
            # Handle any other exceptions
            print(f"An unexpected error occurred: {e}")
        else:
            # This will only execute if no exception was raised
            computed += 1
            y_pred.append(controller.output['Occupancy'] * max_occupancy / 100)
        # finally:
        # This will execute regardless of whether an exception was raised or not
        # print("This is the cleanup code.")

        # Get output and append to y_pred
        # y_pred.append(controller.output['Occupancy'] * max_occupancy / 100)

    # Compute the RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    # Compute the complexity
    complexity = 0
    for rule in individual.get_j():
        complexity += len([f for f in rule[:-1] if f != 0])

    individual.set_objectives([complexity, rmse])

    return rmse, complexity


# Problem-specific constants
RMSE_WEIGHT = -1.0  # Minimize MSE
COMPLEXITY_WEIGHT = -1.0  # Minimize Complexity

# Create fitness and individual classes
creator.create("FitnessMulti", base.Fitness, weights=(RMSE_WEIGHT, COMPLEXITY_WEIGHT))
creator.create("Individual", Chromosome, fitness=creator.FitnessMulti)

# Initialize DEAP tools and operators
toolbox = base.Toolbox()

# Individual and population creation functions
# def create_individual(): return creator.Individual()
# toolbox.register("individual", create_individual)
toolbox.register("individual", creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", crossover)
toolbox.register("mutate1", mutate1)
toolbox.register("mutate2", mutate2)
toolbox.register("evaluate", evaluate)


# toolbox.register("isDominated", is_dominated)


# TODO: implement the selection operator as defined in the paper
# is the selection operator used in the m_paes function?
# toolbox.register("select", tools.selNSGA2)


def m_paes(u_matrix, cntr, x_val, y_val):
    random.seed(42)

    # A general version of PAES, denoted (μ + λ)PAES, uses μ current solutions and generates λ mutants:
    # each mutant is created by mutating one of the μ current solutions, which is selected via binary tournament
    # selection using the fitness values assigned in the previous iteration.
    # The fitness values are computed by considering the dominance of the current solutions with respect to the archive
    # and their crowding degree. In this code, we have used a sort of (2+2)PAES.

    # Start with two random solutions
    c1 = toolbox.individual()
    c2 = toolbox.individual()
    c1.fitness.values = toolbox.evaluate(c1, u_matrix, cntr, x_val, y_val)
    c2.fitness.values = toolbox.evaluate(c2, u_matrix, cntr, x_val, y_val)

    # Initialize the archive
    archive = [c1, c2]

    print("(2+2)M-PAES algorithm with ", str(total_generations), " generations started...")

    # Evolutionary algorithm
    for gen in range(total_generations):  # Number of generations

        # Randomly select new current solutions from the archive
        # Unlike classical (2+2)PAES, which maintains c1 and c2 as current solutions until they are not replaced
        # by solutions with particular characteristics, we randomly extract, at each iteration, the current solutions.
        # If the archive contains a unique solution, c1 and c2 correspond to this unique solution.
        # We experimentally verified that the random extraction of the current solutions from the archive allows us
        # to extend the set of non-dominated solutions contained in the archive so as to obtain a better approximation
        # of the Pareto front. As regards the candidate solution acceptance, s1 and s2 are added to the archive only
        # if they are dominated by no solution contained in the archive; possible solutions in the archive dominated
        # by s1 or s2 are removed. Typically, the size of the archive is fixed at the beginning of
        # the execution of PAES. In this case, when the archive is full and a new solution s has to be added to the
        # archive, if s dominates no solution in the archive, then we insert s into the archive and remove the
        # solution (possibly s itself) that belongs to the region with the highest crowding degree. If the region
        # contains more than one solution, then, the solution to be removed is randomly chosen.
        c1, c2 = random.sample(archive, 2) if len(archive) > 1 else (archive[0], archive[0])

        # Apply crossover and mutation
        if random.random() < 0.5:  # Crossover probability
            s1, s2 = map(toolbox.clone, [c1, c2])
            toolbox.mate(s1, s2)
            # delete the fitness.values attribute from two individuals
            del s1.fitness.values
            del s2.fitness.values

            # Mutation with a probability of 0.01
            if random.random() < 0.01:
                if random.random() < 0.55:
                    toolbox.mutate1(s1)
                    toolbox.mutate1(s2)
                else:
                    toolbox.mutate2(s1)
                    toolbox.mutate2(s2)
                del s1.fitness.values
                del s2.fitness.values
        else:
            # Mutation is always applied if crossover is not
            s1, s2 = map(toolbox.clone, [c1, c2])
            if random.random() < 0.55:
                toolbox.mutate1(s1)
                toolbox.mutate1(s2)
            else:
                toolbox.mutate2(s1)
                toolbox.mutate2(s2)
            del s1.fitness.values
            del s2.fitness.values

        # Evaluate the new solutions
        for s in [s1, s2]:
            if not s.fitness.valid:
                s.fitness.values = toolbox.evaluate(s, u_matrix, cntr, x_val, y_val)
                update_archive(archive, s)

        # Randomly select new current solutions from the archive
        # c1, c2 = random.sample(archive, 2) if len(archive) > 1 else (archive[0], archive[0])

        # print("Generation:", gen, "Archive size:", len(archive), "Best fitness:", archive[0].fitness.values)
        print_progress_bar(gen + 1, total_generations, prefix='Progress:', suffix='Complete', length=50)

    print("\n(2+2)M-PAES algorithm finished.")

    return archive


def m_paes_modified(u_matrix, cntr, x_val, y_val, n_stable_generations=50):

    fvts = {
        'triangular': get_fvts(u_matrix, cntr, input_universes, output_universe, 'triangular'),
        'gaussian': get_fvts(u_matrix, cntr, input_universes, output_universe, 'gaussian'),
        # 'trapezoidal': get_fvts(u_matrix, cntr, input_universes, output_universe, 'trapezoidal'),
    }

    start_time = time.time()  # Capture the start time

    print("(2+2)M-PAES algorithm started...")

    random.seed(42)
    stable_gen_count = 0  # Track stable generations where the archive does not change
    prev_archive = None  # Store the previous state of the archive for comparison

    # Start with two random solutions
    c1 = toolbox.individual()
    c2 = toolbox.individual()
    c1.fitness.values = toolbox.evaluate(c1, u_matrix, cntr, x_val, y_val, fvts['triangular'])
    c2.fitness.values = toolbox.evaluate(c2, u_matrix, cntr, x_val, y_val, fvts['triangular'])

    # Initialize the archive
    archive = [c1, c2]
    gen = 0  # Initialize generation counter

    # Continue evolving until the archive has not changed for n stable generations
    while stable_gen_count < n_stable_generations:
        gen += 1  # Increment generation counter
        # Randomly select new current solutions from the archive
        c1, c2 = random.sample(archive, 2) if len(archive) > 1 else (archive[0], archive[0])

        # Apply crossover and mutation, and evaluate the solutions
        if random.random() < 0.5:  # Crossover probability
            s1, s2 = map(toolbox.clone, [c1, c2])
            toolbox.mate(s1, s2)
            # delete the fitness.values attribute from two individuals
            del s1.fitness.values
            del s2.fitness.values

            # Mutation with a probability of 0.01
            if random.random() < 0.01:
                if random.random() < 0.55:
                    toolbox.mutate1(s1)
                    toolbox.mutate1(s2)
                else:
                    toolbox.mutate2(s1)
                    toolbox.mutate2(s2)
                del s1.fitness.values
                del s2.fitness.values
        else:
            # Mutation is always applied if crossover is not
            s1, s2 = map(toolbox.clone, [c1, c2])
            if random.random() < 0.55:
                toolbox.mutate1(s1)
                toolbox.mutate1(s2)
            else:
                toolbox.mutate2(s1)
                toolbox.mutate2(s2)
            del s1.fitness.values
            del s2.fitness.values

        # Evaluate the new solutions and update the archive
        for s in [s1, s2]:
            if not s.fitness.valid:
                s.fitness.values = toolbox.evaluate(s, u_matrix, cntr, x_val, y_val, fvts['triangular'])
                update_archive(archive, s)

        # Compare the current archive to the previous to check for changes
        if prev_archive is not None:
            # Sort archives for consistent comparison
            current_sorted_archive = sorted([sol.fitness.values for sol in archive])
            prev_sorted_archive = sorted([sol.fitness.values for sol in prev_archive])

            # Check if the archives are the same (no change)
            if current_sorted_archive == prev_sorted_archive:
                stable_gen_count += 1  # Increment stable generation count
            else:
                stable_gen_count = 0  # Reset stable generation count if there was a change
        else:
            stable_gen_count = 0  # Reset stable generation count if it's the first comparison

        prev_archive = list(archive)  # Update the previous archive for the next comparison

        # Optional: Print progress information
        # print_progress_bar(gen, n_stable_generations, prefix='Progress:', suffix='Complete', length=50)
        # Print elapsed time at each generation or at certain intervals
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        # print("Generation: {}, Elapsed Time: {:.2f} seconds".format(gen, elapsed_time))
        print(f"\rGeneration: {gen}, Elapsed Time: {elapsed_time:.2f} seconds", end="")

    print("\n(2+2)M-PAES algorithm finished after {} generations with stable archive for {} generations.".format(gen,
                                                                                                                 n_stable_generations))

    return archive


def is_dominated(ind1, ind2):
    # TODO, potrebbe ssere riscritta utilizzando if fit_i.dominates(fit_j):

    """
    Determine if ind1 is dominated by ind2.

    Parameters:
    - ind1: Individual 1, with its fitness attribute set.
    - ind2: Individual 2, with its fitness attribute set.

    Returns:
    - True if ind1 is dominated by ind2, False otherwise.
    """
    # At least one objective is strictly better in ind2, and all are as good or better.
    return all(x <= y for x, y in zip(ind2.fitness.values, ind1.fitness.values)) and any(
        x < y for x, y in zip(ind2.fitness.values, ind1.fitness.values))


def compute_crowding_distance(individuals):
    # Number of objectives
    num_objectives = len(individuals[0].fitness.values)

    # Initialize crowding distance for each individual
    for individual in individuals:
        individual.crowding_distance = 0

    # For each objective
    for i in range(num_objectives):
        # Sort individuals based on the objective
        # individuals.sort(key=lambda x: x.objectives[i])
        individuals.sort(key=lambda x: x.fitness.values[i])

        # Assign infinite distance to boundary individuals
        individuals[0].crowding_distance = float('inf')
        individuals[-1].crowding_distance = float('inf')

        # For all other individuals
        for j in range(1, len(individuals) - 1):
            # Difference in objective value with the neighbors
            # prev_diff = individuals[j].objectives[i] - individuals[j - 1].objectives[i]
            prev_diff = individuals[j].fitness.values[i] - individuals[j - 1].fitness.values[i]
            # next_diff = individuals[j + 1].objectives[i] - individuals[j].objectives[i]
            next_diff = individuals[j + 1].fitness.values[i] - individuals[j].fitness.values[i]

            # Normalize (if max and min values for the objective are known, replace 1 and 0 below accordingly)
            # max_obj = max(individuals, key=lambda x: x.objectives[i]).objectives[i]
            max_obj = max(individuals, key=lambda x: x.fitness.values[i]).fitness.values[i]
            # min_obj = min(individuals, key=lambda x: x.objectives[i]).objectives[i]
            min_obj = min(individuals, key=lambda x: x.fitness.values[i]).fitness.values[i]
            norm_diff = (next_diff - prev_diff) / (max_obj - min_obj if max_obj - min_obj > 0 else 1)

            # Update crowding distance (summing over all objectives)
            individuals[j].crowding_distance += norm_diff


def update_archive(archive, new_solution):
    # Add new_solution to the archive if it's not dominated by existing solutions
    is_dominated_by_any = any(is_dominated(new_solution, other) for other in archive)
    if not is_dominated_by_any:
        archive.append(new_solution)
        # Remove solutions that are dominated by the new solution
        archive[:] = [sol for sol in archive if not is_dominated(new_solution, sol)]

    # Managing the archive size based on crowding degree
    if len(archive) > ARCHIVE_SIZE:
        # Recompute crowding distances for the updated archive
        compute_crowding_distance(archive)

        # Sort the archive by crowding distance to identify the most crowded regions
        archive.sort(key=lambda ind: ind.crowding_distance)

        # Identify the lowest crowding distance (most crowded)
        min_crowding_distance = archive[0].crowding_distance

        # Filter individuals with the lowest crowding distance
        most_crowded_individuals = [ind for ind in archive if ind.crowding_distance == min_crowding_distance]

        # If more than one individual shares the lowest crowding distance, choose one at random to remove
        if len(most_crowded_individuals) > 1:
            individual_to_remove = random.choice(most_crowded_individuals)
        else:
            # If there's only one individual with the lowest crowding distance, select it for removal
            individual_to_remove = most_crowded_individuals[0]

        # Remove the selected individual from the archive
        archive.remove(individual_to_remove)

        # Optionally, if you need to further reduce the archive size to the limit, continue removing individuals
        # with the next lowest crowding distance until the archive size constraint is satisfied.
        """while len(archive) > ARCHIVE_SIZE:
            archive.pop(0)  # Assumes archive is still sorted by crowding distance"""


def get_fvts(u_matrix, cntr, input_universes, output_universe, mf_type):
    fvt_generator = None
    match mf_type:
        case 'gaussian':
            mf_params = GaussianFMParamsCalculator(u_matrix, cntr, data).calculate()
            fvt_generator = GaussianFVTGenerator(input_universes, output_universe, mf_params)
        case 'triangular':
            mf_params = TriangularFMParamsCalculator(u_matrix, cntr, data, 0.5).calculate()
            fvt_generator = TriangularFVTGenerator(input_universes, output_universe, mf_params)
        case 'trapezoidal':
            mf_params = TrapezoidalFMParamsCalculator(u_matrix, cntr, data, 0.5, 0.5).calculate()
            # TODO: Implement the TrapezoidalFVTGenerator class
            # fvt_generator = TrapezoidalFVTGenerator(input_universes, output_universe, mf_params)
        case _:
            raise KeyError("MF type not recognized.")

    return fvt_generator.generate_antecedents(), fvt_generator.generate_consequent()


# def get_fuzzy_controller(rb, u_matrix, cntr, input_universes, output_universe, mf_type):
def get_fuzzy_controller(rb, fvts):
    """fvt_generator = None
    match mf_type:
        case 'gaussian':
            mf_params = GaussianFMParamsCalculator(u_matrix, cntr, data).calculate()
            fvt_generator = GaussianFVTGenerator(input_universes, output_universe, mf_params)
        case 'triangular':
            mf_params = TriangularFMParamsCalculator(u_matrix, cntr, data, 0.5).calculate()
            fvt_generator = TriangularFVTGenerator(input_universes, output_universe, mf_params)
        case 'trapezoidal':
            mf_params = TrapezoidalFMParamsCalculator(u_matrix, cntr, data, 0.5, 0.5).calculate()
            # TODO: Implement the TrapezoidalFVTGenerator class
            # fvt_generator = TrapezoidalFVTGenerator(input_universes, output_universe, mf_params)
        case _:
            raise KeyError("MF type not recognized.")

    antecedents = fvt_generator.generate_antecedents()
    consequent = fvt_generator.generate_consequent()"""

    # Genereate the FuzzyRuleSet using the antecedents, consequent and RB
    # fuzzy_rule_set = JMatrixFRSGenerator(antecedents, consequent, rb).generate()
    fuzzy_rule_set = JMatrixFRSGenerator(fvts[0], fvts[1], rb).generate()

    # Generate the FuzzyController using the FuzzyRuleSet
    return ConcreteFLC(fuzzy_rule_set).generate()


"""def read_csv_data(filename):
    # Read data from a CSV file
    data = pd.read_csv(filename, usecols=['date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'])

    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    return data"""


def read_csv_data_average_sensors(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Sensor types to be considered
    # sensor_types = ['Temp', 'Light', 'Sound', 'PIR', 'CO2']
    sensor_types = ['Temp', 'Light', 'Sound', 'CO2']

    # Initialize a dictionary to store averaged sensor data
    average_dict = {}

    for sensor in sensor_types:
        # Get the columns for each sensor type
        sensor_columns = [col for col in df.columns if sensor in col]

        # Compute average for each sensor type
        average_dict[sensor] = df[sensor_columns].mean(axis=1)

    # Combine all series into a single dataframe
    average_df = pd.concat(average_dict, axis=1)

    # Add the Room_Occupancy_Count column to the dataframe
    average_df['Room_Occupancy_Count'] = df['Room_Occupancy_Count']

    return average_df


def find_optimal_clusters(data, min_clusters, max_clusters, m, eps, max_iter):
    max_score = -np.inf
    best_cluster_num = min_clusters

    for c in range(min_clusters, max_clusters + 1):
        model = ConcreteFCM(data, c, m, eps, max_iter)
        model.fuzzy_c_means()
        score = model.silhouette_score()

        if score > max_score:
            max_score = score
            best_cluster_num = c

    return best_cluster_num, max_score


def split_data(data, test_size=0.2):
    X = data.drop('Room_Occupancy_Count', axis=1)  # drop the target variable
    y = data['Room_Occupancy_Count']  # target variable

    # split the data into training and validation sets
    X_train, X_val, y_train, y_val = \
        train_test_split(X, y, test_size=test_size, random_state=42)  # using 20% of data for validation

    return X_train, X_val, y_train, y_val


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == "__main__":
    # Read data from CSV
    # data = read_csv_data('sensor_data.csv')
    data = read_csv_data_average_sensors("Occupancy_Estimation.csv")

    # Split the data into training and validation sets
    # y_train will not be used in this case since we are using Fuzzy C-Means clustering
    X_train, X_val, y_train, y_val = split_data(data, test_size=0.2)

    # Apply Fuzzy C-Means clustering to calculating U-matrix and cluster centers to the Training set
    data = X_train.values

    # TODO: Mettere risultato sulla Tesi
    # Last runned with 5 features Best number of clusters: 3, Silhouette score: 0.7330638441167079
    """best_cluster_num, max_score = find_optimal_clusters(data, 2, 5, 2, 0.01, 100)
    print(f"Best number of clusters: {best_cluster_num}, Silhouette score: {max_score}")
    u_matrix, cntr = ConcreteFCM(data, best_cluster_num, 2, 0.005, 1000).fuzzy_c_means()"""

    u_matrix, cntr = ConcreteFCM(data, C, 2, 0.005, 1000).fuzzy_c_means()

    # GaussianPlotter(GaussianFMParamsCalculator(u_matrix, cntr, data).calculate(), input_universes).plot()
    # TriangularPlotter(TriangularFMParamsCalculator(u_matrix, cntr, data, 0.5).calculate(), input_universes).plot()
    # TrapezoidalPlotter(TrapezoidalFMParamsCalculator(u_matrix, cntr, data, 0.5, 0.5).calculate(),
    # input_universes).plot()

    # Convert validation set into dictionary format for the controller
    X_val_dict = X_val.to_dict(orient='records')

    # generate the Pareto front using the (2+2)M-PAES algorithm
    # archive = m_paes(u_matrix, cntr, X_val_dict, y_val)
    archive = m_paes_modified(u_matrix, cntr, X_val_dict, y_val, 50)

    # Post-processing of final_archive

    for solution in archive:
        print("Solution objectives:", solution.fitness.values)
        # You can also print or inspect the solution's rule base (the _J matrix) or any other properties
        # print("Solution rule base:", solution.get_j())

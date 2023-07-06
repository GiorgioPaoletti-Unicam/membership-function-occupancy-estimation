import numpy as np
from deap import creator, base, tools, algorithms
from sklearn.metrics import precision_score
from skfuzzy import control as ctrl

# Number of rules in an individual
IND_SIZE = 15


class RuleSetOptimizer:
    def __init__(self, fvt_generator, input_universes, X_val_dict, y_val, max_occupancy, n_clusters=3, pop_size=50,
                 cxpb=0.5, mutpb=0.2, ngen=100):
        self._input_universes = input_universes
        self.X_val_dict = X_val_dict
        self.y_val = y_val
        self.max_occupancy = max_occupancy
        self.n_clusters = n_clusters
        self.pop_size = pop_size
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen

        # Create the fuzzy control variables
        self.inputs = fvt_generator.generate_antecedents()
        self.output = fvt_generator.generate_consequent()

        # Setup DEAP
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # Generate a random feature cluster number
        self.toolbox.register("a_cluster", np.random.randint, 0, self.n_clusters)

        # Generate a random consequent cluster number
        self.toolbox.register("c_cluster", np.random.randint, 0, self.n_clusters)

        # Generate a random rule
        def generate_rule():
            # Initialize rule dict with features as keys and clusters as values
            rule = {feature: 'cluster' + str(self.toolbox.a_cluster() + 1) for feature in self.inputs.keys()}

            # Assign 'Occupancy' a different cluster
            rule['Occupancy'] = 'cluster' + str(self.toolbox.c_cluster() + 1)

            # Generate condition using bitwise AND (&) operator between each feature in the rule
            condition = None
            for feature in self.inputs.keys():
                if condition is None:
                    condition = self.inputs[feature][rule[feature]]
                else:
                    condition = condition & self.inputs[feature][rule[feature]]

            # Create a new rule using the generated condition and the output for 'Occupancy'
            new_rule = ctrl.Rule(condition, self.output[rule['Occupancy']])

            return new_rule

        # Generate an individual
        """def generate_individual():
            rules = []
            for _ in range(IND_SIZE):
                rules.append(generate_rule())
            return generate_controller(rules)"""

        # Register an individual
        # self.toolbox.register("individual", creator.Individual, generate_individual)
        # self.toolbox.register("individual", tools.initRepeat, creator.Individual, generate_rule, n=n_rules)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, generate_rule, n=IND_SIZE)

        # Register a population
        # self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        # self.toolbox.register("population", tools.initRepeat, list, generate_controller(self.toolbox.individual()))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register the evaluation function
        self.toolbox.register("evaluate", self.evaluate)

        # Swaps half the rules between two controllers.
        def cxController(ind1, ind2):

            # Swap half the rules
            temp = ind1[len(ind1) // 2:]
            ind1[len(ind1) // 2:] = ind2[len(ind2) // 2:]
            ind2[len(ind2) // 2:] = temp

            return ind1, ind2

        self.toolbox.register("mate", cxController)

        # Mutates a controller by changing one rule at random.
        def mutController(individual):
            individual[np.random.randint(0, len(individual))] = generate_rule()
            return individual,

        self.toolbox.register("mutate", mutController)

        self.toolbox.register("select", tools.selTournament, tournsize=3)

    # Generate a controller with random rules
    def generate_controller(self, rules):
        system = ctrl.ControlSystem()
        # I have setted the indices form -1 due to a bug in skfuzzy
        # it lost always the last rule
        for i in range(-1, len(rules)):
            system.addrule(rules[i])
        simulation = ctrl.ControlSystemSimulation(system)

        return simulation

    def evaluate(self, individual):

        simulation = self.generate_controller(individual)

        y_pred = []
        for record in self.X_val_dict:

            # Input values into the controller
            for feature in record:
                simulation.input[feature] = record[feature]

            # Compute the result
            simulation.compute()

            # Get output and append to y_pred
            y_pred.append(simulation.output['Occupancy'] * self.max_occupancy / 100)

        threshold = 0.5
        y_true_binary = [1 if y > threshold else 0 for y in self.y_val]
        y_pred_binary = [1 if y > threshold else 0 for y in y_pred]

        precision = precision_score(y_true_binary, y_pred_binary)
        print('Precision: ', precision)
        return precision,

    def optimize(self):
        population = self.toolbox.population(n=self.pop_size)

        # Calculate fitness for each individual in the population
        fitnesses = map(self.toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        result = algorithms.eaSimple(population, self.toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen)
        best_individual = tools.selBest(population, k=1)[0]

        return best_individual

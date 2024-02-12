import numpy as np
from skfuzzy import control as ctrl
import skfuzzy as fuzz
from deap import creator, base, tools, algorithms
import random
from deap import creator, base, tools, algorithms
from sklearn.metrics import silhouette_score
import random


class AbstractFRSGenerator:

    def __init__(self, input_antecedents, output_consequent):
        self._input_antecedents = input_antecedents
        self._output_consequent = output_consequent
        self._feature_names = list(input_antecedents.keys())
        self._fuzzy_rules = []

    def generate(self):
        raise NotImplementedError("Please Implement this method")


class JMatrixFRSGenerator(AbstractFRSGenerator):

    def __init__(self, input_antecedents, output_consequent, j_matrix):
        super().__init__(input_antecedents, output_consequent)
        self._j_matrix = j_matrix

    def generate(self):

        for i, row in enumerate(self._j_matrix):  # Iterate over each rule
            # conditions = []
            conditions = None

            for f in range(len(self._feature_names)):  # For each feature
                # If the generic element is 0, indicates that the fuzzy set A_f,j_m,f that has been
                # selected for variable X_f in rule Rm correspond to the 'don't care' condition
                if row[f] == 0:
                    continue
                else:
                    linguistic_variable = self._feature_names[f]
                    fuzzy_set = 'cluster' + str(row[f])
                # conditions.append(self._input_antecedents[linguistic_variable][fuzzy_set])
                if conditions is None:
                    conditions = self._input_antecedents[linguistic_variable][fuzzy_set]
                else:
                    conditions = conditions & self._input_antecedents[linguistic_variable][fuzzy_set]

            # The last column of J corresponds to consequent's membership function
            consequent_mf_name = 'cluster' + str(row[-1])
            # rule = ctrl.Rule(ctrl.And(*conditions), self._output_consequent[consequent_mf_name])
            # rule = ctrl.Rule(conditions, self._output_consequent[consequent_mf_name], label='rule_' + str(i+1))
            try:
                rule = ctrl.Rule(conditions, self._output_consequent[consequent_mf_name])
            except ValueError:
                print("ciao")
            else:
                self._fuzzy_rules.append(rule)
            # self._fuzzy_rules.append(rule)

        return self._fuzzy_rules


class ConcreteFRSGenerator(AbstractFRSGenerator):

    def __init__(self, input_antecedents, output_consequent):
        super().__init__(input_antecedents, output_consequent)

    def generate(self):
        # Generate rules for each feature

        """for i in range(len(self._output_consequent.terms)):
            condition = None
            for feature in self._feature_names:
                if condition is None:
                    condition = self._input_antecedents[feature]['cluster' + str(i + 1)]
                else:
                    # condition = condition & self._input_antecedents[feature]['cluster' + str(i + 1)]
                    condition = condition | self._input_antecedents[feature]['cluster' + str(i + 1)]
            rule = ctrl.Rule(condition, self._output_consequent['cluster' + str(i + 1)])
            self._fuzzy_rules.append(rule)"""

        """for feature in self._feature_names:
            # Assuming 'input_antecedents' are the MFs of each feature
            antecedent_mfs = self._input_antecedents[feature]
            consequent_mfs = self._output_consequent

            # Generate rules for each membership function of the antecedent
            for mf in antecedent_mfs.terms:
                # Here we generate a very basic rule: if 'feature' is 'mf', then output is 'consequent'
                # You may want to design more complex rules depending on your specific problem
                rule = ctrl.Rule(antecedent=(self._input_antecedents[feature][mf]),
                                 consequent=self._output_consequent[mf])
                self._fuzzy_rules.append(rule)"""

        self._fuzzy_rules = []

        PIR_antecedent = ctrl.Antecedent(np.arange(0, 2, 1), 'PIR')
        PIR_antecedent['0'] = fuzz.trimf(PIR_antecedent.universe, [0, 0, 1])
        PIR_antecedent['1'] = fuzz.trimf(PIR_antecedent.universe, [1, 2, 2])

        # Rule 1 if the sound is not cluster1 then the output is not cluster1
        """self._fuzzy_rules.append(ctrl.Rule(fuzz.fuzzy_not(self._input_antecedents['Sound']['cluster1']),
                                           fuzz.fuzzy_not(self._output_consequent['cluster1'])))"""
        """self._fuzzy_rules.append(ctrl.Rule(~self._input_antecedents['Sound']['cluster1'],
                                           ~self._output_consequent['cluster1']))"""
        self._fuzzy_rules.append(ctrl.Rule(self._input_antecedents['Sound']['cluster1'],
                                           ~self._output_consequent['cluster1']))

        # Rule 2 if the PIR is 1 then the output is not cluster1
        self._fuzzy_rules.append(ctrl.Rule(PIR_antecedent['1'], fuzz.fuzzy_not(self._output_consequent['cluster1'])))

        # Rule 3 if the light is not cluster1 AND sound is not cluster 1 AND PIR is 1 then the output is not cluster1
        self._fuzzy_rules.append(ctrl.Rule(fuzz.fuzzy_not(self._input_antecedents['Light']['cluster1']) &
                                           fuzz.fuzzy_not(self._input_antecedents['Sound']['cluster1']) &
                                           PIR_antecedent['1'],
                                           fuzz.fuzzy_not(self._output_consequent['cluster1'])))

        return self._fuzzy_rules

from skfuzzy import control as ctrl


class AbstractFRSGenerator:

    def __init__(self, input_antecedents, output_consequent):
        self._input_antecedents = input_antecedents
        self._output_consequent = output_consequent
        self._feature_names = self._feature_names = list(input_antecedents.keys())
        self._fuzzy_rules = []

    def generate(self):
        raise NotImplementedError("Please Implement this method")


class ConcreteFRSGenerator(AbstractFRSGenerator):

    def __init__(self, input_antecedents, output_consequent):
        super().__init__(input_antecedents, output_consequent)

    def generate(self):
        for i in range(len(self._output_consequent.terms)):
            for feature in self._feature_names:
                rule = ctrl.Rule(
                    self._input_antecedents[feature]['cluster' + str(i + 1)],
                    self._output_consequent['cluster' + str(i + 1)]
                )
                self._fuzzy_rules.append(rule)

        """for i in range(len(self._output_consequent.terms)):
            condition = None
            for feature in self._feature_names:
                if condition is None:
                    condition = self._input_antecedents[feature]['cluster' + str(i + 1)]
                else:
                    condition = condition & self._input_antecedents[feature]['cluster' + str(i + 1)]
            rule = ctrl.Rule(condition, self._output_consequent['cluster' + str(i + 1)])
            self._fuzzy_rules.append(rule)"""

        return self._fuzzy_rules

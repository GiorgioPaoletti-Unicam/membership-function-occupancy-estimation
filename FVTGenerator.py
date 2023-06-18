import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class AbstractFVTGenerator(object):

    def __init__(self, input_universes, output_universe, mf_params):
        self._mf_params = mf_params
        self._input_universes = input_universes
        self._output_universe = output_universe
        self._feature_names = list(input_universes.keys())
        self._input_memb_functions = self._generate_input_mf()
        self._output_memb_function = self._generate_output_mf()

    def _generate_input_mf(self):
        raise NotImplementedError("Please Implement this method")

    def _generate_output_mf(self):
        raise NotImplementedError("Please Implement this method")

    def generate_antecedents(self):
        antecedents = {}

        for feature in self._feature_names:
            antecedent = ctrl.Antecedent(self._input_universes[feature], feature)
            for i, mf in enumerate(self._input_memb_functions[feature]):
                antecedent[mf] = self._input_memb_functions[feature][mf]
            antecedents[feature] = antecedent

        return antecedents

    def generate_consequent(self):
        consequent = ctrl.Consequent(self._output_universe, 'Occupancy')
        for i, mf in enumerate(self._output_memb_function):
            consequent[mf] = self._output_memb_function[mf]
        return consequent


class GaussianFVTGenerator(AbstractFVTGenerator):

    def __init__(self, input_universes, output_universe, mf_params):
        super().__init__(input_universes, output_universe, mf_params)

    def _generate_input_mf(self):
        input_memb_functions = {}
        for nf, feature in enumerate(self._feature_names):
            input_memb_functions[feature] = {}

            for i in range(len(self._mf_params)):
                input_memb_functions[feature]['cluster' + str(i + 1)] = \
                    fuzz.gaussmf(
                        self._input_universes[feature],
                        self._mf_params['cluster' + str(i + 1)]['mean'][nf],
                        self._mf_params['cluster' + str(i + 1)]['std'][nf, nf]
                    )

        return input_memb_functions

    def _generate_output_mf(self):
        output_memb_function = {}

        # Calculate the distance between the centers of each function
        step_size = (self._output_universe.max() - self._output_universe.min()) / (
                len(self._mf_params) - 1)

        # Calculate the standard deviation to be used for each function
        sigma = step_size / 2  # This is a heuristic, adjust as needed

        for i in range(len(self._mf_params)):
            output_memb_function['cluster' + str(i + 1)] = \
                fuzz.gaussmf(
                    self._output_universe,
                    self._output_universe.min() + i * step_size,
                    sigma
                )
        return output_memb_function

from skfuzzy import control as ctrl


class AbstractFLC(object):

    def __init__(self, rules):
        self._rules = rules
        self._controller = None

    def generate(self):
        raise NotImplementedError("Please Implement this method")


class ConcreteFLC(AbstractFLC):

    def __init__(self, rules):
        super().__init__(rules)

    def generate(self):
        # Create a new control system
        system = ctrl.ControlSystem(self._rules)

        # Create a simulation of this system
        self._controller = ctrl.ControlSystemSimulation(system)

        return self._controller


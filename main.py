import pandas as pd

from FCM import *
from FMParamsCalculator import *
from FMPlotter import *
from FRSGenerator import *
from FLCGenerator import *
from FLCPerformance import *
from FVTGenerator import *

num_clusters = 3
n_features = 5

# TODO: fix the universe
input_universes = {
    'Temperature': np.arange(18, 30, 0.01),
    'Humidity': np.arange(14, 45, 0.01),
    'Light': np.arange(0, 1800, 0.01),
    'CO2': np.arange(300, 2200, 0.01),
    'HumidityRatio': np.arange(0.01, 0.1, 0.001)
}

output_universe = np.arange(0, 101, 1)

"""def create_universe(data, feature_names):
    universes = {}

    # Iterate over each feature
    for feature in feature_names:
        min_val = np.min(data[:, feature])
        max_val = np.max(data[:, feature])

        universes[feature] = ctrl.Antecedent(np.arange(min_val, max_val, 1), feature)

    return universes"""


def get_feature_names(data):
    return list(data.columns.values)


def read_csv_data(filename):
    # Read data from a CSV file
    data = pd.read_csv(filename, usecols=['date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'])

    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    return data


if __name__ == '__main__':
    # Read data from CSV
    data = read_csv_data('sensor_data.csv')

    # Apply Fuzzy C-Means clustering to calculating U-matrix and cluster centers
    data = data.values
    u_matrix, cntr = ConcreteFCM(data, num_clusters, 2, 0.005, 1000).fuzzy_c_means()

    # Calculate MF parameters and plot MFs

    # Guassian MF
    mf_params_guassian = GaussianFMParamsCalculator(u_matrix, cntr, data).calculate()
    # GaussianPlotter(mf_params_guassian).plot()

    # Triangular MF
    # mf_params_triangular = TriangularFMParamsCalculator(u_matrix, cntr, data, 0.5).calculate()
    # TriangularPlotter(mf_params_triangular).plot()

    # Trapezoidal MF
    # mf_params_trapezoidal = TrapezoidalFMParamsCalculator(u_matrix, cntr, data, 0.5, 0.5).calculate()
    # TrapezoidalPlotter(mf_params_trapezoidal).plot()

    # Generate Guassian Controller

    """gaussianFMGenerator = GaussianFMGenerator(mf_params_guassian, input_universes, output_universe)
    guassian_in_mf = gaussianFMGenerator.generate_input_mf()
    guassian_out_mf = gaussianFMGenerator.generate_output_mf()"""

    gaussianFVTGenerator = GaussianFVTGenerator(input_universes, output_universe, mf_params_guassian)
    gaussianAntecedents = gaussianFVTGenerator.generate_antecedents()
    gaussianConsequent = gaussianFVTGenerator.generate_consequent()

    guassian_fuzzy_rule_set = ConcreteFRSGenerator(gaussianAntecedents, gaussianConsequent).generate()

    guassian_controller = ConcreteFLC(guassian_fuzzy_rule_set).generate()

    # Input values into the controller
    simple_inputs = {'Temperature': 25.7, 'Humidity': 28.272, 'Light': 0.0, 'CO2': 1800.2,
              'HumidityRatio': 0.0087641630241641}
    for feature in simple_inputs:
        guassian_controller.input[feature] = simple_inputs[feature]

    # Compute the result
    guassian_controller.compute()

    # Print the result
    print(guassian_controller.output['Occupancy'])


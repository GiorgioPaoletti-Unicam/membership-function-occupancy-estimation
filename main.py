import pandas as pd

from FCM import *
from FMParamsCalculator import *
from FMPlotter import *

num_clusters = 3
n_features = 5


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
    GaussianPlotter(mf_params_guassian).plot()

    # Triangular MF
    mf_params_triangular = TriangularFMParamsCalculator(u_matrix, cntr, data, 0.5).calculate()
    TriangularPlotter(mf_params_triangular).plot()

    # Trapezoidal MF
    mf_params_trapezoidal = TrapezoidalFMParamsCalculator(u_matrix, cntr, data, 0.5, 0.5).calculate()
    TrapezoidalPlotter(mf_params_trapezoidal).plot()


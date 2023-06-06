import pandas as pd
import skfuzzy as fuzz

num_clusters = 3


def read_csv_data(filename):
    # Read data from a CSV file
    data = pd.read_csv(filename, usecols=['date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'])

    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    return data


def fuzzy_c_means_clustering(data, n_clusters=3):

    fuzziness_coefficient = 2
    error = 0.005
    max_iter = 1000
    input_data = data.T.to_numpy()

    cntr, u, _, _, _, _, _ = fuzz.cmeans(
        input_data,
        n_clusters,
        fuzziness_coefficient,
        error,
        max_iter,
    )
    return cntr, u


if __name__ == '__main__':
    # Read data from CSV
    data = read_csv_data('sensor_data.csv')

    # Apply Fuzzy C-Means clustering
    cntr, u = fuzzy_c_means_clustering(data, num_clusters)

    print(data)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from FCM import ConcreteFCM
from FLCGenerator import ConcreteFLC
from FMParamsCalculator import GaussianFMParamsCalculator, TriangularFMParamsCalculator, TrapezoidalFMParamsCalculator
from FMPlotter import GaussianPlotter, TriangularPlotter, TrapezoidalPlotter
from FRSGenerator import ConcreteFRSGenerator
from FVTGenerator import GaussianFVTGenerator, TriangularFVTGenerator
# from MonteCarloSimulation import monte_carlo_simulation
from RuleSetOptimizer import RuleSetOptimizer

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from real_vs_pred import *

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

num_clusters = 3
max_occupancy = 3

input_universes = {
    'Temp': np.arange(18, 30, 0.01),
    'Light': np.arange(0, 1800, 0.01),
    'Sound': np.arange(0.0, 4.0, 0.01),
    # 'PIR': np.arange(0, 1, 1), # 0 or 1 used directly in the rule set
    'CO2': np.arange(300, 2200, 0.01),
}

output_universe = np.arange(0, 101, 1)


def split_data(data, test_size=0.2):
    X = data.drop('Room_Occupancy_Count', axis=1)  # drop the target variable
    y = data['Room_Occupancy_Count']  # target variable

    # split the data into training and validation sets
    X_train, X_val, y_train, y_val = \
        train_test_split(X, y, test_size=test_size, random_state=42)  # using 20% of data for validation

    return X_train, X_val, y_train, y_val


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


def find_best_defuzzification_method(controller, inputs, actual_values, max_occupancy):
    defuzz_methods = ['centroid', 'bisector', 'mom', 'som', 'lom']
    best_method = None
    best_rmse = float('inf')
    best_method_predicted_values = None

    for method in defuzz_methods:
        # Apply the defuzzification method
        controller.defuzzify_method = method

        # Store the predicted values
        predicted_values = []

        # Iterate over the inputs
        for input_values in inputs:
            # Input values into the controller
            for feature in input_values:
                controller.input[feature] = input_values[feature]

            # Compute the result
            controller.compute()

            # Get output and append to predicted_values
            # output = controller.output['Occupancy']
            # Get output (as a percentage) and convert it to an actual value within the range 0 to n(max_occupancy)
            output = controller.output['Occupancy'] * max_occupancy / 100
            predicted_values.append(output)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))

        # Print the RMSE for the current method
        print(f"Method: {method}, RMSE: {rmse}")

        # Update the best RMSE and method if the current method is better
        if rmse < best_rmse:
            best_rmse = rmse
            best_method = method
            best_method_predicted_values = predicted_values

    return best_method, best_rmse, best_method_predicted_values


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


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()


def plot_roc_curve(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    # Read data from CSV
    data = read_csv_data_average_sensors("Occupancy_Estimation.csv")

    # Split the data into training and validation sets
    # y_train will not be used in this case since we are using Fuzzy C-Means clustering
    X_train, X_val, y_train, y_val = split_data(data, test_size=0.2)

    # Apply Fuzzy C-Means clustering to calculating U-matrix and cluster centers to the Training set
    data = X_train.values

    # Last runned with 5 features Best number of clusters: 3, Silhouette score: 0.7330638441167079
    """best_cluster_num, max_score = find_optimal_clusters(data, 2, 5, 2, 0.01, 100)
    print(f"Best number of clusters: {best_cluster_num}, Silhouette score: {max_score}")
    u_matrix, cntr = ConcreteFCM(data, best_cluster_num, 2, 0.005, 1000).fuzzy_c_means()"""

    u_matrix, cntr = ConcreteFCM(data, num_clusters, 2, 0.005, 1000).fuzzy_c_means()

    # GaussianPlotter(GaussianFMParamsCalculator(u_matrix, cntr, data).calculate(), input_universes).plot()
    # TriangularPlotter(TriangularFMParamsCalculator(u_matrix, cntr, data, 0.5).calculate(), input_universes).plot()
    # TrapezoidalPlotter(TrapezoidalFMParamsCalculator(u_matrix, cntr, data, 0.5, 0.5).calculate(), input_universes).plot()
    # print(TriangularPlotter(TriangularFMParamsCalculator(u_matrix, cntr, data, 0.5).calculate()).describe_mf())

    # Guassian Fuzzy Logic Controller
    gaussianFVTGenerator = GaussianFVTGenerator(input_universes,
                                                output_universe,
                                                GaussianFMParamsCalculator(u_matrix, cntr, data).calculate()
                                                )
    """guassian_controller = ConcreteFLC(
        ConcreteFRSGenerator(gaussianFVTGenerator.generate_antecedents(),
                             gaussianFVTGenerator.generate_consequent()
                             ).generate()
    ).generate()"""

    """# Triangular Fuzzy Logic Controller
    triangularFVTGenerator = TriangularFVTGenerator(input_universes,
                                                    output_universe,
                                                    TriangularFMParamsCalculator(u_matrix, cntr, data, 0.5).calculate()
                                                    )
    triangular_controller = ConcreteFLC(
        ConcreteFRSGenerator(triangularFVTGenerator.generate_antecedents(),
                             triangularFVTGenerator.generate_consequent()
                             ).generate()
    ).generate()"""

    # controllers = [guassian_controller]
    # controller_stats = monte_carlo_simulation(controllers, input_universes)

    """frs_generator = \
        ConcreteFRSGenerator(gaussianFVTGenerator.generate_antecedents(),
                             gaussianFVTGenerator.generate_consequent()
                             )"""

    # Convert validation set into dictionary format for the controller
    X_val_dict = X_val.to_dict(orient='records')

    optimizer = RuleSetOptimizer(gaussianFVTGenerator, input_universes, X_val_dict, y_val, max_occupancy, num_clusters)
    optimized_rules = optimizer.optimize()

    """# Initialize empty list for storing predictions
    y_pred = []

    # Iterate over the validation set
    for record in X_val_dict:
        # Input values into the controller
        for feature in record:
            # guassian_controller.input[feature] = record[feature]
            triangular_controller.input[feature] = record[feature]

        # Compute the result
        # guassian_controller.compute()
        triangular_controller.compute()

        # Get output and append to y_pred
        # output = guassian_controller.output['Occupancy'] * max_occupancy / 100
        output = triangular_controller.output['Occupancy'] * max_occupancy / 100
        y_pred.append(output)

    rme = np.sqrt(np.mean((np.array(y_val) - np.array(y_pred)) ** 2))
    mae = np.mean(np.abs(np.array(y_val) - np.array(y_pred)))

    print(f"RMSE: {rme}")
    print(f"MAE: {mae}")

    # Convert y_true and y_pred to binary format
    # The threshold is the value that separates these two classes.
    # For example, if the outputs of your fuzzy logic system are between 0 and 1, you might set a threshold of 0.5.
    # Any outputs above 0.5 would be considered as class 1, and any outputs below or equal
    # to 0.5 would be considered as class 0.
    threshold = 0.5
    y_true_binary = [1 if y > threshold else 0 for y in y_val]
    y_pred_binary = [1 if y > threshold else 0 for y in y_pred]

    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f_score = f1_score(y_true_binary, y_pred_binary)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F-Score: {f_score}")

    # plot_roc_curve(y_val, y_pred)
    # plot_confusion_matrix(y_val, y_pred)"""

    """best_method, best_rmse, y_pred = \
        find_best_defuzzification_method(guassian_controller, X_val_dict, y_val, max_occupancy)
    print(f"Best defuzzification method: {best_method} with RMSE: {best_rmse}")

    plot_real_vs_predicted(y_val, y_pred)
    plot_scatter_real_vs_predicted(y_val, y_pred)
    plot_histogram_real_vs_predicted(y_val, y_pred)
"""
    print('ciao')

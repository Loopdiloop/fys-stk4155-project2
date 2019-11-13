import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os

from data_generation import data_generate
from fit_matrix import fit
from visualization import plot_3d, plot_bias_var_tradeoff, plot_mse_vs_complexity, plot_bias_variance_vs_complexity
import statistical_functions as statistics
from sampling_methods import sampling

"""Running task c) of the project, as well as some parts of c).
Running additional variance-bias tradeoff calculations
for different degrees of polynomials of OLS. """

deg = range(1,13)           # degree of polynomial
k = 5                       # k batches for k-fold.
#method = "least squares"    # "least squares", "ridge" or "lasso"
method = "lasso"

# Initialize lists for plotting
best_mse_train = []
best_mse_test = []
average_mse_train = []
average_mse_test = []
average_bias = []
average_variance = []

# Load previously generated dataset
dataset = data_generate()
dataset.load_data()

# Normalize and divide in samples
dataset.normalize_dataset()
dataset.sort_in_k_batches(k)

for pol_deg in deg:

    #Run k-fold algorithm and fit models.
    sample = sampling(dataset)
    if method == "least squares":
        sample.kfold_cross_validation(k, method, deg = pol_deg)
    else:
        sample.kfold_cross_validation(k, method, deg = pol_deg, lambd = 1e-2)
    
    # Save data for plotting
    best_mse_test.append(np.min(sample.mse))
    best_mse_train.append(sample.mse_train[ np.argmin(sample.mse)])
    average_mse_test.append(np.average(sample.mse))
    average_mse_train.append(np.average(sample.mse_train))
    average_bias.append(np.average(sample.bias))
    average_variance.append(np.average(sample.variance))
    
    # Print statistics
    print("\n" + "Run for k = ", k, " and deg = ", pol_deg)
    statistics.print_mse(sample.mse)
    statistics.print_R2(sample.R2)

# Plot the graphs showing the bias-variance tradeoff
plot_mse_vs_complexity(deg, average_mse_test, average_mse_train)
plot_bias_variance_vs_complexity(deg, average_bias, average_variance)


try:
    os.remove("backup_data.npz")
except:
    print("Error: backup_data.npz not deleted.")
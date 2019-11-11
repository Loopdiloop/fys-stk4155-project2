import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os


from data_generation import data_generate
from fit_matrix import fit
from visualization import plot_3d, plot_beta, plot_bias_variance_vs_lambdas, plot_mse_vs_lambda
import statistical_functions as statistics
from sampling_methods import sampling

"""Running task d) and e) of the project.
Ridge regression and dependence on lambda.
"""


#Parameters for the simulation
deg = 5                 #degree of polynomial
no_lambdas = 8          # the number of labdas you want to test
k = 5                   # k batches for k-fold.
method = "lasso"        # "least squares", "ridge" or "lasso"
lambdas = [10**(-no_lambdas +5 + i) for i in range(no_lambdas)]


# Load dataset and Franke function
dataset = data_generate()
liste1 = [dataset] #M: Trenger du denne fremdeles, F?
dataset.load_data()

# Normalize the dataset and divide in samples
dataset.normalize_dataset()
dataset.sort_in_k_batches(k)

# Initialize lists for plotting
best_mse_train = []
best_mse_test = []
average_mse_train = []
average_mse_test = []
average_bias = []
average_variance = []
betas = []

for i in range(no_lambdas):

    #Run k-fold algorithm and fit models.
    sample = sampling(dataset)
    sample.kfold_cross_validation(k, method, lambd=lambdas[i])
    
    # Save statistics
    best_mse_test.append(np.min(sample.mse))
    best_mse_train.append(sample.mse_train[ np.argmin(sample.mse)])
    average_mse_test.append(np.average(sample.mse))
    average_mse_train.append(np.average(sample.mse_train))
    average_bias.append(np.average(sample.bias))
    average_variance.append(np.average(sample.variance))
    betas.append(sample.best_predicting_beta)
    
    # Print statistics
    print("\n"+"Batches: k = ", k, " Lambda = ", lambdas[i])
    statistics.print_mse(sample.mse)
    statistics.print_R2(sample.R2)

    # Plotting the best fit/best beta with the lowest mse.
    dataset.reload_data()
    fitted = fit(dataset)
    liste2 = [fitted] #M: Trenger du denne fremdeles, F?
    liste2 = [sample] #M: Trenger du denne fremdeles, F?
    fitted.create_design_matrix(deg = deg)
    z_model_norm = fitted.test_design_matrix(sample.best_predicting_beta)

# Generate analytical solution for plotting purposes
analytical = data_generate()
analytical.generate_franke(150, noise=0)

#rescale for plotting:
rescaled_dataset = dataset.rescale_back(z = z_model_norm)
z_model = rescaled_dataset[2]


# Plot
#plot_3d(dataset.x_unscaled, dataset.y_unscaled, z_model, analytical.x_mesh, analytical.y_mesh, analytical.z_mesh, ["surface", "scatter"])
#beta_numpy = np.array(betas)
#plot_beta(lambdas, beta_numpy)
plot_bias_variance_vs_lambdas(lambdas, average_bias, average_variance)

#plot_mse_vs_lambda(lambdas, average_mse_test, average_mse_train)

try:
    os.remove("backup_data.npz")
except:
    print("Error: backup_data.npz not deleted.")
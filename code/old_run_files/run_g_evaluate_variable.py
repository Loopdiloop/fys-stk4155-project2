import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from functions import reduce4

from data_generation import data_generate
from fit_matrix import fit
from visualization import plot_3d, plot_3d_terrain, plot_bias_var_tradeoff, plot_mse_vs_complexity, plot_terrains
import statistical_functions as statistics
from sampling_methods import sampling


def rung(CV, k, method, lambd, pol_deg):
    """A complimentary run-functionfor for terrain data with different variables. Called by run_g.py """
    #Load the dataset
    dataset = data_generate()
    dataset.load_data()
    shape = dataset.shape
    
    #Normalize and sample the dataset
    dataset.normalize_dataset()
    #Run k-fold algorithm and fit models.
    if CV:
        sample = sampling(dataset)
        dataset.sort_in_k_batches(k)
        if method == "least squares":
            sample.kfold_cross_validation(k, method, deg = pol_deg)
        else:
            sample.kfold_cross_validation(k, method, deg = pol_deg, lambd = lambd)
    
        #Print statistics if CV
        if method == "least squares":
            print("\n" + "Run for k = ", k, " and deg = ", pol_deg)
        else:
            print("\n"+"Batches: k = ", k, " Lambda = ", lambd, " and deg = ", pol_deg) 
        
        bias = np.average(sample.bias)
        variance = np.average(sample.variance)
        mse = np.average(sample.mse)
        R2 = np.average(sample.R2)
        statistics.print_mse(sample.mse)
        statistics.print_R2(sample.R2)
    else:
        print('\n' + 'Run for deg = ', pol_deg)
    
    
    
    # Plotting the best fit/best beta with the lowest mse.
    fitted = fit(dataset)
    fitted.create_design_matrix(deg = pol_deg)
    if CV:
        z_model_norm = fitted.test_design_matrix(sample.best_predicting_beta)
    else:
        if method == "least squares":
            z_model_norm, beta = fitted.fit_design_matrix_numpy()
        elif method == "ridge":
            z_model_norm, beta = fitted.fit_design_matrix_ridge()
        else:
            z_model_norm, beta = fitted.fit_design_matrix_lasso()
        
        #Print statistics if not CV
        mse, R2 = statistics.calc_statistics(dataset.z_1d, z_model_norm)
        print("Mean square error: ", mse, "\n", "R2 score: ", R2)
    
    rescaled_dataset = dataset.rescale_back(z = z_model_norm)
    z_model = rescaled_dataset[2]
    
    z_matrix = np.empty(shape)
    z_matrix[:] = np.nan
    try:
        len(z_matrix) > 10
    except:
        raise Exception("This function is for terrain data only. Are you sure you used the correct dataset? ")
    for i, z_value in enumerate(z_model):
        z_matrix[int(rescaled_dataset[0,i]), int(rescaled_dataset[1,i])] = z_value
    
    return z_matrix, mse, R2, bias, variance
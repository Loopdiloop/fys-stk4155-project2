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
from run_g_evaluate_variable import rung

""" Run-file specifically for terrain-data. Will not work on franke-data, but the same evaluations can be done with other rund files for the generated analytical data. """

#Regression parameters
CV = True                   #Use Cross Validation?
k=5                         #k-fold?
method = 'lasso'            #Method
no_lambdas = 6
lambdas = [10**(-no_lambdas + 3 + i) for i in range(no_lambdas)]
degs = [3,5,7,9,11,13]       #degree of the polynomial
#pol_deg = 5
lambd = 1e-2

ind_var = degs
ind_var_text = "deg"


z_matrices = []
mses = []
R2s = []
biases = []
variances = []
for pol_deg in ind_var:
    current_z_matrix, current_mse, current_R2, current_bias, current_variance = rung(CV, k, method, lambd, pol_deg)
    z_matrices.append(current_z_matrix)
    mses.append(current_mse)
    R2s.append(current_R2)
    biases.append(current_bias)
    variances.append(current_variance)

if CV:
    CV_text = "w/"
else:
    CV_text = "without"

# Make plots of the different results:
plot_terrains(ind_var, ind_var_text, method, CV_text, x_matrices, x_labels, mses, R2s, lambdas, biases, variances)
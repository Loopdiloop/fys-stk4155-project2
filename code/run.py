import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os


from data_generation import data_generate, credit_card
from fit_matrix import fit
from visualization import plot_3d, plot_sigmoid
import statistical_functions as statistics
from sampling_methods import sampling



#Load credit card data
dataset = credit_card()
dataset.load_credit_card_data()
#Split test/training data
dataset.split_training_test_sklearn()
#Preprocessing, removes invalid data and normalizes.
dataset.preprocess_data()




#plot_sigmoid(df, lr, "gender")




"""
        lambdas=np.logspace(-5,7,13)
        parameters = [{'C': 1./lambdas, "solver":["lbfgs"]}]#*len(parameters)}]
        scoring = ['accuracy', 'roc_auc']
        logReg = LogisticRegression()
        gridSearch = GridSearchCV(logReg, parameters, cv=5, scoring=scoring, refit='roc_auc') """

# Or you can generate directly.
#dataset = data_generate()
#dataset.generate_franke(n=100, noise=0.2)

"""
# Normalize the dataset
dataset.normalize_dataset()

# Fit design matrix
fitted_model = fit(dataset)

# Ordinary least square fitting
fitted_model.create_design_matrix(deg)
z_model_norm, beta = fitted_model.fit_design_matrix_numpy()

# Statistical evaluation
mse, calc_r2 = statistics.calc_statistics(dataset.z_1d, z_model_norm)
print("Mean square error: ", mse, "\n", "R2 score: ", calc_r2)

# Scale back the dataset
rescaled_dataset = dataset.rescale_back(z = z_model_norm)
#x_model = rescaled_dataset[0]
#y_model = rescaled_dataset[1]
z_model = rescaled_dataset[2]

# Generate analytical solution for plotting purposes
analytical = data_generate()
analytical.generate_franke(n, noise=0)

# Plot solutions and analytical for comparison
plot_3d(dataset.x_unscaled, dataset.y_unscaled, z_model, analytical.x_mesh, analytical.y_mesh, analytical.z_mesh, ["surface", "scatter"])

"""
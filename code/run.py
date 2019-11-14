import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os


from data_generation import data_generate, credit_card
from fit_matrix import fit
from visualization import plot_3d, plot_sigmoid, plot_traits
import statistical_functions as statistics
from sampling_methods import sampling




showplot = False
saveplot = False

iterations = 1000


print("Loading credit card data...")

#Load credit card data
dataset = credit_card()
dataset.load_credit_card_data()
#Split test/training data
dataset.split_training_test_sklearn()
#Preprocessing, removes invalid data and normalizes.
dataset.preprocess_data()

#Make a more compact version DF, of the dataframe df
DF = dataset.df 
DF = DF.drop(columns=["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"])
DF = DF.drop(columns=["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"])
DF = DF.drop(columns=["PAY_0", "PAY_2", "PAY_3","PAY_4", "PAY_5","PAY_6"])

#Print and plot some info about data.
statistics.print_info_dataframe(dataset.df, DF)
statistics.print_info_input_output(dataset.XTrain, dataset.yTrain)
plot_traits(DF,show=showplot, save=saveplot)



model = fit(dataset)
model.fit_logistic_regression(delta = 0.0001, iterations=iterations)
model.test_logistic_regression(data="test")



plt.title("Evolution of the accuracy score.")
plt.plot(np.linspace(1,iterations, iterations), model.training_score)
plt.show()


print("Score, own model: ", statistics.calc_accuracy(pred = model.prediction_test, target = model.y_test_target))

model.fit_logistic_regression_sklearn()



"""
## Neural network - not done yet.

#input - > hidden - > output
hidden_layers = 2
nodes_hidden = 30

NN = fit(dataset)
NN.fit_neural_network(number_of_hidden_nodes=30)

"""

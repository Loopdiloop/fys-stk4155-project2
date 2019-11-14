from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import sys

from scipy.special import expit
import seaborn as sns




def plot_sigmoid(df, lr, x_string):
    sigmoid_function = expit(df[x_string] * lr.coef_[0][0] + lr.intercept_[0]).ravel()
    plt.plot(df[x_string], sigmoid_function)
    plt.scatter(df[x_string], df['y'], c=df['y'], cmap='rainbow', edgecolors='b')


def plot_traits(df, show = False, save = False):
    #df["defaultPaymentNextMonth"] += 0.00001*np.random.rand(28497)
    if show != False or save != False:
        sns.pairplot(df)#, hue="defaultPaymentNextMonth")#, height=2.0)
        #sns.lmplot(x="EDUCATION", y="LIMIT_BAL", hue="defaultPaymentNextMonth", data=df) #, height=2.0)
        if show == True:
            plt.show()
        if save == True:
            plt.savefig("df_seaborn_plots.png")


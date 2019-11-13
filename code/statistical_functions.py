import numpy as np
from sklearn.metrics import r2_score as r2

def calc_accuracy(pred, target):
    pred = np.rint(pred)
    n = len(target)
    #return np.average(abs(target-pred))
    score = 0
    tol = 1e-7
    for j in range(n):
        if abs(abs(pred[j]) - abs(target[j])) <= tol:
            score += 1
    return score/n



def calc_MSE(z, z_tilde):
    mse = 0
    n=len(z)
    for i in range(n):
        mse += (z[i] - z_tilde[i])**2
    return mse/n


def calc_R2_score(z, z_tilde):
    mse = 0
    ms_avg = 0
    n=len(z)
    mean_z = np.mean(z)
    for i in range(n):
        mse += (z[i] - z_tilde[i])**2
        ms_avg += (z[i] - mean_z)**2
    return 1. - mse/ms_avg

def calc_R2_score_sklearn(z, z_tilde):
    return r2(z, z_tilde)

def calc_bias_variance(z, z_tilde):
    """ Calculate the bias and the variance of a given model"""
    n = len(z)
    Eztilde = np.mean(z_tilde)
    bias = 1/n * np.sum((z - Eztilde)**2)
    variance = 1/n * np.sum((z_tilde - Eztilde)**2)
    return bias, variance


def print_mse(mse):
    print("Average mse: ", np.average(mse))
    print("Best mse: ", np.min(mse[np.argmin(np.abs(np.array(mse)))]))

def print_R2(R2):
    print("Average R2: ", np.average(R2))
    print("Best R2: ", R2[np.argmax(np.array(R2))])

def print_bias_variance(bias_variance):
    print("Average bias-variance: ", np.average(bias_variance))
    print("Best bias-variance: ", R2[np.argmax(np.array(bias_variance))])


def calc_statistics(z, z_tilde):
    mse = calc_MSE(z, z_tilde)
    calc_r2 = calc_R2_score(z, z_tilde)
    bias_variance = calc_bias_variance(z,z_tilde)
    return mse, calc_r2, bias_variance

def print_info_dataframe(df, DF):

    correlation_compact = DF.corr()
    shape_df = df.shape
    len_input_param = shape_df[1]-1
    len_users = shape_df[0]

    print("\n")
    print("Info:                    ")
    print(df.info())
    print("\n")
    print("Correlation:             ", correlation_compact)
    print("\n")
    print("Shape of df:             ", shape_df)
    print("Input parameters:        ", len_input_param)
    print("Users:                   ", len_users)

    print("\n")

def print_info_input_output(X, y):

    shape_X = X.shape
    shape_y = y.shape


    print("\n")
    print("Input shape X:           ", shape_X)
    print("Output shape y:          ", shape_y)
    print("\n")
    print("X: ", X)
    print("\n")


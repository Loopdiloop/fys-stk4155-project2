import numpy as np
import sys
from sklearn import preprocessing
from sklearn.linear_model import Lasso

import statistical_functions as statistics
from functions import sigmoid

class fit():
    def __init__(self, inst): 
        self.inst = inst

    def create_design_matrix_simple(self, x=0, y=0, z=0, N=0, deg=6):
        """ Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
        Input is x and y mesh or raveled mesh, keyword argument deg is the degree of the polynomial you want to fit. """
        """
        if type(x) == int:
            x = self.inst.x_1d
            y = self.inst.y_1d
            z = self.inst.z_1d
            N = self.inst.N

        self.x = x
        self.y = y
        self.z = z"""

        #self.l = int((deg + 1)*(deg + 2) / 2)		# Number of elements in beta
        X = np.ones((N, deg))

        for i in range(1, deg + 1):
            #q = int( i * (i + 1) / 2)
            for k in range(i + 1):
                X[i,k] = i*k
                    
        #Design matrix
        self.X = X
        return X


    def create_design_matrix_polynomial(self, x=0, y=0, z=0, N=0, deg=17):
        """  MAKE
        Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
        Input is x and y mesh or raveled mesh, keyword argument deg is the degree of the polynomial you want to fit. """
        
        if type(x) == int:
            x = self.inst.x_1d
            y = self.inst.y_1d
            z = self.inst.z_1d
            N = self.inst.N

        self.x = x
        self.y = y
        self.z = z

        self.l = int((deg + 1)*(deg + 2) / 2)		# Number of elements in beta
        X = np.ones((N, self.l))

        for i in range(1, deg + 1):
            q = int( i * (i + 1) / 2)
            for k in range(i + 1):
                X[:, q + k] = x**(i - k) + y**k
                    
        #Design matrix
        self.X = X
        return X

    def create_design_matrix_logistic(self, x=0, y=0, z=0, N=0, deg=17):
        """  MAKE
        Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
        Input is x and y mesh or raveled mesh, keyword argument deg is the degree of the polynomial you want to fit. """
        
        if type(x) == int:
            x = self.inst.x_1d
            y = self.inst.y_1d
            z = self.inst.z_1d
            N = self.inst.N

        self.x = x
        self.y = y
        self.z = z

        self.l = int((deg + 1)*(deg + 2) / 2)		# Number of elements in beta
        X = np.ones((N, self.l))

        for i in range(1, deg + 1):
            q = int( i * (i + 1) / 2)
            for k in range(i + 1):
                X[:, q + k] = x**(i - k) + y**k
                    
        #Design matrix
        self.X = X
        return X

    def fit_design_matrix_logistical_sklearn(self):
        """Fitting matrix from sklearn logistical fitting. """
        X = self.X
        z = self.z
        
        beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z)
        
        y_tilde = X @ beta
        return y_tilde, beta

    
    def fit_design_matrix_numpy(self):
        """Method that uses the design matrix to find the coefficients beta, and
        thus the prediction y_tilde"""
        X = self.X
        z = self.z
        
        beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z)
        
        y_tilde = X @ beta
        return y_tilde, beta

    def fit_design_matrix_ridge(self, lambd):
        """Method that uses the design matrix to find the coefficients beta with 
        the ridge method, and thus the prediction y_tilde"""
        X = self.X
        z = self.z

        beta = np.linalg.pinv(X.T.dot(X) + lambd*np.identity(self.l)).dot(X.T).dot(z)
        y_tilde = X @ beta
        return y_tilde, beta

    def fit_design_matrix_lasso(self, lambd):
        """The lasso regression algorithm implemented from scikit learn."""
        lasso = Lasso(alpha = lambd, max_iter = 10e5, tol = 0.01, normalize= (not self.inst.normalized), fit_intercept=(not self.inst.normalized))
        lasso.fit(self.X,self.z)
        beta = lasso.coef_
        y_tilde = self.X@beta
        return y_tilde, beta

    

    def fit_design_matrix_logistic(self, lambd):
        """Method that uses the design matrix to find the coefficients beta with 
        the ridge method, and thus the prediction y_tilde"""
        X = self.X
        z = self.z

        beta = np.ones(n)

        #beta = np.linalg.pinv(X.T.dot(X) + lambd*np.identity(self.l)).dot(X.T).dot(z)

        for i in iterations:
            y_tilde = X @ beta
            p = sigmoid(y_tilte)

            #Gradient from dC(beta)/dbeta
            gradient = -X.T *(y - p)

            #deriv_gradient = x.T * np.diag(p())W * X
            deriv_gradient = 0.1
            # Iff acceleration is zero, then no movement. If steep, move a lot. Only works if only global minimum.
            beta -= dx * deriv_gradient* gradient
        
        
        return y_tilde, beta

    def test_design_matrix(self, beta, X = 0):
        """Testing the design matrix with a beta and"""
        if isinstance(X, int):
            X = self.X
        y_tilde = X @ beta
        return y_tilde
        

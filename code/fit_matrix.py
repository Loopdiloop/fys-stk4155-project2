import numpy as np
import sys
from sklearn import preprocessing
from sklearn.linear_model import Lasso, LogisticRegression


import statistical_functions as statistics
from functions import sigmoid

class fit():
    def __init__(self, inst): 
        self.inst = inst


    def fit_neural_network(self, iterations =10, hidden_layers = 1, number_of_hidden_nodes = 30):
        """ Do a feed forward neural network. Only one hidden layer this far :) """
        
        # n = no. of users, p = predictors, i.e. parameters.
        n,p = np.shape(self.inst.XTrain)
        k = number_of_hidden_nodes

        input_nodes = self.inst.XTrain
        #input_nodes = input_nodes.T
        expected_output = self.inst.yTrain

        hidden_nodes = np.ones((k))
        weights_a = np.ones((p,k)) #From input to hidden
        #weights_b = np.ones((hidden_nodes, 2)) #If more hidden layers??
        weights_c = np.ones((k, 2)) #From hidden nodes to output

        bias_hidden = np.ones(k)
        bias_output = np.ones(2)
        print("\n \n")
        for i in range(iterations):
            
            #Calc the activation for data form user i:
            hidden_nodes = np.dot(weights_a.T, sigmoid(input_nodes[i,:])) + bias_hidden
            output = np.dot(weights_c.T, sigmoid(hidden_nodes)) + bias_output

            dC_dw = (output - expected_output)* output*(1-output)*hidden_nodes
            
            dC_da = 
            output_error_hidden = sigmoid(hidden_layers)*dC_da
            output_error_output =

        """
        X = np.ones((n,p+1))
        X[:,1:] = self.inst.XTrain
        X[:,0] = 1

        y = self.inst.yTrain

        beta = np.random.uniform(0,1,size=(p+1,1))*5
        score = np.zeros(iterations)
        
        #print(beta)
        for i in range(iterations):
            pred = sigmoid(np.dot(X, beta))
            dCost_dBeta = np.dot(-X.T, y-pred)
            beta -= delta * dCost_dBeta
            score[i] = np.average(abs(y-pred))
        #print("SCORE = ", score[-1])

        self.beta = beta
        self.n = n
        self.p = p

        self.training_score = score"""

        #return beta, score


    def fit_logistic_regression(self, delta=0.01, iterations = 1000):
        """ Do a linear, logistic fit for matrix X with sigmoid function"""
        
        # n = no. of users, p = predictors, i.e. parameters.
        n,p = np.shape(self.inst.XTrain)

        X = np.ones((n,p+1))
        X[:,1:] = self.inst.XTrain
        X[:,0] = 1

        y = self.inst.yTrain

        beta = np.random.uniform(0,1,size=(p+1,1))*5
        score = np.zeros(iterations)
        
        #print(beta)
        for i in range(iterations):
            pred = sigmoid(np.dot(X, beta))
            dCost_dBeta = np.dot(-X.T, y-pred)
            beta -= delta * dCost_dBeta
            score[i] = np.average(abs(y-pred))
        #print("SCORE = ", score[-1])

        self.beta = beta
        self.n = n
        self.p = p

        self.training_score = score

        return beta, score

    def fit_logistic_regression_sklearn(self, delta=0.01, iterations = 1000):
        """ Do a linear, logistic fit for matrix X with sigmoid function, from sklearn"""
        
        # n = no. of users, p = predictors, i.e. parameters.
        n,p = np.shape(self.inst.XTrain)

        X = np.ones((n,p+1))
        X[:,1:] = self.inst.XTrain
        X[:,0] = 1

        y = self.inst.yTrain

        skl_reg = LogisticRegression(solver='lbfgs') #solver="lbfgs")
        skl_reg.fit(X,y)

        n,p = np.shape(self.inst.XTest)
        X_test = np.ones((n,p+1))
        X_test[:,1:] = self.inst.XTest
        y_test = self.inst.yTest
        X_test[:,0] = 1

        score = skl_reg.score(X_test, self.inst.yTest)
        print("SKLEARN SCORE: ", score)


    def test_logistic_regression(self, data="test", beta=0):
        """ Do a linear, logistic fit for matrix X with sigmoid function"""
        
        # n = no. of users, p = predictors, i.e. parameters.

        if beta == 0:
            beta = self.beta
        
        if data == "test":
            n,p = np.shape(self.inst.XTest)
            X = np.ones((n,p+1))
            X[:,1:] = self.inst.XTest
            y = self.inst.yTest
        elif data == "train":
            n,p = self.n, self.p
            X[:,1:] = self.inst.XTrain
            y = self.inst.yTrain
        else:
            raise ValueError ("Error. Wrong input for data = test or training, not %s ." % data)
        X[:,0] = 1

        pred = sigmoid(np.dot(X, beta))
        print(abs(y-pred))
        test_score = np.average(abs(y-pred))

        #print("SCORE OWN MODEL= ", test_score)
        self.prediction_test = pred
        self.y_test_target = y
        self.test_score = test_score
        return test_score





















































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



    def fit_design_matrix_logistical_sklearn(self):
        """Fitting matrix from sklearn logistical fitting. """
        X = self.X
        z = self.z
        
        beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z)
        
        y_tilde = X @ beta

        
        lambdas=np.logspace(-5,7,13)
        parameters = [{'C': 1./lambdas, "solver":["lbfgs"]}]#*len(parameters)}]
        scoring = ['accuracy', 'roc_auc']
        logReg = LogisticRegression()
        gridSearch = GridSearchCV(logReg, parameters, cv=5, scoring=scoring, refit='roc_auc')


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

    



    def test_design_matrix(self, beta, X = 0):
        """Testing the design matrix with a beta and"""
        if isinstance(X, int):
            X = self.X
        y_tilde = X @ beta
        return y_tilde
        

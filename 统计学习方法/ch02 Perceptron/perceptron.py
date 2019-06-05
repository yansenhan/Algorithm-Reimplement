'''
Author: Yansen Han
Date: June 05, 2019
Email: yansen.han@stonybrook.edu
'''

import numpy as np

class perceptron():
    '''
    This class contains two kinds of perceptrons, prime form and dual form.
    
    Parameters:
    -- learning_rate: learning rate. default: 0.1

    Example 1 (Prime Form):
        X = np.array([[3, 3], [4, 3], [1, 1]])
        Y = np.array([1, 1, -1])

        perceptron = perceptron(learning_rate=1)
        perceptron.fit_prime(X, Y)
        perceptron._get_coeff()
        perceptron.prime_predict(X)

    Example 2 (Dual Form):
        X = np.array([[3, 3], [4, 3], [1, 1]])
        Y = np.array([1, 1, -1])

        perceptron = perceptron(learning_rate=1)
        perceptron.fit_dual(X, Y)
        perceptron._get_coeff()
        perceptron.dual_predict(X, Y)
    '''
    def __init__(self, learning_rate=0.1):     
        self.learning_rate = learning_rate
    
    def check_true(self, X, Y, w, b):
        '''
        check whether there are some mis-classified points
        '''
        temp1 = np.squeeze(w.T.dot(X.T)) + b
        temp1 = np.apply_along_axis(lambda x: -1 if x <= 0 else 1, 1, temp1.reshape((len(temp1), 1)))
        temp = np.sum(abs(temp1 - np.squeeze(Y)))
        if temp > 0:
            return False
        else:
            return True
        
    def check_dual_true(self, X, y, parameter, b):
        '''
        check whether there are some mis-classified points
        '''
        temp1 = y * (np.squeeze(parameter.dot(X.T)) + b)
        temp1 = np.apply_along_axis(lambda x: -1 if x <= 0 else 1, 1, temp1.reshape((len(temp1), 1)))
        temp = np.sum(abs(temp1 - np.squeeze(Y)))
        if temp > 0:
            return False
        else:
            return True
        
    def fit_prime(self, X, Y):
        '''
        Parameters:
        -- X: [Matrix] each row denotes one sample
        -- Y: [Vector] each row denotes the corresponding label
        Return: object
        '''
        # Initialization
        Y = np.squeeze(Y)
        w = np.zeros((np.shape(X)[1], 1))
        b = 0
        counter = 0
        num_sample = np.shape(X)[0]
        
        # Training
        while True:
            row = counter % num_sample
            if Y[row] * (w.T.dot(X[row].T) + b) <= 0:
                add1 = self.learning_rate * Y[row] * X[row]
                w = w + add1.reshape(np.shape(w))
                b = b + self.learning_rate * Y[row]
            counter = counter + 1
            if self.check_true(X, Y, w, b) == True:
                self._coeff = w
                self._intercept = b
                return self
            
    def fit_dual(self, X, Y):
        '''
        Parameters:
        -- X: [Matrix] each row denotes one sample
        -- Y: [Vector] each row denotes the corresponding label
        Return: object
        '''
        # Initialization
        Y = np.squeeze(Y)
        a = np.zeros(np.shape(X)[0])
        b = 0
        counter = 0
        num_sample = np.shape(X)[0]
        
        # Training
        while True:
            row = counter % num_sample
            parameter = np.zeros(np.shape(X)[1])
            for i in range(num_sample):
                parameter += a[i] * Y[i] * X[i]
            if Y[row] * (parameter.dot(X[row].T) + b) <= 0:
                a[row] = a[row] + self.learning_rate
                b = b + self.learning_rate * Y[row]
            counter = counter + 1
            if self.check_dual_true(X, Y[row], parameter, b) == True:
                self._coeff = a
                self._intercept = b
                return self
            
    def _get_coeff(self):
        print("Coefficients: {},\n Intercept: {}".format(np.squeeze(self._coeff), self._intercept))
        return self._coeff, self._intercept
            
    def prime_predict(self, X):
        temp = self._coeff.T.dot(X.T)
        temp = temp + np.ones(temp.shape) * self._intercept
        temp = np.squeeze(temp)
        out = np.apply_along_axis(lambda x: 1 if x>=0 else -1, 1, temp.reshape((len(temp),1)))
        return out  
    
    def dual_predict(self, X, Y):
        num_sample = np.shape(X)[0]
        parameter = np.zeros(np.shape(X)[1])
        for i in range(num_sample):
            parameter += self._coeff[i] * Y[i] * X[i]
            
        temp = parameter.dot(X.T)
        temp = temp + np.ones(temp.shape) * self._intercept
        temp = np.squeeze(temp)
        out = np.apply_along_axis(lambda x: 1 if x>=0 else -1, 1, temp.reshape((len(temp),1)))
        return out 
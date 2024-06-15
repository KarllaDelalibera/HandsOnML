import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=1.0e-2, iterations=10000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = 0
        self.b = 0
    
    def compute_cost(self, X, y):
        """
        Computes the cost function (mean squared error) for linear regression.

        Args:
          x (numpy.ndarray): data
          y (numpy.ndarray): target

        Returns
            total_cost (float): The total cost of using w,b as the parameters
            for linear regression to fit the data points in x and y.
        """

        n = X.shape[0]
        f_wb = self.w * X + self.b
        cost_sum = np.sum((f_wb - y) ** 2)
        cost = (1 / (2 * n)) * cost_sum

        return cost

    def compute_gradient(self, X, y):
        """
        Computes the gradient for linear regression.

        Args:
          x (numpy.ndarray): data
          y (numpy.ndarray): target

        Returns
          dj_dw (float): The gradient of w
          dj_db (float): The gradient of b     
         """

        n = X.shape[0]
        f_wb = self.w * X + self.b
        error = f_wb - y
        dj_dw = (1 / n) * np.sum(error * X)
        dj_db = (1 / n) * np.sum(error)

        return dj_dw, dj_db

    def fit(self, X, y):
        """
        Performs gradient descent to fit w,b.

        Args:
          x (numpy.ndarray): data 
          y (numpy.ndarray): target

        Returns:
          w (float): fit value of w
          b (float): fit value of b
          J_cost_values (List): List of cost values
          theta_values (list): List of parameters [w,b] 
        """
        
        J_cost_values = []
        theta_values = []

        for i in range(self.iterations):
            dj_dw, dj_db = self.compute_gradient(X, y)
            
            self.w  = self.w - self.learning_rate * dj_dw
            self.b  = self.b - self.learning_rate * dj_db
            
            if i < 100000:
                J_cost_values.append(self.compute_cost(X, y))
                theta_values.append([self.w, self.b])
        
        return self.w, self.b, J_cost_values, theta_values

    def predict(self, X):
        """
        Make predictions using the fitted parameters of the model.

        Args:
          x (numpy.ndarray): data

        Returns:
          numpy.ndarray: model predictions
        """
        return self.w * X + self.b

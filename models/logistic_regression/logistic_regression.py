import numpy as np


def sigmoid(z):
    """
    Compute the sigmoid of z.
    """
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, w, b):
    """
    Computes the cost function (cross rntropy) for logistic regression.
    Args:
      X (numpy.ndarray): data, m examples with n features
      y (numpy.ndarray) : target values
      w (numpy.ndarray) : model parameters
      b (float)       : model parameter
    Returns:
      total_cost (float): cost
    """
    m = X.shape[0]
    cost = 0.0

    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        loss_i = -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i) 
        cost =  cost + loss_i
    
    total_cost = cost / m
    
    return total_cost

        
def compute_gradient(X, y, w, b):
    """
    Computes the gradient for logistic regression.
    Args:
      x (numpy.ndarray): data
      y (numpy.ndarray): target
    Returns
      dj_dw (float): The gradient of w
      dj_db (float): The gradient of b     
     """
    m,n = X.shape
    dj_db = 0
    dj_dw = np.zeros((n,))

    for i in range(m):
        z_i = np.dot(X[i], w) +  b
        f_wb_i = sigmoid(z_i)
        error_i  = f_wb_i  - y[i]
        dj_db = dj_db + error_i
    
        for j in range(n):
            aux = error_i * X[i,j]
            dj_dw[j] = dj_dw[j] + aux
    
    dj_dw = dj_dw/m
    dj_db = dj_db/m 
    
    return dj_dw, dj_db


def fit(X, y, learning_rate, iterations):
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
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        
        w  = w - learning_rate * dj_dw
        b  = b - learning_rate * dj_db
        
        if i < 100000:
            J_cost_values.append(compute_cost(X, y, w, b))
    
    return w, b, J_cost_values


def predict(X, w, b):
    """
    Make predictions using the fitted parameters of the model.
    Args:
      X (numpy.ndarray): data
    Returns:
      numpy.ndarray: model predictions
    """

    m = X.shape[0]
    y_pred = np.zeros(m)
    
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        y_pred[i] = sigmoid(z_i)
    return y_pred

import numpy as np
import util
import math
from linear_model import LinearModel

train_path = "/home/aryan/Desktop/CS229-Solutions/ps1/data/ds1_train.csv"
eval_path = "/home/aryan/Desktop/CS229-Solutions/ps1/data/ds1_valid.csv"
pred_path = "/home/aryan/Desktop/CS229-Solutions/ps1/data/ds1_pred.csv"
def main(train_path, eval_path, pred_path):
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    clf = LogisticRegression()
    theta = clf.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = clf.predict(x_eval, theta)
    #print(y_pred)
    #util.plot(x_train,y_train,theta)


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.
    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        m,n = x.shape
        theta = np.zeros(n)
        theta_prev = np.ones(n)
        while(np.linalg.norm((theta-theta_prev),2)>=0.00001):
            theta_prev = theta
            H,Div_Theta = Cal_back(x,y,theta)
            theta = theta - np.transpose(np.matmul(np.linalg.inv(H),np.transpose(Div_Theta)))
        return theta
    def predict(self, x, theta):
        return 1/(1 + np.exp(-np.dot(x, theta)))  


def Cal_back(x,y,theta):
    m,n = x.shape
    H = np.ones((n,n))
    for j in range(0,n):
        for l in range(0,n):
            sum = 0
            for i in range(0,m):
                sum += h_theta(x[i], theta)*(1-h_theta(x[i],theta))*x[i][j]*x[i][l]
            sum = sum/m
            H[j][l] = sum
    Div_Theta = np.ones(n)
    for j in range(0,n):
        sum = 0
        for i in range(0,m):
            sum+=(y[i]-h_theta(x[i], theta))*x[i][j]
        sum = -sum/m
        Div_Theta[j] = sum
    return H, Div_Theta 

def h_theta(x, theta):
        t = np.matmul(theta,np.transpose(x))
        
        h = 1/(1+math.exp(-t))
        return h

main(train_path, eval_path,pred_path)
import numpy as np
import util

from linear_model import LinearModel

train_path = "/home/aryan/Desktop/CS229-Solutions/ps1/data/ds1_train.csv"
eval_path = "/home/aryan/Desktop/CS229-Solutions/ps1/data/ds1_valid.csv"
pred_path = "/home/aryan/Desktop/CS229-Solutions/ps1/data/ds1_pred.csv"


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    clf = GDA()
    print()
    clf.fit(x_train,y_train)
    # *** START CODE HERE ***
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m,n = x.shape
        count_o = np.dot(np.ones(m),y)
        count_z = np.dot(np.ones(m),np.ones(m)-y)
        phi = count_o/m
        mu_1 = np.matmul(np.transpose(x),y)/count_o
        mu_0 = np.matmul(np.transpose(x),np.ones(m)-y)/count_z
        cov_m = np.zeros((n,n))
        for i in range(0,m):
            if(y[i] == 0):
                cov_m = cov_m+np.matmul((x[i]-mu_0).reshape((-1, 1)),(x[i]-mu_1).reshape((1, -1)))
            if(y[i] == 1):
                cov_m = cov_m+np.matmul((x[i]-mu_1).reshape((-1, 1)),(x[i]-mu_1).reshape((1, -1)))
        cov_m = cov_m/m
        theta = -np.matmul(np.linalg.inv(cov_m),mu_0)+np.matmul(np.linalg.inv(cov_m),mu_1)
        cov_m_inv = np.linalg.inv(cov_m)
        theta_zero = 1/2*mu_0 @ cov_m_inv @ mu_0 -1/2*mu_1 @ cov_m_inv @ mu_1 - np.log((1-phi)/phi)
        theta = np.insert(theta,0,theta_zero)
        util.plot(x,y,theta)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE
main(train_path, eval_path,pred_path)
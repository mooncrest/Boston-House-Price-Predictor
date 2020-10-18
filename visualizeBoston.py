from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data():
    boston = load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features

def visualize(X, y, features):
    fig, table = plt.subplots(nrows=3, ncols=5, gridspec_kw={'hspace': .7})
    fig.delaxes(table[2][3])
    fig.delaxes(table[2][4])
    index = -1
    for i in range(13):
        if (i % 5 == 0):
            index += 1
        plot(table[index][i % 5], i, X, y, features)
    plt.tight_layout()
    plt.show()

def plot(ax, dim, X, y, features):
    column = np.array([data[dim] for data in X])
    ax.scatter(column, y, s=.7)
    ax.set_xlabel(f"{features[dim]} data")
    ax.set_ylabel("target")
    ax.set_title(f"{features[dim]}")

def fit_regression(X, Y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    XTX = np.matmul(X.transpose(), X)
    XTY = np.matmul(X.transpose(), Y)
    weights = np.linalg.solve(XTX, XTY)


    error = np.matmul(X, weights) - Y
    bias = -sum(error)/len(error)
    return weights, bias

def compute_MSE(W, bias, data_test, label_test):
    diff = np.linalg.norm(np.matmul(data_test, W) + bias \
                       * np.ones(len(data_test)) - label_test) ** 2

    return diff / len(label_test)

def compute_MAE(W, bias, data_test, label_test):
    abs_diff = sum(np.absolute(np.matmul(data_test, W) + bias \
                       * np.ones(len(data_test)) - label_test))

    return abs_diff / len(label_test)

def compute_R2(W, bias, data_test, label_test):
    MODEL = np.linalg.norm(np.matmul(data_test, W) + bias \
                       * np.ones(len(data_test)) - label_test) ** 2


    BASELINE = np.linalg.norm(label_test - np.ones(len(label_test)) \
                                                 * sum(label_test)/len(label_test)) ** 2
 
    return 1 - MODEL / BASELINE

def print_weights(features, W, extra=""):
    print(f"=={extra}Features and Weights==")
    for ind, X in enumerate(zip(features, W)):
        print(f"{ind: <2}", f"{X[0]: <8}", X[1])

def print_adjusted_weights(features, W, X):
    adjusted = [i for i in W]
    for i in range(len(features)):
        mean_data = sum(vec[i] for vec in X) / len(X)
        adjusted[i] *= mean_data
    print_weights(features, adjusted, "adjusted ")
    

def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    
    # Visualize the features
    visualize(X, y, features)

    #TODO: Split data into train and test
    data_train, data_test, label_train, label_test = \
                    train_test_split(X, y, test_size=.2, train_size=.8, random_state=12)

    # Fit regression model
    w, b = fit_regression(X, y)
    print_weights(features, w)
    print_adjusted_weights(features, w, X)
    # Compute fitted values, MSE, etc.
    print("MSE:", compute_MSE(w, b, data_test, label_test))
    print("R^2:", compute_R2(w, b, data_test, label_test))
    print("MAE:", compute_MAE(w, b, data_test, label_test))



if __name__ == "__main__":
    main()

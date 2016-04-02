"""A driver module for running five machine learning algorithms on a breast
cancer data set.

Initial source:
https://github.com/joshuamorton/Machine-Learning/blob/master/P1/Adults/adult.py
"""

from matplotlib import rc
import matplotlib.pyplot as plt

import numpy as np

from pybrain.datasets.classification import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.validation import CrossValidator
from pybrain.supervised.trainers import BackpropTrainer

from sklearn import tree, neighbors, ensemble, cross_validation
from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from time import localtime, strftime

import warnings


# For sanity's sake, filter out useless deprecation warnings.
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Set styles for MatPlotLib to be LaTeX-friendly.
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


in_file = "wdbc.data.txt"


def load_data(filename):
    with open(filename) as data:
        # Initialize the input with the titles of each column.
        data_list = ["mean-radius,mean-texture,mean-perimeter,mean-area,mean-smoothness,mean-compactness, mean-concavity,mean-concave-points,mean-symmetry,mean-fractal-dimension,SE-radius,SE-texture,SE-perimeter,SE-area,SE-smoothness,SE-compactness, SE-concavity,SE-concave-points,SE-symmetry,SE-fractal-dimension,worst-radius,worst-texture,worst-perimeter,worst-area,worst-smoothness,worst-compactness, worst-concavity,worst-concave-points,worst-symmetry,worst-fractal-dimension,diagnosis"]

        # Do not add files with unknown data (denoted with "?"), and replace
        # "M" (malignant) with 1 and "B" (benign) with 0. This is to make the
        # input data entirely numerical, and NumPy understands 0=False 1=True.
        for line in data.readlines():
            if '?' not in line:
                if 'M' in line:
                    to_add = str(line[line.find('M') + 2:]).strip() + ",1"
                else:
                    to_add = str(line[line.find('B') + 2:]).strip() + ",0"
                data_list.append(to_add)

    return np.loadtxt(data_list,
                      delimiter=',',
                      dtype='u4',
                      skiprows=1
                      )


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    (This method taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)

    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation (test) score")

    plt.legend(loc="best")
    return plt

def initialize():
    """
    tx - training x axes
    ty - training y axis
    rx - result (testing) x axes
    ry - result (testing) y axis
    """
    full_data = load_data(in_file)
    tr, te = train_test_split(full_data, train_size = 0.7)
    tx, ty = np.hsplit(tr, [30])
    rx, ry = np.hsplit(te, [30])
    ty = ty.flatten()
    ry = ry.flatten()
    return tx, ty, rx, ry

def knnTest(tx, ty, rx, ry):
    print "KNN start"
    print strftime("%a, %d %b %Y %H:%M:%S", localtime())
    estimator = neighbors.KNeighborsClassifier(weights='distance')
    cv = ShuffleSplit(tx.shape[0], n_iter=10, test_size=0.2)
    k = np.logspace(0, 1.5, 20, dtype=int)
    classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(n_neighbors=k))
    classifier.fit(tx, ty)
    title = 'k Nearest Neighbors ( best k = %s)' % classifier.best_estimator_.n_neighbors

    estimator = neighbors.KNeighborsClassifier( n_neighbors=classifier.best_estimator_.n_neighbors)

    plot_learning_curve(estimator, title, tx, ty, cv=cv)
    plt.savefig('knn.png', dpi=500)
    print "Classifier score:", classifier.score(rx, ry)
    print "Best k was", classifier.best_estimator_.n_neighbors
    print "All scores:"
    print classifier.grid_scores_
    with open("knn_results.txt", 'w') as f:
        f.write("Best:" + str(classifier.best_estimator_.n_neighbors))
        f.write("All classifier grid scores:\n\n" + str(classifier.grid_scores_))
    print "KNN end"
    print strftime("%a, %d %b %Y %H:%M:%S", localtime())

def svmTest(tx, ty, rx, ry):
    print "SVM start"
    print strftime("%a, %d %b %Y %H:%M:%S", localtime())
    estimator = SVC()
    cv = ShuffleSplit(tx.shape[0], n_iter=10, test_size=0.2)
    kernels = ["sigmoid", "rbf"]
    classifier = GridSearchCV(
        estimator=estimator,
        cv=cv,
        param_grid=dict(kernel=kernels))

    classifier.fit(tx, ty)
    title = 'SVM ( best kernel = %s)' % (classifier.best_estimator_.kernel)

    estimator = SVC(kernel=classifier.best_estimator_.kernel)

    plot_learning_curve(estimator, title, tx, ty, cv=cv)
    estimator.fit(tx, ty)
    plt.savefig('svm.png', dpi=500)
    print "Classifier score:", classifier.score(rx, ry)
    print "Best number of estimators was", classifier.best_estimator_.kernel
    print "All scores:"
    print classifier.grid_scores_
    with open("svm_results.txt", 'w') as f:
        f.write("Best:" + str(classifier.best_estimator_.kernel))
        f.write("All classifier grid scores:\n\n" + str(classifier.grid_scores_))
    print "SVM end"
    print strftime("%a, %d %b %Y %H:%M:%S", localtime())


def boostTest(tx, ty, rx, ry):
    print "Boost start"
    print strftime("%a, %d %b %Y %H:%M:%S", localtime())
    estimator = ensemble.AdaBoostClassifier()
    cv = ShuffleSplit(tx.shape[0], n_iter=10, test_size=0.2)
    estimators = np.logspace(0, 3, 20, dtype=int)
    classifier = GridSearchCV(
        estimator=estimator,
        cv=cv,
        param_grid=dict(n_estimators=estimators))

    classifier.fit(tx, ty)
    title = 'Ada Boost (estimators = %s)' % (classifier.best_estimator_.n_estimators)

    estimator = ensemble.AdaBoostClassifier( n_estimators=classifier.best_estimator_.n_estimators)

    plot_learning_curve(estimator, title, tx, ty, cv=cv)
    plt.savefig('boost.png', dpi=500)
    print "Classifier score:", classifier.score(rx, ry)
    print "Best number of estimators was", classifier.best_estimator_.n_estimators
    print "All scores:"
    print classifier.grid_scores_
    with open("boost_results.txt", 'w') as f:
        f.write("Best:" + str(classifier.best_estimator_.n_estimators))
        f.write("All classifier grid scores:\n\n" +  str(classifier.grid_scores_))
    print "Boost end"
    print strftime("%a, %d %b %Y %H:%M:%S", localtime())

def nnTest(tx, ty, rx, ry, iterations):
    """
    builds, tests, and graphs a neural network over a series of trials as it is
    constructed
    """
    print "NN start"
    print strftime("%a, %d %b %Y %H:%M:%S", localtime())

    """
    builds, tests, and graphs a neural network over a series of trials as it is
    constructed
    """
    resultst = []
    resultsr = []
    positions = range(iterations)
    network = buildNetwork(30, 30, 1, bias=True)
    ds = ClassificationDataSet(30, 1, class_labels=["1", "0"])
    for i in xrange(len(tx)):
        ds.addSample(tx[i], [ty[i]])
    trainer = BackpropTrainer(network, ds, learningrate=0.05)
    validator = CrossValidator(trainer, ds, n_folds=10)
    print validator.validate()
    for i in positions:
        print trainer.train()
        resultst.append(sum((np.array([round(network.activate(test)) for test in tx]) - ty)**2)/float(len(ty)))
        resultsr.append(sum((np.array([round(network.activate(test)) for test in rx]) - ry)**2)/float(len(ry)))
        print i, resultst[i], resultsr[i]
    plt.plot(positions, resultst, 'g-', positions, resultsr, 'r-')
    plt.axis([0, iterations, 0, 1])
    plt.ylabel("Percent Error")
    plt.xlabel("Network Epoch")
    plt.title("Neural Network Error")
    plt.savefig('nn.png', dpi=500)
    print "NN end"
    print strftime("%a, %d %b %Y %H:%M:%S", localtime())

def treeTest(tx, ty, rx, ry):
    print "Tree start"
    print strftime("%a, %d %b %Y %H:%M:%S", localtime())
    estimator = tree.DecisionTreeClassifier(criterion="gini")
    cv = ShuffleSplit(tx.shape[0], n_iter=10, test_size=0.2)
    max_depths = range(1, 300)
    classifier = GridSearchCV(
        estimator=estimator,
        cv=cv,
        param_grid=dict(max_depth=max_depths))

    classifier.fit(tx, ty)
    title = 'Decision Tree ( best depth = %s)' % (classifier.best_estimator_.max_depth)

    estimator = tree.DecisionTreeClassifier(max_depth=classifier.best_estimator_.max_depth)

    plot_learning_curve(estimator, title, tx, ty, cv=cv)
    plt.savefig('tree.png', dpi=500)
    print "Classifier score:", classifier.score(rx, ry)
    print "Best depth was", classifier.best_estimator_.max_depth
    print "All scores:", classifier.grid_scores_
    estimator.fit(tx, ty)
    with open("tree_results.txt", 'w') as f:
        f.write("Best:" +  str(classifier.best_estimator_.max_depth))
        f.write("All classifier grid scores:\n\n" + str(classifier.grid_scores_))
        f.write("Classifier imoportances\n\n" + str(estimator.feature_importances_))

    print "Classifier imoportances\n\n", str(estimator.feature_importances_)

    print "Tree end"
    print strftime("%a, %d %b %Y %H:%M:%S", localtime())

def main():
    # Get the testing and training values from file.
    tx, ty, rx, ry = initialize()

    # Run the tests with the testing and training values.
    knnTest(tx, ty, rx, ry)
    boostTest(tx, ty, rx, ry)
    treeTest(tx, ty, rx, ry)
    nnTest(tx, ty, rx, ry, 100)
    svmTest(tx, ty, rx, ry)

if __name__ == "__main__":
    main()

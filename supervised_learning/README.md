# Supervised learning

A run of five supervised learning algorithms over 2 data sets. For an in-depth
analysis and explanation, see the [analysis][0].

## Algorithms
Five algorithms were run. These were:

1. Adaboost;
2. _k_-nearest neighbors;
3. decision tree (Gini);
4. neural network; and
5. support vector machines.

## Data sets
Two data sets were used. These were:

1. [Bank marketing][1], from the UCI machine learning repository; and
2. [Breast cancer][2], also from the UCI machine learning repository.

The data sets are included in the repo for convenience. I do not own these
data sets. See the links for full authorship information.

## Analysis
A detailed analysis is located in the [analysis][0] directory. It is written in
LaTeX, so you'll need to have that installed, and compile `main.tex` to read it 
in its formatted glory (you could read the source code if you're really
strapped for time).

## How to run
To run this stuff, have all the Python packages listed in [requirements.txt][3]
installed, and the latest version of Python2. Venture into the respective
data set's directory, and run the similarly named python file. Then run. It
will take a while.

[0]: analysis/
[1]: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
[2]: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
[3]: requirements.txt

# Artificial Neural Networks

This part implements a custom Pytorch Regressor to solve the house value regression problem. The hyperparameters (```number of layers, number of nodes, number of epochs, batch size, optimization function```) of the Regressor have been optimized using a Grid search.

Different models (```Linear Regression, Unoptimized Regressor, Optimized Regressor```) are compared in the example main of the part2_house_value_regression.py file.

### house_value_regression.py

Running this file performs a Grid search for hyperparameter tuning, evaluate the optimal model and save it in the pickle file.

### model.pickle

The optimal Regressor model after parameter tuning is saved in the ```part2_model.pickle``` file.


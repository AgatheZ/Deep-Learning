from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

import math
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as Data
import sklearn.preprocessing as pp
import scipy.stats as stats


class Regressor(BaseEstimator):
  def __init__(
        self, 
        x, 
        nb_epoch = 50,
        loss_fn = torch.nn.MSELoss(), 
        lr = 0.3, 
        batch_size = 64, 
        n_nodes = 128,          
        n_hidden = 2, 
        activation = nn.ReLU(), 
        optimization = torch.optim.Adam
    ):
    """ 
    Initialise the model.
      
    Arguments:
        - x {pd.DataFrame} -- Raw input data of shape 
            (batch_size, input_size), used to compute the size 
            of the network.
        - nb_epoch {int} -- number of epoch to train the network.
        - loss_fn -- Loss function of the network
        - activation -- Activation function for the hidden layers of the network
        - lr {float} -- Learning rate of the model 
        - batch_size {int} -- size of the batches (number of samples processed model update)
        - n_nodes {int} -- number of nodes for the hidden layers
        - optimization -- optimization function for the neural network
    """ 

    self.x = x
    X, _ = self._preprocessor(x, training = True)
    self.input_size = X.shape[1]
    self.nb_epoch = nb_epoch
    self.n_hidden = n_hidden
    self.n_nodes = n_nodes
    self.batch_size = batch_size
    self.lr = lr
    self.optimization= optimization
    self.activation = activation
    self.loss_fn = loss_fn
    return

  def _preprocessor(self, x, y = None, training = True):
    """
    Preprocess input of the network.
    Arguments:
      - x {pd.DataFrame} -- Raw input array of shape 
          (batch_size, input_size).
      - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
      - training {boolean} -- Boolean indicating if we are training or 
        testing the model.

    Returns:
        - {torch.tensor}  -- Preprocessed input array of
          size (batch_size, input_size).
        - {torch.tensor}  -- Preprocessed target array of
          size (batch_size, 1).

    """
    

    #Separate the data based on their type
    X_num = x.select_dtypes(include=np.number)
    X_object = x.select_dtypes(include=object)

    #Filling missing values with column mean + normalizing data
    X_num = X_num.fillna(value = X_num.mean())
    X_num = preprocessing.normalize(X_num, axis = 0) 

    #Train the one hot encoder on the training data 
    if training == True:
      self.onehot = preprocessing.LabelBinarizer()
      self.onehot.fit(X_object)

    #Transforms categorical data
    X_object = self.onehot.transform(X_object)
    X_num = X_num.astype('float32')
    X_object = X_object.astype('float32')

    #Outputs are tensors 
    X = torch.from_numpy(np.concatenate((X_num, X_object), axis=1))
    Y = (torch.from_numpy(y.to_numpy().astype('float32')) if isinstance(y, pd.DataFrame) else None)  
        
    #Return preprocessed data
    return X, Y

  def fit(self, x, y):
      """
      Regressor training function
      Arguments:
          - x {pd.DataFrame} -- Raw input array of shape 
              (batch_size, input_size).
          - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

      Returns:
          self {Regressor} -- Trained model.
      """
      X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
      torch_dataset = torch.utils.data.TensorDataset(X, Y)
      data_loader = torch.utils.data.DataLoader(dataset=torch_dataset,batch_size=self.batch_size, shuffle=True)
      self.losses = []
      #Definition of the model 
      layers = []
      layers.append(nn.Linear(X.shape[1], self.n_nodes)) #input layer
      layers.append(self.activation)
        
      for layer in range(self.n_hidden):  #hidden layers
          layers.append(nn.Linear(self.n_nodes, self.n_nodes))
          layers.append(self.activation)
        
      layers.append(nn.Linear(self.n_nodes, 1)) #output layer #output size = 1 as it is a Regression problem
      self.model = nn.Sequential(*layers)  
      optimizer = self.optimization(self.model.parameters(), self.lr)
        
      #Iterate over the number of epochs 
      for epoch in range(self.nb_epoch):
        batch_loss = []
        for _, (batch_X, batch_Y) in enumerate(data_loader):
          optimizer.zero_grad()

          #Forward pass
          y_pred = self.model(batch_X)    
          
          # Loss update
          loss = self.loss_fn(y_pred, batch_Y)
          batch_loss.append(loss.detach().numpy().ravel())
          # Backward pass
          loss.backward()
          optimizer.step()
        self.losses.append(np.mean(batch_loss))
            
      return self


  def predict(self, x):
      """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

      X, _ = self._preprocessor(x, training = False) # Do not forget
      prediction = self.model(X)
      return prediction.detach().numpy()

  def score(self, x, y):
      """
      Function to evaluate the model accuracy on a validation dataset.

      Arguments:
          - x {pd.DataFrame} -- Raw input array of shape 
              (batch_size, input_size).
          - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

      Returns:
          {float} -- Quantification of the efficiency of the model.

      """
      X, Y = self._preprocessor(x, y = y, training = False) 
      Y_pred = self.model(X)
      return math.sqrt(self.loss_fn(Y_pred, Y))

  
def save_regressor(trained_model):
      """ 
      Utility function to save the trained regressor model in part2_model.pickle.
      """
      with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
      print("\nSaved model in part2_model.pickle\n")


def load_regressor():
      """ 
      Utility function to load the trained regressor model in part2_model.pickle.
      """
      with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
      print("\nLoaded model in part2_model.pickle\n")
      return trained_model

def RegressorHyperParameterSearch(regressor, X, Y, cv_outer = 5, cv_inner = 5):
  """ This function performs a grid search to tune the parameters of the the Regressor.

      Arguments:
      - regressor {Regressor} -- The regression model used to tune the parameters
      - X {pd.DataFrame} -- Raw input array of shape (batch_size, input_size).
      - Y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).
      - cv_outer {int} -- Number of CV folds.
      - cv_inner {int} -- Number of times CV needs to be repeated.

      Returns:
      - tuned_parameters {dict} -- Dictionnary that contains the optimal parameters for the Regressor estimator
  """

  
  param_dict_test = {
    "nb_epoch":[10,20,50,100],
    "n_hidden": [1,2,3,4],
    "n_nodes": [16,32,64,128,256],
    "batch_size": [64, 128, 256],
    "lr": [1e-4, 1e-3, 1e-2, 1e-1, 1e-0],
    "optimization": [torch.optim.Adam, torch.optim.SGD],
    } 

  
  cv = KFold(n_splits=cv_inner, shuffle= True)
  cv_outer = KFold(n_splits=cv_inner, shuffle= True)

  # Instantiate best model dictionary to store scores and outer CV fold counter 
  best_models = []
  best_params = []
  best_scores = []
  i = 1

  for train_index, test_index in cv_outer.split(X):
            
    # Split data 
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
            
    # Define parameter search space 
    param_search = GridSearchCV(regressor, param_dict_test, cv=cv, refit=True) # check whether we want
            
    # Execute search, obtain best model and score on heldout test fold
    search_results = param_search.fit(X_train, y_train)
    best_model = search_results.best_estimator_
    y_hat = best_model.predict(X_test)
    best_model_score = math.sqrt(mean_squared_error(y_hat, y_test))
    print(search_results.best_params_)
    
    # Save best model, best model parameters and best model score in best_models dictionary

    best_models.append(search_results.best_estimator_)
    best_params.append(search_results.best_params_)
    best_scores.append(best_model_score)
    print(f'Completed outer CV fold {i}')
    i += 1
  
  #Returns the parameters associated with the best score
  max_indx = np.argmin(best_scores)
  tuned_parameters = best_params[max_indx]
  print(type(tuned_parameters))
  return tuned_parameters

class RegressorWrapper(BaseEstimator):
    """This class wraps our custom model into a sklearn estimator, enabling us to perform a GridSearch on our custom estimator"""

    def __init__(self, x, loss_fn = torch.nn.MSELoss(), cv_outer = 5):
      self.x = x
      self.loss_fn = loss_fn
      self.cv_outer = cv_outer

    def fit(self, x, y):
      """
      Regressor training function that sets the parameters to the tuned ones, using the RegressorHyperParameterSearch function
      Arguments:
          - x {pd.DataFrame} -- Raw input array of shape 
              (batch_size, input_size).
          - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

      """
      self.regressor = Regressor(x, loss_fn=self.loss_fn)
      best_params = RegressorHyperParameterSearch(self.regressor,x,y,cv_outer=self.cv_outer)
      self.regressor.set_params(**best_params)
      self.regressor.fit(x, y)

    def predict(self, x):
      """
        Ouput the value corresponding to an input x, using the predict function of the custom Regressor class.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).
        """
      return self.regressor.predict(x)

    def score(self, x, y):
      """
      Function to evaluate the model accuracy on a validation dataset using the score function of the custom Regressor class.

      Arguments:
          - x {pd.DataFrame} -- Raw input array of shape 
              (batch_size, input_size).
          - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

      Returns:
          {float} -- Quantification of the efficiency of the model.

      """
      return self.regressor.score(x,y)

def example_main():

  output_label = "median_house_value"

  # Use pandas to read CSV data as it contains various object types
  # Feel free to use another CSV reader tool
  # But remember that LabTS tests take Pandas Dataframe as inputs
  data = pd.read_csv("housing.csv")

  #Spliting input and output
  x_train = data.loc[:, data.columns != output_label]
  y_train = data.loc[:, [output_label]]

  #Crossvalidation 
  cv = KFold(n_splits=10) #number of splits

  #Metrics to be displayed 
  metrics = {"R-squared": "r2","MAE": "neg_mean_absolute_error","MSE": "neg_mean_squared_error"}


  #Baseline model - Linear regression 
  # linear_regressor = Regressor(x_train, nb_epoch=50, n_hidden = 0)
  # scores_linear = cross_validate(linear_regressor, X=x_train, y=y_train, scoring=list(metrics.values()), cv=cv)
  # mean_MSE_linear = np.nanmean(-scores_linear['test_neg_mean_squared_error'])
  # mean_r2_linear = np.nanmean(-scores_linear['test_r2'])
  # mean_MAE_linear = np.nanmean(-scores_linear['test_neg_mean_absolute_error'])
  # print('MAE linear', mean_MAE_linear, 'MSE linear', mean_MSE_linear, 'r2 linear', mean_r2_linear)
  

  # #Scoring of our first model
  # first_model = Regressor(x_train)
  # scores_model = cross_validate(first_model, X=x_train, y=y_train, scoring=list(metrics.values()), cv=cv)
  # mean_MSE = np.nanmean(-scores_model['test_neg_mean_squared_error'])
  # mean_r2 = np.nanmean(-scores_model['test_r2'])
  # mean_MAE = np.nanmean(-scores_model['test_neg_mean_absolute_error'])
  # print('MAE', mean_MAE, 'MSE', mean_MSE, 'r2', mean_r2)
  

  #Scoring of the best Estimator
  opt_regressor = RegressorWrapper(x_train)
  scores_opt_model = cross_validate(opt_regressor, X=x_train, y=y_train, scoring=list(metrics.values()))
  mean_MSE_opt = np.nanmean(-scores_opt_model['test_neg_mean_squared_error'])
  mean_r2_opt = np.nanmean(-scores_opt_model['test_r2'])
  mean_MAE_opt = np.nanmean(-scores_opt_model['test_neg_mean_absolute_error']) 
  print('MAE', mean_MAE_opt, 'MSE', mean_MSE_opt, 'r2', mean_r2_opt)
  
  
  
  # #learning curves
  # linear_regressor = Regressor(x_train, nb_epoch=50, n_hidden = 0)
  # linear_regressor.fit(x_train, y_train)
  # loss_linear =  linear_regressor.losses
  # first_model = Regressor(x_train)
  # first_model.fit(x_train, y_train)
  # loss_first = first_model.losses
  # opt_model = Regressor(x_train, lr = 0.001, batch_size= 256, n_hidden=3, n_nodes=32)
  # opt_model.fit(x_train, y_train)
  # loss_opt = opt_model.losses
  
  # plt.plot(list(range(50)), loss_linear)
  # plt.plot(list(range(50)), loss_first)
  # plt.plot(list(range(50)), loss_opt)
  # plt.xlabel('Epochs')
  # plt.ylabel('Loss')
  # plt.title('Comparison of the learning curves')
  # plt.legend(['Linear Regression', 'First Regressor', 'Optimized Regressor'])
  # plt.show()
  
  
  save_regressor(opt_regressor.regressor)
  
if __name__ == "__main__":
  example_main()
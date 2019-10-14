#!/usr/bin/env python
# coding: utf-8

# # CS 7641 - Supervised Learning
# ## Manish Mehta

# This file provides the code base for each of the 5 classification algorithms for each of the two datasets. All the analysis in the report are based on this (Running this Notebook generates the same outputs as referred to in the report)
# 
# Datasets: Phishing Websites, Bank Marketing.
# 
# Classification Algorithms: Decision Tree, Neural Network, Boosting, Support Vector Machines, k-Nearest Neighbors.
# 
# The learning curve is plotted, along with the plot for model complexity, after hyperparameter tuning has been performed.

# # Data Loading and Preprocessing

# Please save the datasets to your local machine and change the current directory to a file where you have the data stored (for the purpose of this assignment, we can assume that the location of the notebook is same as the location of the 2 CSV files containing the Bank Marketing and Phishing Websites data respectively.

# In[ ]:


import os
import pandas as pd
import numpy as np
import random
import gc

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


# ### Bank Marketing Data

# In[ ]:


## Load the Bank Marketing Data
df_bank_orig = pd.read_csv('BankMarketingData.csv')
print("Data has",len(df_bank_orig),"rows and", len(list(df_bank_orig)),"columns.")
if df_bank_orig.isnull().values.any():
    print("Warning: Missing Data")


# In[ ]:


# Make a copy the original dataset for further processing
df_bank = df_bank_orig.copy()


# Since this dataset has columns with various non-numeric levels (eg. 'job' has levels like 'student', 'admin', 'technician', etc.), we will have convert all those categorical columns using one-hot encoding. Standardization of all of the numeric features is also required. We will rewrite the target variables from {no,yes} to {0,1}.
# 
# Also, the feature 'pdays' is numeric but contains values that are '999' representing if the customer was not called before. For the sake of better interpretability, we can create a new feature that defines whether or not a customer had been called before denoted by {0,1} respectively. Otherwise '999' may be outlier during standardization of the columns.

# In[ ]:


# Banking Data
col_1hot = df_bank.drop(df_bank.describe().columns.tolist(), axis=1).columns.tolist()[:-1]
df_1hot = df_bank[col_1hot]
df_1hot = pd.get_dummies(df_1hot).astype('category')
df_others = df_bank.drop(col_1hot,axis=1)
df_bank = pd.concat([df_others,df_1hot],axis=1)
column_order = list(df_bank)
column_order = column_order[:-1] + column_order[-1:]

df_bank = df_bank.loc[:, column_order]
df_bank['y'].replace("no",0,inplace=True)
df_bank['y'].replace("yes",1,inplace=True)
df_bank['y'] = df_bank['y'].astype('category')

numericcols = df_bank.describe().columns.tolist()
df_num = df_bank[numericcols]
df_stand =(df_num-df_num.min())/(df_num.max()-df_num.min())
df_bank_categorical = df_bank.drop(numericcols,axis=1)
df_bank = pd.concat([df_bank_categorical,df_stand],axis=1)
df_bank.describe(include='all')


# In[ ]:


# Garbaage collection
gc.collect()


# ### Phishing Website Data

# In[ ]:


## Loading Phishing Websites data

df_phish_orig = pd.read_csv('PhishingWebsitesData.csv').astype('category')
print("Data has",len(df_phish_orig),"rows and", len(list(df_phish_orig)),"columns.")
if df_phish_orig.isnull().values.any():
    print("Warning: Missing Data")


# In[ ]:


# Copy the original data into another dataframe for further use
df_phish = df_phish_orig.copy()


# Again, we need to do some preprocessing. A lot of columns are categorical with the levels {-1,0,1} and the rest are all binary with levels {-1,1}. We can rewrite the {-1,1} columns to {0,1} and for the 3-level columns, we will use one-hot encoding to create additional features which have level {0,1}. This leads to an increase in the number of features, and all are binary features.

# In[ ]:


# Identifying the list of columns which have three categories so that they can be converted to multiple columns with one-hot encoding
temp1 = pd.DataFrame(df_phish.describe())
idx_uniq = list(temp1.index).index('unique')
col_1hot = temp1.columns[temp1.iloc[idx_uniq] == 3].tolist()
df_1hot = df_phish[col_1hot]

# Convert categorical to dummy variables, same as one-hot encoding
df_1hot = pd.get_dummies(df_1hot)
# Non-categorical variables/columns separated
df_others = df_phish.drop(col_1hot,axis=1)


df_phish = pd.concat([df_1hot,df_others],axis=1)
df_phish = df_phish.replace(-1,0).astype('category')
column_order = df_phish.columns.tolist()
# Getting response variable to be the first in the dataframe
column_order = column_order[-1:] + column_order[:-1]  

df_phish = df_phish.loc[:, column_order]
df_phish.describe(include='all')


# We now have a file with no missing data in the format [y, X] where all features are binary {0,1}. The phishing data is ready to go! Now we move on to loading the Bank Marketing data.

# In[ ]:


gc.collect()


# # Useful Modules

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import itertools
import timeit

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['font.size'] = 12

def import_data():

    X1 = np.array(df_phish.values[:,1:],dtype='int64')
    Y1 = np.array(df_phish.values[:,0],dtype='int64')
    X2 = np.array(df_bank.values[:,1:],dtype='int64')
    Y2 = np.array(df_bank.values[:,0],dtype='int64')

    return X1, Y1, X2, Y2

# Plot the learning curve for different training sizes
def plot_LC(train_sizes, train_mean, train_std, cv_mean, cv_std, title):
    
    plt.figure()
    plt.title("Learning Curve: "+ title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model F1 Score")
    plt.fill_between(train_sizes, train_mean - 2*train_std, train_mean + 2*train_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, cv_mean - 2*cv_std, cv_mean + 2*cv_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training Score")
    plt.plot(train_sizes, cv_mean, 'o-', color="r", label="Cross-Validation Score")
    plt.legend(loc="best")
    plt.show()
    
    
def plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title):
    
    plt.figure()
    plt.title("Modeling Time: "+ title)
    plt.xlabel("Training Examples")
    plt.ylabel("Training Time (s)")
    plt.fill_between(train_sizes, fit_mean - 2*fit_std, fit_mean + 2*fit_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, pred_mean - 2*pred_std, pred_mean + 2*pred_std, alpha=0.1, color="r")
    plt.plot(train_sizes, fit_mean, 'o-', color="b", label="Training Time (s)")
    plt.plot(train_sizes, pred_std, 'o-', color="r", label="Prediction Time (s)")
    plt.legend(loc="best")
    plt.show()
    

def plot_learning_curve(clf, X, y, title="Insert Title"):
    
    n = len(y)
    train_mean = []; train_std = [] #model performance score (f1)
    cv_mean = []; cv_std = [] #model performance score (f1)
    fit_mean = []; fit_std = [] #model fit/training time
    pred_mean = []; pred_std = [] #model test/prediction times
    train_sizes=(np.linspace(.05, 1.0, 20)*n).astype('int')  
    
    for i in train_sizes:
        idx = np.random.randint(X.shape[0], size=i)
        X_subset = X[idx,:]
        y_subset = y[idx]
        scores = cross_validate(clf, X_subset, y_subset, cv=10, scoring='f1', n_jobs=-1, return_train_score=True)
        
        # Find the mean (of all metrics) for all cv splits, for each size of subset
        train_mean.append(np.mean(scores['train_score'])); train_std.append(np.std(scores['train_score']))
        cv_mean.append(np.mean(scores['test_score'])); cv_std.append(np.std(scores['test_score']))
        fit_mean.append(np.mean(scores['fit_time'])); fit_std.append(np.std(scores['fit_time']))
        pred_mean.append(np.mean(scores['score_time'])); pred_std.append(np.std(scores['score_time']))
    
    # Calculate the mean value for all the metrics across all subset sizes
    train_mean = np.array(train_mean); train_std = np.array(train_std)
    cv_mean = np.array(cv_mean); cv_std = np.array(cv_std)
    fit_mean = np.array(fit_mean); fit_std = np.array(fit_std)
    pred_mean = np.array(pred_mean); pred_std = np.array(pred_std)
    
    plot_LC(train_sizes, train_mean, train_std, cv_mean, cv_std, title)
    plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title)
    
    return train_sizes, train_mean, cv_mean, fit_mean, pred_mean
    

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    
def final_classifier_evaluation(clf,X_train, X_test, y_train, y_test):
    
    start_time = timeit.default_timer()
    clf.fit(X_train, y_train)
    end_time = timeit.default_timer()
    training_time = end_time - start_time
    
    start_time = timeit.default_timer()    
    y_pred = clf.predict(X_test)
    end_time = timeit.default_timer()
    pred_time = end_time - start_time
    
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)

    print("Model Evaluation Metrics Using Test Dataset")
    print("*****************************************************")
    print("Model Train Time (s):   "+"{:.5f}".format(training_time))
    print("Model Prediction Time (s): "+"{:.5f}\n".format(pred_time))
    print("F1 Score:  "+"{:.2f}".format(f1))
    print("Accuracy:  "+"{:.2f}".format(accuracy)+"     AUC:       "+"{:.2f}".format(auc))
    print("Precision: "+"{:.2f}".format(precision)+"     Recall:    "+"{:.2f}".format(recall))
    print("*****************************************************")
    plt.figure()
    plot_confusion_matrix(cm, classes=["0","1"], title='Confusion Matrix')
    plt.show()


# # Supervised ML Algorithms

# ## Neural Network

# In[ ]:


from sklearn.neural_network import MLPClassifier
    
def NNGridSearchCV(X_train, y_train, h_units, learning_rates):
    #parameters to search:
    #number of hidden units
    #learning_rate
    param_grid = {'hidden_layer_sizes': h_units, 'learning_rate_init': learning_rates}

    nnet = GridSearchCV(estimator = MLPClassifier(solver='adam',activation='logistic',random_state=100),
                       param_grid=param_grid, cv=10, n_jobs=-1)
    nnet.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(nnet.best_params_)
    return nnet.best_params_['hidden_layer_sizes'], nnet.best_params_['learning_rate_init']


# In[ ]:


from sklearn.model_selection import validation_curve

phishX,phishY, bankX, bankY = import_data()
X_train, X_test, y_train, y_test = train_test_split(np.array(phishX),np.array(phishY), test_size=0.20, random_state=100)


# In[ ]:


gc.collect()


# In[ ]:


# Tuning for just number of neurons in 1 layer (with only 1 layer possible)
param_range = np.linspace(1, 250, 25).astype('int')
train_scores, test_scores = validation_curve(
    MLPClassifier(solver='adam', activation='logistic', learning_rate_init=0.05, random_state=100), 
    X_train, y_train, param_name='hidden_layer_sizes', param_range=param_range, 
    cv=10, scoring='f1', n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)


# In[ ]:


plt.plot(param_range, train_scores_mean, 'o-', color='b', label='Train F1 Score')
plt.plot(param_range, test_scores_mean, 'o-', color = 'r', label='Test F1 Score')
plt.ylim(0.9, 1.0)
plt.ylabel('Model F1 Score')
plt.xlabel('No. of neurons in 1 hidden layer structure')

plt.title("Model Complexity Curve for NN (Phishing Data)\nHyperparameter : No. of neurons in 1 hidden layer structure")
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# In[ ]:


# Max validation accuracy obtained at
param_range[np.argmax(test_scores_mean)]


# In[ ]:


# Same thing as above but using actual test set, not a good way - NOT SURE TO INCLUDE OR NOT?
# hyperNN(X_train, y_train, X_test, y_test, "Model Complexity Curve for NN (Phishing Data)\nHyperparameter : No. of neurons in 1 hidden layer structure")


# In[ ]:


gc.collect()


# In[ ]:


# Tuning for just number of neurons in multiple layers and neurons
param_range = [(100,100,100), (100,50,25), (100,50), (50,100), (50,50), (100), (50), (25)]
train_scores, test_scores = validation_curve(
    MLPClassifier(solver='adam', activation='logistic', learning_rate_init=0.05, random_state=100), 
    X_train, y_train, param_name='hidden_layer_sizes', param_range=param_range, 
    cv=10, scoring='f1', n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)


# In[ ]:


plt.plot(range(1, len(param_range) + 1), train_scores_mean, 'o-', color='b', label='Train F1 Score')
plt.plot(range(1, len(param_range) + 1), test_scores_mean, 'o-', color = 'r', label='Test F1 Score')
plt.ylim(0.9, 1.0)
plt.ylabel('Model F1 Score')
plt.xlabel('Model Complexity (Decreasing order)')

plt.title("Model Complexity Curve for NN (Phishing Data)\nHyperparameter : Different No. of Layer and Neuron Combinations")
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# In[ ]:


# Max validation accuracy obtained at
param_range[np.argmax(test_scores_mean)]


# In[ ]:


gc.collect()


# In[ ]:


# Tuning for learning rate
param_range = np.logspace(-3, -1, 21)
train_scores, test_scores = validation_curve(
    MLPClassifier(solver='adam', activation='logistic', hidden_layer_sizes = (100), random_state=100), 
    X_train, y_train, param_name='learning_rate_init', param_range=param_range, 
    cv=10, scoring='f1', n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)


# In[ ]:


plt.semilogx(param_range, train_scores_mean, 'o-', color='b', label='Train F1 Score')
plt.semilogx(param_range, test_scores_mean, 'o-', color = 'r', label='Test F1 Score')
plt.ylim(0.9, 1.0)
plt.ylabel('Model F1 Score')
plt.xlabel('Learning Rate')

plt.title("Model Complexity Curve for NN (Phishing Data)\nHyperparameter : Learning Rate")
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# In[ ]:


# Max obtained at
param_range[np.argmax(test_scores_mean)]


# In[ ]:


gc.collect()


# In[ ]:


param_range


# In[ ]:


# Running the final grid search
h_units = [(50,50), (100), (50), (25)]
learn_rate = [0.01, 0.015, 0.02, 0.025]

h_units, learn_rate = NNGridSearchCV(X_train, y_train, h_units, learn_rate)
estimator_phish = MLPClassifier(hidden_layer_sizes=(h_units,), solver='adam', activation='logistic', 
                               learning_rate_init=learn_rate, random_state=100)
train_samp_phish, NN_train_score_phish, NN_cv_score_phish, NN_fit_time_phish, NN_pred_time_phish = plot_learning_curve(estimator_phish, X_train, y_train,title="Neural Net Phishing Data")
print("Training Score: ", NN_train_score_phish[1])
print("CV Score: ", NN_cv_score_phish[-1])

final_classifier_evaluation(estimator_phish, X_train, X_test, y_train, y_test)
estimator_phish.fit(X_train, y_train)
loss_phish = estimator_phish.loss_curve_


# In[ ]:


gc.collect()


# Banking Data Run

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(np.array(bankX),np.array(bankY), test_size=0.20, random_state=100)


# In[ ]:


gc.collect()


# In[ ]:


# Tuning for just number of neurons in 1 layer (with only 1 layer possible)
param_range = np.linspace(1, 300, 15).astype('int')
train_scores, test_scores = validation_curve(
    MLPClassifier(solver='adam', activation='logistic', learning_rate_init=0.05, random_state=100), 
    X_train, y_train, param_name='hidden_layer_sizes', param_range=param_range, 
    cv=10, scoring='f1', n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)


# In[ ]:


train_scores_mean.min()


# In[ ]:


plt.plot(param_range, train_scores_mean, 'o-', color='b', label='Train F1 Score')
plt.plot(param_range, test_scores_mean, 'o-', color = 'r', label='Test F1 Score')
plt.ylim(0, 0.6)
plt.ylabel('Model F1 Score')
plt.xlabel('No. of neurons in 1 hidden layer structure')

plt.title("Model Complexity Curve for NN (Banking Data)\nHyperparameter : No. of neurons in 1 hidden layer structure")
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# In[ ]:


# Max validation accuracy obtained at
param_range[np.argmax(test_scores_mean)]


# In[ ]:


gc.collect()


# In[ ]:


# Tuning for just number of neurons in multiple layers and neurons
param_range = [(150,150), (170,150), (50,50), (50,25), (150), (100), (50), (25)]
train_scores, test_scores = validation_curve(
    MLPClassifier(solver='adam', activation='logistic', learning_rate_init=0.05, random_state=100), 
    X_train, y_train, param_name='hidden_layer_sizes', param_range=param_range, 
    cv=10, scoring='f1', n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)


# In[ ]:


plt.plot(range(1, len(param_range) + 1), train_scores_mean, 'o-', color='b', label='Train F1 Score')
plt.plot(range(1, len(param_range) + 1), test_scores_mean, 'o-', color = 'r', label='Test F1 Score')
plt.ylim(0, 0.8)
plt.ylabel('Model F1 Score')
plt.xlabel('Model Complexity (Decreasing order)')

plt.title("Model Complexity Curve for NN (Banking Data)\nHyperparameter : Different No. of Layer and Neuron Combinations")
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# In[ ]:


# Max validation accuracy obtained at
param_range[np.argmax(test_scores_mean)]


# In[ ]:


gc.collect()


# In[ ]:


# Tuning for learning rate
param_range = np.logspace(-3, -1, 21)
train_scores, test_scores = validation_curve(
    MLPClassifier(solver='adam', activation='logistic', hidden_layer_sizes = (50,25), random_state=100), 
    X_train, y_train, param_name='learning_rate_init', param_range=param_range, 
    cv=10, scoring='f1', n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)


# In[ ]:


plt.semilogx(param_range, train_scores_mean, 'o-', color='b', label='Train F1 Score')
plt.semilogx(param_range, test_scores_mean, 'o-', color = 'r', label='Test F1 Score')
plt.ylim(0, 1.0)
plt.ylabel('Model F1 Score')
plt.xlabel('Learning Rate')

plt.title("Model Complexity Curve for NN (Banking Data)\nHyperparameter : Learning Rate")
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# In[ ]:


# Max obtained at
param_range[np.argmax(test_scores_mean)]


# In[ ]:


gc.collect()


# In[ ]:


# Running the final grid search
h_units = [(50,25), (50), (10)]
learn_rate = [0.0158, 0.02, 0.025, 0.05]

h_units, learn_rate = NNGridSearchCV(X_train, y_train, h_units, learn_rate)
estimator_bank = MLPClassifier(hidden_layer_sizes=(h_units,), solver='adam', activation='logistic', 
                               learning_rate_init=learn_rate, random_state=100)
train_samp_bank, NN_train_score_bank, NN_cv_score_bank, NN_fit_time_bank, NN_pred_time_bank = plot_learning_curve(estimator_bank, X_train, y_train,title="Neural Net Banking Data")
print("Training Score: ", NN_train_score_bank[-1])
print("CV Score: ", NN_cv_score_bank[-1])

final_classifier_evaluation(estimator_bank, X_train, X_test, y_train, y_test)
estimator_bank.fit(X_train, y_train)
loss_bank = estimator_bank.loss_curve_


# In[ ]:


gc.collect()


# The final section for neural network will plot the loss curve for each dataset over the iterations.

# In[ ]:


# Loss Curve
plt.figure()
plt.title("Loss Curve")
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
plt.plot(loss_phish, 'o-', color="b", label="Phishing Data")
plt.plot(loss_bank, 'o-', color="r", label="Banking Data")
plt.legend(loc="best")
plt.show()


# In[ ]:


gc.collect()


# ## Support Vector Machine

# In[ ]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, normalize

def hyperSVM(X_train, y_train, X_test, y_test, title):

    f1_test = []
    f1_train = []
    kernel_func = ['linear','poly','sigmoid', 'rbf']
    for i in kernel_func:         
            if i == 'poly':
                for j in [2,3,4,5,6,7,8]:
                    clf = SVC(kernel=i, degree=j,random_state=100, gamma='auto')
                    clf.fit(X_train, y_train)
                    y_pred_test = clf.predict(X_test)
                    y_pred_train = clf.predict(X_train)
                    f1_test.append(f1_score(y_test, y_pred_test))
                    f1_train.append(f1_score(y_train, y_pred_train))
            else:    
                clf = SVC(kernel=i, random_state=100, gamma='auto')
                clf.fit(X_train, y_train)
                y_pred_test = clf.predict(X_test)
                y_pred_train = clf.predict(X_train)
                f1_test.append(f1_score(y_test, y_pred_test))
                f1_train.append(f1_score(y_train, y_pred_train))
                
    xvals = ['linear','poly2','poly3','poly4','poly5','poly6','poly7','poly8','sigmoid','rbf']
    plt.plot(xvals, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(xvals, f1_train, 'o-', color = 'b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('Kernel Function')
    
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
def SVMGridSearchCV(X_train, y_train, kernels='rbf'):
    #parameters to search:
    #penalty parameter, C
    #
    Cs = [0.01, 0.1, 1, 10]
    gammas = [0.1, 1, 10]
    if kernels != 'rbf':
        gammas = [1]
        
    param_grid = {'C': Cs, 'gamma': gammas, 'kernel': kernels}

    clf = GridSearchCV(estimator = SVC(random_state=100),
                       param_grid=param_grid, cv=10, n_jobs=4)
    clf.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(clf.best_params_)
    return clf.best_params_['C'], clf.best_params_['gamma'], clf.best_params_['kernel']


# In[ ]:


gc.collect()


# In[ ]:


phishX,phishY, bankX, bankY = import_data()
X_train, X_test, y_train, y_test = train_test_split(np.array(phishX),np.array(phishY), test_size=0.20, random_state=100)

hyperSVM(X_train, y_train, X_test, y_test,title="Model Complexity Curve for SVM (Phishing Data)\nHyperparameter : Kernel Function")
C_val, gamma_val, kernel_val = SVMGridSearchCV(X_train, y_train)
estimator_phish = SVC(C=C_val, gamma=gamma_val, kernel=kernel_val, random_state=100)
train_samp_phish, SVM_train_score_phish, SVM_cv_score_phish, SVM_fit_time_phish, SVM_pred_time_phish = plot_learning_curve(estimator_phish, X_train, y_train,title="SVM Phishing Data")
print("Training Score: ", SVM_train_score_phish[-1])
print("CV Score: ", SVM_cv_score_phish[-1])

final_classifier_evaluation(estimator_phish, X_train, X_test, y_train, y_test)


# In[ ]:


gc.collect()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(np.array(bankX),np.array(bankY), test_size=0.20, random_state=100)

hyperSVM(X_train, y_train, X_test, y_test,title="Model Complexity Curve for SVM (Banking Data)\nHyperparameter : Kernel Function")
C_val, gamma_val, kernel_val = SVMGridSearchCV(X_train, y_train, ['rbf','linear'])

# estimator_bank = SVC(C=C_val, gamma=gamma_val, kernel=kernel_val, random_state=100)
estimator_bank = SVC(C=1, kernel='rbf', random_state=100)
train_samp_bank, SVM_train_score_bank, SVM_cv_score_bank, SVM_fit_time_bank, SVM_pred_time_bank = plot_learning_curve(estimator_bank, X_train, y_train,title="SVM Banking Data")
print("Training Score: ", SVM_train_score_bank[-1])
print("CV Score: ", SVM_cv_score_bank[-1])

final_classifier_evaluation(estimator_bank, X_train, X_test, y_train, y_test)


# In[ ]:


gc.collect()


# # KNN Classifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier as kNN

def hyperKNN(X_train, y_train, X_test, y_test, title):
    
    f1_test = []
    f1_train = []
    klist = np.linspace(1,250,25).astype('int')
    for i in klist:
        clf = kNN(n_neighbors=i,n_jobs=-1)
        clf.fit(X_train,y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))
        
    plt.plot(klist, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(klist, f1_train, 'o-', color = 'b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('No. Neighbors')
    
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    


# In[ ]:


gc.collect()


# In[ ]:


phishX,phishY,bankX,bankY = import_data()

X_train, X_test, y_train, y_test = train_test_split(np.array(phishX),np.array(phishY), test_size=0.20, random_state=100)
hyperKNN(X_train, y_train, X_test, y_test,title="Model Complexity Curve for kNN (Phishing Data)\nHyperparameter : No. Neighbors")
estimator_phish = kNN(n_neighbors=40, n_jobs=-1)
train_samp_phish, kNN_train_score_phish, kNN_cv_score_phish, kNN_fit_time_phish, kNN_pred_time_phish = plot_learning_curve(estimator_phish, X_train, y_train,title="kNN Phishing Data")
print("Training Score: ", kNN_train_score_phish[-1])
print("CV Score: ", kNN_cv_score_phish[-1])

final_classifier_evaluation(estimator_phish, X_train, X_test, y_train, y_test)


# In[ ]:


gc.collect()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(np.array(bankX),np.array(bankY), test_size=0.20, random_state=100)

hyperKNN(X_train, y_train, X_test, y_test,title="Model Complexity Curve for kNN (Banking Data)\nHyperparameter : No. Neighbors")
estimator_bank = kNN(n_neighbors=10, n_jobs=-1)
train_samp_bank, kNN_train_score_bank, kNN_cv_score_bank, kNN_fit_time_bank, kNN_pred_time_bank = plot_learning_curve(estimator_bank, X_train, y_train,title="kNN Banking Data")
print("Training Score: ", kNN_train_score_bank[-1])
print("CV Score: ", kNN_cv_score_bank[-1])

final_classifier_evaluation(estimator_bank, X_train, X_test, y_train, y_test)


# In[ ]:


gc.collect()


# # Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

def hyperTree(X_train, y_train, X_test, y_test, title):
    
    f1_test = []
    f1_train = []
    max_depth = list(range(1,31))
    for i in max_depth:         
            clf = DecisionTreeClassifier(max_depth=i, random_state=100, min_samples_leaf=1, criterion='entropy')
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            f1_test.append(f1_score(y_test, y_pred_test))
            f1_train.append(f1_score(y_train, y_pred_train))
      
    plt.plot(max_depth, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(max_depth, f1_train, 'o-', color = 'b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('Max Tree Depth')
    
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
     
    
def TreeGridSearchCV(start_leaf_n, end_leaf_n, X_train, y_train):
    param_grid = {'min_samples_leaf':np.linspace(start_leaf_n,end_leaf_n,20).round().astype('int'), 'max_depth':np.arange(1,20)}

    tree = GridSearchCV(DecisionTreeClassifier(random_state=100), param_grid=param_grid, cv=10, n_jobs=-1)
    tree.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(tree.best_params_)
    return tree.best_params_['max_depth'], tree.best_params_['min_samples_leaf']


# In[ ]:


gc.collect()


# In[ ]:


phishX,phishY,bankX,bankY = import_data()

X_train, X_test, y_train, y_test = train_test_split(np.array(phishX),np.array(phishY), test_size=0.20, random_state=100)

hyperTree(X_train, y_train, X_test, y_test,title="Model Complexity Curve for Decision Tree (Phishing Data)\nHyperparameter : Tree Max Depth")
start_leaf_n = round(0.005*len(X_train))
# Leaf nodes of size [0.5%, 5% will be tested]
end_leaf_n = round(0.05*len(X_train)) 
max_depth, min_samples_leaf = TreeGridSearchCV(start_leaf_n,end_leaf_n,X_train,y_train)
estimator_phish = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=100, criterion='entropy')
train_samp_phish, DT_train_score_phish, DT_cv_score_phish, DT_fit_time_phish, DT_pred_time_phish = plot_learning_curve(estimator_phish, X_train, y_train,title="Decision Tree Phishing Data")
print("Training Score: ", DT_train_score_phish[-1])
print("CV Score: ", DT_cv_score_phish[-1])

final_classifier_evaluation(estimator_phish, X_train, X_test, y_train, y_test)


# In[ ]:


gc.collect()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(np.array(bankX),np.array(bankY), test_size=0.20, random_state=100)

hyperTree(X_train, y_train, X_test, y_test,title="Model Complexity Curve for Decision Tree (Banking Data)\nHyperparameter : Tree Max Depth")
start_leaf_n = round(0.005*len(X_train))
end_leaf_n = round(0.05*len(X_train)) #leaf nodes of size [0.5%, 5% will be tested]
max_depth, min_samples_leaf = TreeGridSearchCV(start_leaf_n,end_leaf_n,X_train,y_train)
estimator_bank = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=100, criterion='entropy')
train_samp_bank, DT_train_score_bank, DT_cv_score_bank, DT_fit_time_bank, DT_pred_time_bank = plot_learning_curve(estimator_bank, X_train, y_train,title="Decision Tree Banking Data")
print("Training Score: ", DT_train_score_bank[-1])
print("CV Score: ", DT_cv_score_bank[-1])

final_classifier_evaluation(estimator_bank, X_train, X_test, y_train, y_test)


# In[ ]:


gc.collect()


# # Boosting

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

def hyperBoost(X_train, y_train, X_test, y_test, max_depth, min_samples_leaf, title):
    
    f1_test = []
    f1_train = []
    n_estimators = np.linspace(1,250,40).astype('int')
    for i in n_estimators:         
            clf = GradientBoostingClassifier(n_estimators=i, max_depth=int(max_depth/2), 
                                             min_samples_leaf=int(min_samples_leaf/2), random_state=100,)
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            f1_test.append(f1_score(y_test, y_pred_test))
            f1_train.append(f1_score(y_train, y_pred_train))
      
    plt.plot(n_estimators, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(n_estimators, f1_train, 'o-', color = 'b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('No. Estimators')
    
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def BoostedGridSearchCV(start_leaf_n, end_leaf_n, X_train, y_train):
    param_grid = {'min_samples_leaf': np.linspace(start_leaf_n,end_leaf_n,3).round().astype('int'),
                  'max_depth': np.arange(1,4),
                  'n_estimators': np.linspace(10,100,3).round().astype('int'),
                  'learning_rate': np.linspace(.001,.1,3)}

    boost = GridSearchCV(GradientBoostingClassifier(), param_grid=param_grid, cv=10, n_jobs=-1)
    boost.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(boost.best_params_)
    return boost.best_params_['max_depth'], boost.best_params_['min_samples_leaf'], boost.best_params_['n_estimators'], boost.best_params_['learning_rate']


# In[ ]:


gc.collect()


# In[ ]:


phishX,phishY,bankX,bankY = import_data()

X_train, X_test, y_train, y_test = train_test_split(np.array(phishX),np.array(phishY), test_size=0.20, random_state=100)

hyperBoost(X_train, y_train, X_test, y_test, 3, 50, title="Model Complexity Curve for Boosted Tree (Phishing Data)\nHyperparameter : No. Estimators")
start_leaf_n = round(0.005*len(X_train))
end_leaf_n = round(0.05*len(X_train))
max_depth, min_samples_leaf, n_est, learn_rate = BoostedGridSearchCV(start_leaf_n,end_leaf_n,X_train,y_train)
estimator_phish = GradientBoostingClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, 
                                              n_estimators=n_est, learning_rate=learn_rate, random_state=100)
train_samp_phish, BT_train_score_phish, DT_cv_score_phish, BT_fit_time_phish, BT_pred_time_phish = plot_learning_curve(estimator_phish, X_train, y_train,title="Boosted Tree Phishing Data")
print("Training Score: ", BT_train_score_phish)
print("CV Score: ", BT_cv_score_phish)

final_classifier_evaluation(estimator_phish, X_train, X_test, y_train, y_test)


# In[ ]:


gc.collect()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(np.array(bankX),np.array(bankY), test_size=0.20, random_state=100)

hyperBoost(X_train, y_train, X_test, y_test, 3, 50, title="Model Complexity Curve for Boosted Tree (Banking Data)\nHyperparameter : No. Estimators")
start_leaf_n = round(0.005*len(X_train))
end_leaf_n = round(0.05*len(X_train))
max_depth, min_samples_leaf, n_est, learn_rate = BoostedGridSearchCV(start_leaf_n,end_leaf_n,X_train,y_train)
estimator_bank = GradientBoostingClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, 
                                              n_estimators=n_est, learning_rate=learn_rate, random_state=100)
train_samp_bank, BT_train_score_bank, BT_cv_score_bank, BT_fit_time_bank, BT_pred_time_bank = plot_learning_curve(estimator_bank, X_train, y_train,title="Boosted Tree Banking Data")
print("Training Score: ", BT_train_score_bank)
print("CV Score: ", BT_cv_score_bank)

final_classifier_evaluation(estimator_bank, X_train, X_test, y_train, y_test)


# In[ ]:


gc.collect()


# # Model Comparison Plots

# In[ ]:


def compare_fit_time(n,NNtime, SMVtime, kNNtime, DTtime, BTtime, title):
    
    plt.figure()
    plt.title("Model Training Times: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model Training Time (s)")
    plt.plot(n, NNtime, '-', color="b", label="Neural Network")
    plt.plot(n, SMVtime, '-', color="r", label="SVM")
    plt.plot(n, kNNtime, '-', color="g", label="kNN")
    plt.plot(n, DTtime, '-', color="m", label="Decision Tree")
    plt.plot(n, BTtime, '-', color="k", label="Boosted Tree")
    plt.legend(loc="best")
    plt.show()
    
def compare_pred_time(n,NNpred, SMVpred, kNNpred, DTpred, BTpred, title):
    
    plt.figure()
    plt.title("Model Prediction Times: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model Prediction Time (s)")
    plt.plot(n, NNpred, '-', color="b", label="Neural Network")
    plt.plot(n, SMVpred, '-', color="r", label="SVM")
    plt.plot(n, kNNpred, '-', color="g", label="kNN")
    plt.plot(n, DTpred, '-', color="m", label="Decision Tree")
    plt.plot(n, BTpred, '-', color="k", label="Boosted Tree")
    plt.legend(loc="best")
    plt.show()


def compare_learn_time(n,NNlearn, SMVlearn, kNNlearn, DTlearn, BTlearn, title):
    
    plt.figure()
    plt.title("Model Learning Rates: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model F1 Score")
    plt.plot(n, NNlearn, '-', color="b", label="Neural Network")
    plt.plot(n, SMVlearn, '-', color="r", label="SVM")
    plt.plot(n, kNNlearn, '-', color="g", label="kNN")
    plt.plot(n, DTlearn, '-', color="m", label="Decision Tree")
    plt.plot(n, BTlearn, '-', color="k", label="Boosted Tree")
    plt.legend(loc="best")
    plt.show() 


# In[ ]:


compare_fit_time(train_samp_phish, NN_fit_time_phish, SVM_fit_time_phish, kNN_fit_time_phish, 
                 DT_fit_time_phish, BT_fit_time_phish, 'Phishing Dataset')              
compare_pred_time(train_samp_phish, NN_pred_time_phish, SVM_pred_time_phish, kNN_pred_time_phish, 
                 DT_pred_time_phish, BT_pred_time_phish, 'Phishing Dataset')   
compare_learn_time(train_samp_phish, NN_train_score_phish, SVM_train_score_phish, kNN_train_score_phish, 
                 DT_train_score_phish, BT_train_score_phish, 'Phishing Dataset')  



compare_fit_time(train_samp_bank, NN_fit_time_bank, SVM_fit_time_bank, kNN_fit_time_bank, 
                 DT_fit_time_bank, BT_fit_time_bank, 'Banking Dataset')       
compare_pred_time(train_samp_bank, NN_pred_time_bank, SVM_pred_time_bank, kNN_pred_time_bank, 
                 DT_pred_time_bank, BT_pred_time_bank, 'Banking Dataset')           
compare_learn_time(train_samp_bank, NN_train_score_bank, SVM_train_score_bank, kNN_train_score_bank, 
                 DT_train_score_bank, BT_train_score_bank, 'Banking Dataset')


# In[ ]:


gc.collect()


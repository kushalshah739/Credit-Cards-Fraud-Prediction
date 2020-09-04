#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #MATRIX COMPUTATION
import numpy as np #LINEAR ALGEBRA COMPUTATION
import seaborn as sns # FOR PLOTTING


# In[2]:


dataset = pd.read_csv("C:/Users/apurva/Desktop/creditcard.csv")


# In[67]:


dataset.info()


# In[3]:


# display the first ten rows of the dataset.
dataset.head(10)


# In[4]:


# find the number of rows and coloumns of the dataset
dataset.shape


# In[5]:


dataset.isnull().sum()
# number of empty/null values


# In[6]:


dataset.columns
# lists all coloumns


# In[7]:


dataset['Fraudulent'].value_counts()


# In[8]:


x = dataset.drop('Fraudulent',axis=1)
x
y = dataset.Fraudulent


# In[9]:


# Before we make a machine learning model, we need to check the dataframe for Fradulent/Non-Fradulent if it is imbalanced.
# Else, the model will simply 'memorise' the pattern and allocate most transactions to be 'N' because of the majority.

import matplotlib.pyplot as plt 

sns.countplot(x='Fraudulent',data=dataset)


# In[10]:


#clearly there is a huge imbalance for fraudulent and non-fraudulent values.
# In the next two cells, implement the undersampling library using the NearMiss() function that will undersample the exceessive Non-Fradulent (0)

from imblearn.under_sampling import NearMiss
under_sampler = NearMiss()
x_res,y_res = under_sampler.fit_sample(x,y)


# In[1]:


from collections import Counter
print("before oversampling:",Counter(y))
print("after oversampling:",Counter(y_res))


# In[12]:


y_res.value_counts()
# now it is balanced


# In[13]:


x_res= pd.concat([x_res,y_res],axis=1)
x_res = x_res.loc[:,~x_res.columns.duplicated()]
x_res


# In[14]:


sns.countplot(x='Fraudulent',data= x_res)


# In[15]:


# We now split the dataset into 90-10 where 90% of the dataset will be used for building our model and 10% to test on.
data = x_res.sample(frac=0.90, random_state=786)
data_unseen = x_res.drop(data.index)


#data.reset_index(inplace=True, drop=True)
#data_unseen.reset_index(inplace=True, drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


# In[16]:


# the 90% of our dataset above will further be split into 70-30 (default ratio) as a training and testing set
from pycaret.classification import *

fraud_detector = setup(data = data, target = 'Fraudulent') 


# In[68]:


# compare() function compares all exisiting machine learning algorithms and automatically computes the highest scores.
# catboost model has been blacklisted since it takes a lot of time computing and over time it can affect the performance and speed of the system.
# AUC is considered a better factor to measure performace instead of accuracy because,
# Accuracy ignores probability estimations of classification in favor of class labels
compare_models(blacklist = ['catboost'], sort = "AUC")


# In[18]:


# sample reference names of the machine learning models which will be used in the code.
models()


# In[19]:


# Since the Light Gradient Boosting Machine has the best classification scores, that we will be our preferred model.


# In[71]:


final_model = create_model('lightgbm')
# Scores are computed for the 70% TRAINING DATA SCORES (620,30)


# In[72]:


tuned_final_model = tune_model(final_model, optimize = "AUC", n_iter = 500)
# We tune the hyperparameters with the hope that the scores might improve. 
# n_iter = 500 will randomly search for hyper-parameters from 500 hyper-parameter combinations


# In[73]:


# To get the best hyper-parameters, print the tuned model object.
tuned_final_model


# In[43]:


# For the most part here, the tuned model has better scores than the untuned model,
# so we will proceed with the tuned model (tuned_final_model)


# In[74]:


plot_model(tuned_final_model,plot='learning')


# In[75]:


plot_model(tuned_final_model,plot='feature')
# it seems time, amount, and v4 are the key fraud predictor factors.


# In[76]:


plot_model(tuned_final_model,plot='confusion_matrix')


# In[77]:


plot_model(tuned_final_model,plot='class_report')


# In[78]:


plot_model(tuned_final_model,plot='auc')


# In[79]:


plot_model(tuned_final_model, plot='error')


# In[80]:


plot_model(tuned_final_model, plot='boundary')


# In[87]:


plot_model(tuned_final_model, plot='pr')


# In[81]:


plot_model(tuned_final_model, plot='vc')


# In[82]:


# we will now run our final model and predict the Fraud on the test data (266,31).
final_predictions = predict_model(tuned_final_model)
# TEST SET SCORES


# In[83]:


# Since the accuracy and the AUC of test set <= training set, this model can be considered valid.
# AUC of 0.9923 is really very good and close to the training AUC of 0.9926
final_predictions
# a detailed view of predictions on the test model (the last 3 coloumns)


# In[84]:


finalize_model(tuned_final_model)
#we finalize our model thus ending the  predictive model analysis
# below are some parameters of our model which can be manipulated in the setup() function above.


# In[85]:


save_model(tuned_final_model,'final_model')
#we fully confirm our model and it is ready to be deployed on foreign test sets.


# In[86]:


#we now deploy our model on a foreign test set i.e the one we initially divided with 90-10 ratio (98,33)

new_test_set = predict_model(tuned_final_model, data = data_unseen)
new_test_set.head(98)

# our model correctly predicts the label compared to the ground truth (default Fradulent section) with a very high score
# which proves we have built a successful model.


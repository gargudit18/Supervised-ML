#!/usr/bin/env python
# coding: utf-8

# ## TASK 1 - Prediction using Supervised Machine Learning
# ## NAME   - Udit Garg

# Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Reading Dataset


# In[2]:


data=pd.read_csv('http://bit.ly/w-data')


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.corr()


# In[ ]:


#There is a strong correlation between Scores and Hours.


# DATA VISUALISATION

# In[10]:


plt.rcParams['figure.figsize']=10,5
data.plot(x='Hours',y='Scores',style='o')
plt.title("Hours V/s Percentage")
plt.xlabel('Hours Studies')
plt.ylabel('Percentage Scored')
plt.show()


# SPLITTING THE DATA

# In[17]:


#Divide the data into dependent and independent variables
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


# In[18]:


# Split thE data into training and test sets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[19]:


#Train the Data
regression = LinearRegression()
regression.fit(X_train, y_train)


# In[23]:


#Plotting the regression line(y=mx+c)
line = regression.coef_*x+regression.intercept_
# Plotting for the test data
plt.scatter(X, y, color = 'blue')
plt.plot(X, line, color='red')
plt.show()


# TESTING THE MODEL

# In[25]:


print(X_test) # Testing data - In Hours
y_pred = regression.predict(X_test) # Predicting Scores.


# In[26]:


#Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# MODEL EVALUATION

# In[28]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# PREDICTION

# In[ ]:


##Predict the value of percentage scored when number of hours studied is 9.25 hours.


# In[34]:


regression.predict([[9.25]])


# The score of a student if he studies for 9.25 hrs/day is 93.69173249

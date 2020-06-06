#!/usr/bin/env python
# coding: utf-8

# # Linear Regression using Python
# 

# ## Where can Linear Regression be used?

# 
# It is a very powerful technique and can be used to understand the factors that influence profitability. It can be used to forecast sales in the coming months by analyzing the sales data for previous months. It can also be used to gain various insights about customer behaviour. By the end of the blog we will build a model which looks like the below picture i.e, determine a line which best fits the data.
# 

# ## What is Linear Regression
# 

# The objective of a linear regression model is to find a relationship between one or more features(independent variables) and a continuous target variable(dependent variable). When there is only feature it is called Uni-variate Linear Regression and if there are multiple features, it is called Multiple Linear Regression.
# 

# ## Data-set
# 

# ### Using math

# In[8]:


#Let’s create some random data-set to train our model.
import random
import numpy as np
import matplotlib.pyplot as plt
random.seed(0)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)

# plot
plt.scatter(x,y,s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[14]:


#split data intro x and y var
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)


# In[15]:


mean_x = np.mean(x)
mean_y = np.mean(y)
m = len(x)


# In[17]:


numer = 0
demon = 0
for i in range(m):
    numer += (x[i]-mean_x)*(y[i]-mean_y)
    demon += (x[i]-mean_x)**2
b1 = numer/demon
b0 = mean_y-(b1*mean_x)
print(b1,b0)


# In[19]:


max_x = np.max(x)+100
min_x = np.min(x)-100
X = np.linspace(min_x,max_x,1000)
Y = b1+b0*X


# In[39]:



plt.scatter(x,y,color='black',label='scatterplot')
plt.plot(X,Y,color='red',label='RegressionLine')
plt.xlabel('Headsize in cm3')
plt.ylabel('brainsize in gram')
plt.legend('regression')
plt.show()


# # accuracy

# In[31]:


ss_t = 0
ss_r = 0
for i in range(m):
    y_pred = b0+b1*X[i]
    ss_t += (Y[i]-mean_y)**2
    ss_r += (Y[i]-y_pred)**2
r2 = 1-ss_r/ss_t
print(r2)


# # By scikit learn

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[11]:


# generate random data-set
np.random.seed(0)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)

# sckit-learn implementation

# Model initialization
regression_model = LinearRegression()
# Fit the data(train the model)
regression_model.fit(x, y)
# Predict
y_predicted = regression_model.predict(x)


# In[12]:


# model evaluation
rmse = mean_squared_error(y, y_predicted)
r2 = r2_score(y, y_predicted)

# printing values
print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)


# In[13]:


# plotting values

# data points
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')

# predicted values
plt.plot(x, y_predicted, color='r')
plt.show()


# In[ ]:





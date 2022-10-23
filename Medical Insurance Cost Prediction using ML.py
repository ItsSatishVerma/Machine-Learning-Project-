#!/usr/bin/env python
# coding: utf-8

# # SATISH  VERMA  , STATISTICS AND COMPUTING (BHU)

# ## Machine Learning in insurance
# 

# ### Machine Learning (ML) is all about programming the unprogrammable. For example, if you want to predict an insurance price, ML helps to predict the price. An insurance price depends on various features such as age, type of coverage, amount of coverage needed, gender, body mass index (BMI), region, and other special factors like smoking to determine the price of the insurance.
# 
# ### Traditionally most insurance companies employ actuaries to calculate the insurance premiums. Actuaries are business professionals who use mathematics and statistics to assess the risk of financial loss and predict the likelihood of an insurance premium and claim, based on the factors/features like age and gender, etc. They typically produce something called an actuarial table provided to an insurance company’s underwriting department, which uses the input to set insurance premiums. The insurance company calculates and writes all the programs, but it becomes much simpler by using Machine Learning.
# 
# ### Machine Learning allows a program to learn from a set of data to figure out particular problem characteristics. The ML program looks at different factors like gender, smoking, the number of children, and region to find the overall highest medical charges and determine the price by using specific algorithms based upon the requirement. Smokers and customers with more children tend to have higher medical costs. Hence premiums will be more for those groups. As ML trains more and more data, the ML program becomes more intelligent and smarter in predicting the exact price. In the end, you have a function/program to call to get the insurance premium for a particular person based upon the input factors provided. You don’t need to write all of these constructs yourself. ML program looks at all the sets of data provided and trains/learns, and it gives a function, and this function is a machine learning model that you can use in your application.
# 

# ## Abstract :-
# 

# ### Insurance is a policy that eliminates or decreases loss costs occurred by various risks. Various factors influence the cost of insurance. These considerations contribute to the insurance policy formulation. Machine learning (ML) for the insurance industry sector can make the wording of insurance policies more efficient. This study demonstrates how different models of regression can forecast insurance costs. And we will compare the results of models, for example, Linear Regression, Decision tree, Random Forest Regressor, Ridge regression

# ## Dataset:-
# 

# ### The medical cost personal datasets are obtained from the KAGGLE repository. This dataset contains seven attributes.

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


data = pd.read_csv('./insurance.csv')
data.head()


# ### For some general information about data

# In[9]:


data.info()


# In[10]:


data.shape


# ### The dataset that I am using for the task of health insurance premium prediction is collected from Kaggle.it has 1338 Data items/points and It contains data about:
# 
# #### 1.The age of the person
# #### 2.Gender of the person
# #### 3.Body Mass Index of the person
# #### 4.How many children the person is having
# #### 5.Whether the person smokes or not
# #### 6.The region where the person live 
# #### 7.Charges of the insurance premium
# 
# 
# 

# ### Before moving forward, let’s have a look at whether this dataset contains any null values or not:

# In[11]:


data.isnull().sum()


# ### There are no missing values as such

# ### This means we don’t have to worry about imputation or dropping rows or columns with missing data

# ### now for counting the number of elements in different features 

# In[12]:


data['region'].value_counts().sort_values()


# In[13]:


data['children'].value_counts().sort_values()


# ### Categorical Features:
#  (1).Sex
#  (2).Smoker
#  (3).Region

# ### Note: Regression algorithms seem to be working on features represented as numbers only By looking at our dataset we see that columns — ‘sex’, ‘smoker’ and ‘region’ are in string format, so we can work on converting them to numerical values as below 

# ### Converting Categorical Features to Numerical

# In[14]:


clean_data = {'sex': {'male' : 0 , 'female' : 1} ,
                 'smoker': {'no': 0 , 'yes' : 1},
                   'region' : {'northwest':0, 'northeast':1,'southeast':2,'southwest':3}
               }
data_copy = data.copy()
data_copy.replace(clean_data, inplace=True)


# In[15]:


data_copy.head()


# In[16]:


data_copy.describe()


# ## Feature Engineering and Correlation Matrix:-

# In[17]:


corr = data_copy.corr()
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr,cmap='BuPu',annot=True,fmt=".2f")
plt.title("Dependencies of Medical Charges")
plt.show()


# In[18]:


plt.figure(figsize=(6,6))
sns.pairplot(data_copy)
plt.show()


# ### When it comes to machine learning, feature engineering is the process of extracting features from raw data while applying domain expertise in order to improve the performance of ML algorithms. In the medical insurance cost dataset, attributes such as smoker, BMI, and age are the most important factors that determine charges. Also, we see that sex, children, and region do not affect the charges. We might drop these 3 columns as they have less correlation by plotting the heat map graph to see the dependency of dependent value on independent features. The heat map makes it easy to identify which features are most related to the other features or the target variable. Outcomes are shown in above Figure
# 
# 

# ## Results and Analysis:-
# ### The results of applied ML models are discussed in this section. Now for this, we can proceed with exploratory data analysis for plotting feature vs. feature (charges) for data visualization.

# In[19]:


print(data['sex'].value_counts().sort_values()) 
print(data['smoker'].value_counts().sort_values())
print(data['region'].value_counts().sort_values())


# ### Now we are confirmed that there are no other values in above pre-preocessed column, We can proceed with EDA

# ## Age vs. Charges

# In[20]:


plt.figure(figsize=(12,9))
plt.title('Age vs Charge')
sns.barplot(x='age',y='charges',data=data_copy,palette='husl')


# ### We can see in figure that with the growing age, the insurance charges are going to be increased. For example, when the age touches 64, the insurance charge is 23000, as shown in Figure .Age is shown on the x-axis, and charges are given on the y-axis

# ## Region vs charges:-

# In[21]:


plt.figure(figsize=(10,7))
plt.title('Region vs Charge')
sns.barplot(x='region',y='charges',data=data_copy,palette='Set3')


# ### Insurance charges vary concerning certain regions as shown in Figure The health insurance charges in the southeast are greater than in other regions. The region is displayed on the x-axis, and charges are shown on the y-axis.

# ## BMI vs Charges

# In[22]:


plt.figure(figsize=(7,5))
sns.scatterplot(x='bmi',y='charges',hue='sex',data=data_copy,palette='Reds')
plt.title('BMI VS Charge')


# ### In this Figure the zero value is used to represent the females and one value is used for the males. The BMI values of sex or gender types (male and female) are given in the x-axis, and the charges are presented in the y-axis. It can be clearly seen that when the values of BMI are varied, the insurance charges will vary accordingly as shown in Figure .

# 
# 
# 
# ## Smoker vs Charge

# In[23]:


plt.figure(figsize=(10,7))
plt.title('Smoker vs Charge')
sns.barplot(x='smoker',y='charges',data=data_copy,palette='Blues',hue='sex')


# ### The Figure illustrates that as a normal smoker, the medical insurance cost varies slightly. However, men are more addicted and passionate to smoking as compared to women so the health insurance cost for females is greater as compared to the males. We can see in Figure that with the increase of smoking habits, the insurance charges are going to be decreased for men and increased for women. Smokers’ values are shown on the x-axis, and charges are shown on the y-axis.
# 
# 

# 
# 
# 
# 
# ##  Sex vs. Charges
# 

# In[24]:


plt.figure(figsize=(10,7))
plt.title('Sex vs Charges')
sns.barplot(x='sex',y='charges',data=data_copy,palette='Set1')


# ### The medical insurance charges for the female gender are always greater than for the male as shown in Figure. It gives the sex types on the x-axis and the charges on the y-axis. The figure illustrates that the insurances charges for the female are 14000, and for the male, the charges are around 13000.

# ### For Boxplot

# In[25]:


for i in range(0,6):
    sns.boxplot(data_copy.iloc[:,i])
    plt.show()


# 
# 
# ## Skewness and Kurtosis

# ### Skewness is a metric that quantifies symmetry in a given scenario, or more specifically, the lack of it. If a distribution or data set appears the same on all sides of the graph to the left and right of the centre point, it is said to be symmetric. Kurtosis is a measure of how heavy-tailed or light-tailed the data are when compared to the normal distribution, according to the normal distribution. Heavy tails or outliers are more probable in data sets with a high kurtosis than data sets with a low kurtosis. When there is a low kurtosis in a data collection, it is more likely that there will be no outliers . The most extreme instance would be if there is a uniform distribution. Table displays the values for the skew and kurtosis of the attributes of a medical dataset.

# In[26]:


print('Printing Skewness and Kurtosis for all columns')
print()
for col in list(data_copy.columns):
    print('{0} : Skewness {1:.3f} and  Kurtosis {2:.3f}'.format(col,data_copy[col].skew(),data_copy[col].kurt()))


# In[27]:


table = pd.DataFrame( {'skewness':data_copy.skew(), 'kurtosis': data_copy.kurt()})


# In[28]:


print(table)


# In[29]:


plt.figure(figsize=(10,7))
sns.distplot(data_copy['age'])
plt.title('Plot for Age')
plt.xlabel('Age')
plt.ylabel('Count')


# In[30]:


plt.figure(figsize=(10,7))
sns.distplot(data_copy['bmi'])
plt.title('Plot for BMI')
plt.xlabel('BMI')
plt.ylabel('Count')


# In[31]:


plt.figure(figsize=(10,7))
sns.distplot(data_copy['charges'])
plt.title('Plot for charges')
plt.xlabel('charges')
plt.ylabel('Count')


# ### There might be few outliers in Charges but then we cannot say that the value is an outlier as there might be cases in which Charge for medical was very les actually!

# ### Preparing data - We can scale BMI and Charges Column before proceeding with Prediction

# In[32]:


from sklearn.preprocessing import StandardScaler
data_pre = data_copy.copy()

tempBmi = data_pre.bmi
tempBmi = tempBmi.values.reshape(-1,1)
data_pre['bmi'] = StandardScaler().fit_transform(tempBmi)

tempAge = data_pre.age
tempAge = tempAge.values.reshape(-1,1)
data_pre['age'] = StandardScaler().fit_transform(tempAge)

tempCharges = data_pre.charges
tempCharges = tempCharges.values.reshape(-1,1)
data_pre['charges'] = StandardScaler().fit_transform(tempCharges)

data_pre.head()


# ###  Next we will split our dataset(insurance.csv) into a training set and a testing set. We will train our model on the training set and then use the test set to evaluate the model(Predict ‘y’ variable). Please note that we will also compare the testing set predicted results with actual results.

# In[33]:


X = data_pre.drop('charges',axis=1).values
y = data_pre['charges'].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

print('Size of X_train : ', X_train.shape)
print('Size of y_train : ', y_train.shape)
print('Size of X_test : ', X_test.shape)
print('Size of Y_test : ', y_test.shape)


# In[34]:


from sklearn.linear_model import LinearRegression
model1=LinearRegression()
model1.fit(X_train,y_train)
LinearRegression()
from sklearn.metrics import r2_score
y_pred1=model1.predict(X_test)
rscore1=r2_score(y_test,y_pred1)
rscore1


# In[35]:


import statsmodels.api as sm
lin_reg=sm.OLS(y_train,X_train).fit()
lin_reg.summary()


# ## Assumptions of linear regression

# ### 1.Checking for normality of residuals

# In[36]:


residuals = lin_reg.resid
sm.qqplot(residuals)
plt.show()
np.mean(residuals)


# ### from this graph,we can conclude that the residuals are normally distributed

# ## 2.Checking for homoscedasticity

# In[37]:


plt.scatter(lin_reg.predict(X_train), residuals)
plt.plot(y_train, [0]*len(y_train),c='r')


# ### from the above graph we can say that data is homoscedasticity

# In[38]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)


# In[47]:


X1=pd.DataFrame(X_train)
VIF=calc_vif(X1)
print(X1)


# In[48]:


print(VIF)


# ## For training and testing the models:
# 

# ## LinearRegression

# In[44]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred_lr=lr.predict(X_test)


# In[45]:


print('Train Score: ', lr.score(X_train, y_train))  
print('Test Score: ', lr.score(X_test, y_test))


# ## DecisionTreeRegressor

# In[46]:


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(X_train,y_train)
y_pred_dt=dt.predict(X_test)


# In[47]:


print('Train Score: ', dt.score(X_train, y_train))  
print('Test Score: ', dt.score(X_test, y_test))


# ## Random Forest Regressor

# In[48]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(X_train,y_train)
y_pred_rf=rf.predict(X_test)


# In[49]:


print('Train Score: ', rf.score(X_train, y_train))  
print('Test Score: ', rf.score(X_test, y_test))


# ## Support Vector Machine (Regression)

# In[50]:


from sklearn.svm import SVR
svr= SVR(kernel = 'rbf' , C=10, gamma=0.1, tol=0.0001)
svr.fit(X_train,y_train)
y_pred_svr=svr.predict(X_test)


# In[51]:




print('Train Score: ', svr.score(X_train, y_train))  
print('Test Score: ',svr.score(X_test, y_test))


# ## Ridge Regression

# In[52]:


from sklearn.linear_model import Ridge
ridge=Ridge()
ridge.fit(X_train,y_train)
y_pred_r=ridge.predict(X_test)


# In[53]:


print('Train Score: ', ridge.score(X_train, y_train))  
print('Test Score: ', ridge.score(X_test, y_test))


# ## GradientBoostingRegressor

# In[54]:


from sklearn.ensemble import GradientBoostingRegressor
gb=GradientBoostingRegressor()
gb.fit(X_train,y_train)
y_pred_gb=gb.predict(X_test)


# In[55]:


print('Train Score: ', gb.score(X_train, y_train))  
print('Test Score: ',gb.score(X_test, y_test))


# In[56]:


ab=[np.array([20,1,28,0,1,3])]
ab


# In[57]:


a =dt.predict(np.array(ab))
print(a)


# In[58]:


models = [('Linear Regression',lr.score(X_train, y_train),lr.score(X_test, y_test)),
          ('Decision Tree Regression',dt.score(X_train, y_train),dt.score(X_test, y_test)),
          ('Random Forest Regression',rf.score(X_train, y_train),rf.score(X_test, y_test)),
          ('Support Vector Regression',svr.score(X_train, y_train),svr.score(X_test, y_test)),
          ('Ridge Regression',ridge.score(X_train, y_train),ridge.score(X_test, y_test)),
          ('Gradient Boosting Regression',gb.score(X_train, y_train),gb.score(X_test, y_test))
         ]


# In[59]:


predict = pd.DataFrame(data = models, columns=['Model', 'R2_Score(training)', 'R2_Score(test)'])
predict


# # Conclusion:
# 
# ## In this project, by using a set of ML algorithms, a computational intelligence approach is applied to predict healthcare insurance costs. The medical insurance dataset was obtained from the KAGGLE repository and was utilised for training and testing the Linear Regression, Ridge Regressor, Support Vector Regression, GradientBoostingRegressor, Decision Tree and Random Forest Regressor, ML algorithms. The regression analysis of this dataset followed the steps of preprocessing, feature engineering, data splitting, Fitting the regression models, and evaluation. The resultant outcome revealed that Gradient Boosting Regressor achieved a high accuracy of 87.9% .

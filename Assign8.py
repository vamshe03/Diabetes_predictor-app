#!/usr/bin/env python
# coding: utf-8

# # 1. Data Exploration:
# a. Load the dataset and perform exploratory data analysis (EDA).
# 
# b. Examine the features, their types, and summary statistics.
# 
# c. Create visualizations such as histograms, box plots, or pair plots to visualize the distributions and relationships between features.
# 
# Analyze any patterns or correlations observed in the data.
# 

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df=pd.read_csv("diabetes (1).csv")
df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


#checking for null values
df.isnull().sum()


# In[8]:


# Histogram
df.hist(figsize=(10,8))
plt.show()

# Boxplot for outlier Detection
plt.figure(figsize=(10,8))
sns.boxplot(data=df)
plt.xticks(rotation=60)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[9]:


# pair plot
sns.pairplot(df, hue="Outcome")
plt.show()


# ###  Patterns and Observations
# 
# From the exploratory analysis:
# 
# Glucose level has the strongest relationship with diabetes outcome.
# 
# BMI is moderately correlated with diabetes.
# 
# Higher BMI increases risk.
# 
# Age and pregnancies show some relationship with diabetes.
# 
# Some variables such as Insulin and SkinThickness contain outliers.
# 
# The dataset contains mostly numeric variables, making it suitable for machine learning models.

# # Data Preprocessing
# 
# a. Handle missing values (e.g., imputation).
# 
# b. Encode categorical variables.
# 

# In[10]:


# Check missing values
df.isnull().sum()


# In[11]:


df.tail()


# In[12]:


import numpy as np
cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for col in cols:
    df[col] = df[col].replace(0, df[col].median())
    


# In[13]:


df.isnull().sum()


# ### 
# b. Encode categorical variables
# 
# the diabetes dataset doesn't contain any categorial column,so encoding is not neccessary.
# 
# However, if categorical variables existed, we could encode them using One-Hot Encoding:
# 
# df = pd.get_dummies(df, drop_first=True)

# # Model Building
# 
# a.Build a logistic regression model using appropriate libraries (e.g., scikit-learn).
# 
# b. Train the model using the training data.
# 
# 

# In[14]:


from sklearn.model_selection import train_test_split

#Splitting the data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Training the Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# # 4. Model Evaluation:
# 
# a. Evaluate the performance of the model on the testing data using accuracy, precision, recall, F1-score, and ROC-AUC score.
# Visualize the ROC curve.

# In[15]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))



# In[16]:


# ROC-AUC
y_prob = model.predict_proba(X_test)[:,1]
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# ROC Curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr,label="Logistic Regression")
plt.plot([0,1], [0,1], linestyle="--",color='red')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# # 5.Interpretation
# 
# a. Interpret the coefficients of the logistic regression model.
# 
# b. Discuss the significance of features in predicting the target variable (survival probability in this case).
# 
# 
# 

# In[17]:


coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

print(coefficients)


# ### b) Feature Significance
# 
# Important features for predicting diabetes include:
# 
# Glucose
# 
# Strongest predictor of diabetes.
# 
# Higher blood sugar levels significantly increase risk.
# 
# BMI
# 
# Indicates obesity, which is strongly linked to diabetes.
# 
# Age
# 
# Risk of diabetes increases with age.
# 
# Pregnancies
# 
# Higher pregnancies may increase risk in some cases.
# 
# Diabetes Pedigree Function
# 
# Indicates genetic influence on diabetes.

# In[18]:


import joblib
# SAVE MODEL
joblib.dump(model, "model.pkl")

print("model.pkl created successfully!")


# In[ ]:





# In[ ]:





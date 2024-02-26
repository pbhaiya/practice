#!/usr/bin/env python
# coding: utf-8

# ## **Sample Project : Bank Account Fraud Detection**

# In[1]:


pip install ipykernel


# ### Importing Necessary Libraries and Data from Azure Data Storage 
# 
# - Took a sample dataset from kaggle, stored it in the local system, saved the data in the azure data storage and loaded it in the notebook.

# In[2]:


import pandas as pd

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
data_asset = ml_client.data.get("BankAccountFraudDetection", version="1")

df = pd.read_csv(data_asset.path)
df.head()


# ### Data Understanding & Feature Understanding-

# In[3]:


print(df.info(), "The shape of the dataset is", df.shape)


# ## Feature Engineering-
# 
# Converted the feature Credit_Risk_Score to bins, to 

# In[4]:


def cred_score(x):
    if x<0:
        return 'Risky Applicant'
    if x>0 and x<100:
        return 'Average Applicant'
    if x>100 and x<250:
        return 'Standard Applicant'
    if x>250:
        return 'Good Applicant'

df['Credit_Score'] = df['credit_risk_score'].apply(cred_score)
    


# In[5]:


df['fraud_bool'].value_counts()


# In[6]:


import matplotlib.pyplot as plt


# In[37]:


credit_counts = df['Credit_Score'].value_counts().sort_index()
cc = credit_counts.plot(kind='bar', color='cyan')
plt.xlabel('Applicant')
plt.ylabel('Count of Customers')
plt.title('Count of Customers w.r.t Credit Scores')

for i,v in enumerate(credit_counts):
    cc.text(i,v,str(v), ha='center', va='bottom')
plt.show()


# 1 - Good Applicant
# 2 - Average Applicant
# 3 - Standard Applicant
# 4 - Risky Applicant

# ###### We have comparatively lesser Risky Applicants and a good number of Standard applicants who have a good credit score

# In[8]:


df['customer_age'].unique()


# In[9]:


df.describe(include='object')


# In[10]:


from sklearn import preprocessing


# In[11]:


le = preprocessing.LabelEncoder()
df['Credit_score'] = le.fit_transform(df['Credit_Score'])

df['Credit_Score'].unique()


# ## Converting the Income Column into categories - Low Income to High Income

# In[12]:


#Defining Income Thresholds

low_threshold = 0.3
high_threshold = 0.6

def categorize_income(income):
    if income <= low_threshold:
        return 1
    elif low_threshold < income <= high_threshold:
        return 2
    else:
        return 3

# Apply the categorize_income function to create the "income_group" column
df['income_group'] = df['income'].apply(categorize_income)

df['income_group'] = df['income_group'].astype('int64')

column_to_move = df.pop('income_group')

df.insert(1, 'income_group', column_to_move)
df = df.drop(columns=['income'])

df.head()


# ## Number of Customers by Age-

# In[13]:


age_counts = df['customer_age'].value_counts().sort_index()

ax = age_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Age')
plt.ylabel('Number of Customers')
plt.title('Number of Customers by Age')

for i, v in enumerate(age_counts):
    ax.text(i, v, str(v), ha='center', va='bottom')

plt.show()


# #### Count of each category-

# In[14]:


numerical_features = df.select_dtypes(include=['int64', 'float64'])
categorical_features = df.select_dtypes(include='object')

for i in categorical_features:
    print(df[i].value_counts())


# In[28]:


from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing  
label_encoder = preprocessing.LabelEncoder() 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
  


# #### Converting 'obj' data to labels, using label encoder-

# In[16]:


df['payment_type']= label_encoder.fit_transform(df['payment_type']) 
df['employment_status']= label_encoder.fit_transform(df['employment_status']) 
df['housing_status']= label_encoder.fit_transform(df['housing_status']) 
df['source']= label_encoder.fit_transform(df['source']) 
df['device_os']= label_encoder.fit_transform(df['device_os'])
df['Credit_Score']= label_encoder.fit_transform(df['Credit_Score'])


# sc = StandardScaler()
# df_std = sc.fit_transform(df)
# 
# df_std

# df_scaled = pd.DataFrame(sc.fit_transform(df_std),columns = df.columns)
# 
# df_scaled.head()

# df_scaled.shape

# ### **Building the First_Base_Model-**

# In[17]:


X = df.drop('fraud_bool',axis=1)
y = df['fraud_bool']

xtrain,xtest,ytrain,ytest = train_test_split(X,y, test_size=0.3, random_state=1)


# In[18]:


dt_base = DecisionTreeClassifier(random_state=1)

dt_base.fit(xtrain, ytrain)



# In[19]:


from sklearn.metrics import classification_report


# In[20]:


train_pred = dt_base.predict(xtest)
prob = dt_base.predict_proba(xtest)
train_pred_dt = dt_base.predict(xtrain)
dt_base_prob = prob[:,1]


# In[21]:


print('Classification Report :' , classification_report(ytrain, train_pred_dt) )


# ##### This is the case of overfitting, since our target variable was very biased for one class, hence we perform some sampling techniques and try to balance out our target variable.

# In[22]:


pip install imblearn


# In[23]:


from imblearn.over_sampling import SMOTE


# In[24]:


sm = SMOTE(sampling_strategy='auto', random_state=1)
X_new , Y_new = sm.fit_resample(X,y)
df_smote = pd.concat([X_new, Y_new])

print('The distributon of classes after over-sampling is : ' , Y_new.value_counts().sort_index())


# In[27]:


## We will build a model with above newly created dataframe, and split the data into X & Y new.

xtrain_sm, xtest_sm, ytrain_sm, ytest_sm = train_test_split(X_new,Y_new, test_size=0.3, random_state=2)


# In[ ]:


## Building a Random Forest Classifier-

rf = RandomForestClassifier()
rf_smote = rf.fit(xtrain_sm, ytrain_sm) #fitting the after SMOTE Samples

trainpred_rf_smote = rf_smote.predict(xtrain_sm) ##Predicting the train values
testpred_rf_smote = rf_smote.predict(xtest_sm) ## Predicting the test values
prob1 = rf_smote.predict_proba(xtest_sm)
rf_smote_pred_prob = prob[:,1]


# In[35]:


print('Classification Report for train data', classification_report(ytrain_sm,trainpred_rf_smote))
print('Classification Report for Test Data : ' , classification_report(ytest_sm,testpred_rf_smote))


# In[ ]:





# In[ ]:





# In[25]:


ws = Workspace.create(name='myworkspace',
           subscription_id='9c09575e-cca3-47d5-95ea-f708138be66a',
           resource_group='parulbhaiya-rg',
           create_resource_group=True,
           location='eastus2'
           )


# In[ ]:





# In[ ]:





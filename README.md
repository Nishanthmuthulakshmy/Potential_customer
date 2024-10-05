# Potential_customer
Propensify is a propensity modeling project designed to identify the likelihood of specific customer segments responding to marketing campaigns. As businesses invest significant resources into data-driven marketing strategies, understanding customer behavior is crucial for optimizing these efforts. This project aims to forecast customer responses

# Data Overview
The project uses datasets containing customer information, which includes features such as age, profession, marital status, and other relevant attributes. The target variable indicates whether the customer responded positively to marketing efforts.


# ## Propensity model to identify potential customers
# 

# In[1]:


#importing all the neccessory libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# In[2]:


#Loading the training data set
train = pd.read_excel('train.xlsx')


# In[3]:


#checking the shape
train.shape


# In[4]:


# Have a look at the data
train.head(30)


# In[5]:


# Datatypes of the columns
train.dtypes


# In[6]:


# checkout the duplicate rows
duplicate = train[train.duplicated()]
print("Duplicate Rows :") 
duplicate


# In[7]:


train.drop_duplicates(inplace=True)


# In[8]:


train.groupby(['profession']).count().plot(kind='pie', y='responded',figsize = (10,11))


# In[9]:


# Group by 'profession' and count 'responded'
profession_counts = train.groupby('profession')['responded'].count()

# Sort the counts in descending order
profession_counts = profession_counts.sort_values(ascending=False)

# Calculate cumulative percentage
cumulative_percentage = profession_counts.cumsum() / profession_counts.sum() * 100

# Plot the Pareto chart
fig, ax = plt.subplots(figsize=(12, 6))  # Set figure size for better readability

# Bar chart (counts of 'responded')
ax.bar(profession_counts.index, profession_counts, color='blue')

# Line chart (cumulative percentage)
ax2 = ax.twinx()
ax2.plot(profession_counts.index, cumulative_percentage, color='red', marker='o')

# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha='right')

# Labeling
ax.set_xlabel('Profession')
ax.set_ylabel('Count of Responded', color='blue')
ax2.set_ylabel('Cumulative Percentage (%)', color='red')
plt.title('Pareto Chart for Profession based on Responded')

# Adjust the layout to prevent label cutoff
plt.tight_layout()

# Display the chart
plt.show()


# As we can see, majority of our potential customer comes from admin,blue-collar, technician workgroup. followed by services abd Management  
# these 6 groups contributes to more than 80% of the potental customer

# In[10]:


import seaborn as sns
plt.figure(figsize=(15,15))
sns.pairplot(train[['campaign','pdays','previous','responded']],hue='responded')
plt.show()


# In[11]:


# Checking Null Values
train.isna().sum()/len(train)*100


# In[12]:


#lets find out the Average age of each proffession
mean_age_by_both = train.groupby(['profession','marital'])['custAge'].mean()
mean_age_by_both

# we can see the age group shows better variance when it is grouped on the
# basis of  profession and marital status


# In[13]:


# now lets fill the missing values of the custAge column with the mean

def fill_age(row):
    # Check if 'custAge' is missing
    if pd.isnull(row['custAge']):
        # Check if 'profession' or 'marital' is NaN
        if pd.isnull(row['profession']) or pd.isnull(row['marital']):
            return train['custAge'].mean()  # Fill with overall mean if either is NaN
        else:
            # Fill with the mean age for the corresponding profession + marital status
            return mean_age_by_both.get((row['profession'], row['marital']), train['custAge'].mean())  
    else:
        return row['custAge']  # If not missing, keep the original value

# Apply the function to fill missing 'custAge' values
train['custAge'] = train.apply(fill_age, axis=1)


# In[14]:


# Check if there are any remaining missing values in the Age column
train['custAge'].isnull().sum()/len(train)*100


# In[15]:


# Group by 'profession' and calculate the mode of 'schooling'
mode_schooling_by_profession = train.groupby('profession')['schooling'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

# Check the mode schooling for each profession (optional)
print(mode_schooling_by_profession)


# In[16]:


# Define a function to fill missing 'schooling' values using the group-specific mode
def fill_schooling(row):
    if pd.isnull(row['schooling']):  # Check if 'schooling' is missing
        # Fill with the mode schooling for the corresponding profession
        return mode_schooling_by_profession.get(row['profession'], train['schooling'].mode().iloc[0])  
    else:
        return row['schooling']  # If not missing, return the original value

# Apply the function to fill missing 'schooling' values
train['schooling'] = train.apply(fill_schooling, axis=1)


# In[17]:


# Check for any remaining missing values in 'schooling'
print(train['schooling'].isnull().sum())


# In[18]:


train.isna().sum()/len(train)*100


# In[19]:


# we can see that the column profit is having 88% missing values
# so lets drop that column
train.drop(columns=['profit'], inplace=True)


# In[20]:


train.isna().sum()


# In[21]:


# Check unique values and their frequency in 'day_of_week'
train.dropna(inplace=True)


# In[22]:


train.isna().sum()


# In[23]:


len(train)


# # Exploratory data analysis
# 

# In[24]:


#Lets start with descriptive statistics
train.describe()


# In[25]:


train.groupby(['profession']).count().plot(kind='pie', y='responded',figsize = (10,11))


# In[26]:


train.info()


# # Encoding and feature scaling

# In[27]:


#seperating the numerical and categorical columns
from sklearn.preprocessing import StandardScaler
def data_type(train):
    numerical = []
    categorical = []
    for i in train.columns:
        if train[i].dtype == 'int64' or train[i].dtype=='float64':
            numerical.append(i)
        else:
            categorical.append(i)
    return numerical, categorical
numerical, categorical = data_type(train)

# Identifying the binary columns and ignoring them from scaling
def binary_columns(train):
    binary_cols = []
    for col in train.select_dtypes(include=['int', 'float']).columns:
        unique_values = train[col].unique()
        if np.in1d(unique_values, [0, 1]).all():
            binary_cols.append(col)
    return binary_cols

binary_cols = binary_columns(train)


# In[28]:


# Remove the binary columns from the numerical columns
numerical = [i for i in numerical if i not in binary_cols]

def encoding(train, categorical):
    for i in categorical:
        train[i] = train[i].astype('category')
        train[i] = train[i].cat.codes
    return train

train = encoding(train, categorical)


# In[29]:


def feature_scaling(train, numerical):
    sc_x = StandardScaler()
    train[numerical] = sc_x.fit_transform(train[numerical])
    return train

train = feature_scaling(train, numerical)


# In[30]:


train.head(10)


# In[31]:


# now we can see that the 


# In[32]:


X = train.drop(columns=['responded'])
y = train['responded']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


# # logistic Regression Model

# In[36]:


# 1. Logistic Regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(solver='liblinear', max_iter=200)
logmodel.fit(X_train, y_train)
log_pred = logmodel.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)
log_cm = confusion_matrix(y_test, log_pred)
print(f'Logistic Regression Accuracy: {log_acc}')
print(f'Logistic Regression Confusion Matrix:\n{log_cm}')


# In[37]:


# 2. Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
dt_cm = confusion_matrix(y_test, dt_pred)
print(f'Decision Tree Accuracy: {dt_acc}')
print(f'Decision Tree Confusion Matrix:\n{dt_cm}')


# In[38]:


# 3. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_cm = confusion_matrix(y_test, rf_pred)
print(f'Random Forest Accuracy: {rf_acc}')
print(f'Random Forest Confusion Matrix:\n{rf_cm}')


# In[39]:


# 4. Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
gb_cm = confusion_matrix(y_test, gb_pred)
print(f'Gradient Boosting Accuracy: {gb_acc}')
print(f'Gradient Boosting Confusion Matrix:\n{gb_cm}')


# In[41]:


# 5. Support Vector Machine (SVM)
from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
svm_cm = confusion_matrix(y_test, svm_pred)
print(f'SVM Accuracy: {svm_acc}')
print(f'SVM Confusion Matrix:\n{svm_cm}')


# In[43]:


# 6. Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)
nb_cm = confusion_matrix(y_test, nb_pred)
print(f'Naive Bayes Accuracy: {nb_acc}')
print(f'Naive Bayes Confusion Matrix:\n{nb_cm}')


# The evaluation of various classification models revealed that tree-based methods, such as Decision Tree, Random Forest, and Gradient Boosting, achieved perfect accuracy (100%) on the test set, which may indicate overfitting. Logistic Regression also performed strongly with an accuracy of 98.86%, effectively identifying candidates to market. In contrast, Support Vector Machine and Naive Bayes classifiers had lower accuracies of 95.91% and 95.44%, respectively, demonstrating higher false negatives.

# In[50]:


# Load the unseen data
unseen_data = pd.read_excel('test.xlsx')


# In[56]:


print(unseen_data.isnull().sum())


# In[51]:


unseen_data.info()


# In[57]:


# Calculate mean age grouped by profession and marital status for unseen data
mean_age_by_both_unseen = unseen_data.groupby(['profession', 'marital'])['custAge'].mean()

# Function to fill missing 'custAge' values in unseen_data
def fill_age_unseen(row):
    # Check if 'custAge' is missing
    if pd.isnull(row['custAge']):
        # Check if 'profession' or 'marital' is NaN
        if pd.isnull(row['profession']) or pd.isnull(row['marital']):
            return train['custAge'].mean()  # Fill with overall mean if either is NaN
        else:
            # Fill with the mean age for the corresponding profession + marital status
            return mean_age_by_both_unseen.get((row['profession'], row['marital']), train['custAge'].mean())
    else:
        return row['custAge']  # If not missing, keep the original value

# Apply the function to fill missing 'custAge' values in unseen_data
unseen_data['custAge'] = unseen_data.apply(fill_age_unseen, axis=1)


# In[58]:


# Function to separate numerical and categorical columns
def data_type(unseen_data):
    numerical = []
    categorical = []
    for column in unseen_data.columns:
        if unseen_data[column].dtype == 'int64' or unseen_data[column].dtype == 'float64':
            numerical.append(column)
        else:
            categorical.append(column)
    return numerical, categorical

# Identify binary columns
def binary_columns(unseen_data):
    binary_cols = []
    for column in unseen_data.select_dtypes(include=['int', 'float']).columns:
        unique_values = unseen_data[column].unique()
        if np.in1d(unique_values, [0, 1]).all():
            binary_cols.append(column)
    return binary_cols

# Get numerical and categorical columns
numerical_unseen, categorical_unseen = data_type(unseen_data)

# Identify binary columns and remove them from numerical columns
binary_cols_unseen = binary_columns(unseen_data)
numerical_unseen = [col for col in numerical_unseen if col not in binary_cols_unseen]

# Function to encode categorical variables
def encoding(unseen_data, categorical):
    for column in categorical:
        unseen_data[column] = unseen_data[column].astype('category')
        unseen_data[column] = unseen_data[column].cat.codes
    return unseen_data

# Encode categorical variables in the unseen data
unseen_data = encoding(unseen_data, categorical_unseen)

# Feature scaling for numerical columns
def feature_scaling(unseen_data, numerical):
    sc_x = StandardScaler()
    unseen_data[numerical] = sc_x.fit_transform(unseen_data[numerical])
    return unseen_data

# Scale numerical columns in the unseen data
unseen_data = feature_scaling(unseen_data, numerical_unseen)


# In[59]:


import joblib

joblib.dump(rf_model, 'random_forest_model.pkl')


# In[60]:


model = joblib.load('random_forest_model.pkl')  
# Make predictions
predictions = model.predict(unseen_data)

# Add predictions to the unseen data DataFrame
unseen_data['predicted_market'] = predictions

# Save the updated unseen data with predictions to the same file
unseen_data.to_excel('test_with_predictions.xlsx', index=False)





#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings(action="ignore")
from scipy import stats


# In[3]:


#Converting the dataset into dara frame
df=pd.read_csv("UCI_Credit_Card.csv")


# In[4]:


df.head()


# In[5]:


#check 5 random samples
df.sample(5)


# In[6]:


#Renaming the columns
df = df.rename(columns={'default.payment.next.month': 'DEFAULT_PAY'})


# In[ ]:





# In[7]:


#Dropping the ID column as it is not necessary
df=df.drop('ID', axis=1)


# In[8]:


df.shape


# In[9]:


#To check the data typesof the dataframe
df.info()


# In[10]:


df.describe()


# In[11]:


df.isnull().sum()


# In[12]:


#Checking the duplicate values
df.duplicated().sum()


# In[13]:


#Dropping the duplicates
df = df.drop_duplicates()


# In[14]:


df['DEFAULT_PAY'].value_counts()


# In[15]:


#Check the correlation
df.corr()['DEFAULT_PAY']


# In[16]:


df.nunique()


# In[17]:


df['SEX'].value_counts().plot(kind='pie', autopct='%.2f')


# In[18]:


df.EDUCATION.value_counts()


# In[19]:


df.loc[:,"EDUCATION"]=df.loc[:,"EDUCATION"].replace(6,4)
df.loc[:,"EDUCATION"]=df.loc[:,"EDUCATION"].replace(0,4)
df.loc[:,"EDUCATION"]=df.loc[:,"EDUCATION"].replace(5,4)


# In[20]:


df.EDUCATION.value_counts()


# In[21]:


df.MARRIAGE.value_counts()


# In[22]:


df.loc[:,"MARRIAGE"]=df.loc[:,"MARRIAGE"].replace(0,3)


# In[23]:


df.MARRIAGE.value_counts()


# sns.boxplot(df['AGE'])

# In[24]:


sns.boxplot(df['SEX'], df['AGE'], hue=df['DEFAULT_PAY'])


# In[25]:


# Delete the outliers
# Calculate the IQR (Interquartile Range)
Q1 = df['AGE'].quantile(0.25)
Q3 = df['AGE'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds to identify outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df = df[(df['AGE'] >= lower_bound) & (df['AGE'] <= upper_bound)]


# In[26]:


sns.boxplot(df['SEX'], df['AGE'], hue=df['DEFAULT_PAY'])


# In[27]:


df["LIMIT_BAL"].describe()


# In[28]:


sns.boxplot(df['LIMIT_BAL'])


# In[29]:


# Delete the outliers
# Calculate the IQR (Interquartile Range)
Q1 = df['LIMIT_BAL'].quantile(0.25)
Q3 = df['LIMIT_BAL'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds to identify outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df = df[(df['LIMIT_BAL'] >= lower_bound) & (df['LIMIT_BAL'] <= upper_bound)]


# In[30]:


sns.boxplot(df['LIMIT_BAL'])


# In[31]:


df


# In[32]:


plt.figure(figsize=(7, 5))
plt.ylabel("Frequency")
res = stats.probplot(df["LIMIT_BAL"].dropna(), plot=plt)
plt.show()


# In[33]:


df.DEFAULT_PAY.value_counts()


# In[34]:


i = list(df[df['DEFAULT_PAY'] == 1]['LIMIT_BAL'])
j = list(df[df['DEFAULT_PAY'] == 0]['LIMIT_BAL'])

plt.figure(figsize=(10,4))
plt.hist([i, j], bins = 30, color=['purple', 'plum'])
plt.xlim([0,600000])
plt.legend(['Yes', 'No'], title = 'DEFAULT_PAY', loc='upper right', facecolor='white', shadow=True)
plt.xlabel('Limit Balance (in Dollars)')
plt.ylabel('Frequency')
plt.title('Limit Balance Histogram by "default_pay" type')


# In[35]:


u = list(df[df['DEFAULT_PAY'] == 1]['AGE'])
v = list(df[df['DEFAULT_PAY'] == 0]['AGE'])

plt.figure(figsize=(10,4))
plt.hist([u,v], bins = 20, color=['purple', 'plum'])
plt.legend(['Yes', 'No'], title = 'DEFAULT_PAY', loc='upper right', facecolor='white', shadow=True)
plt.xlabel('AGE')
plt.ylabel('Frequency')
plt.title('AGE Histogram by "default_pay" type')
plt.show()


# In[36]:


plt.figure(figsize=(4,4))

ax=sns.countplot(data=df, x="SEX", hue="DEFAULT_PAY", palette="magma")
for label in ax.containers:
    ax.bar_label(label)
plt.xticks([0,1], labels=["Male", "Female"])
plt.title("Sex variable distribution by Default_pay")
plt.legend(['No', 'Yes'], title = 'DEFAULT_PAY', loc='upper left', facecolor='white', shadow=True)
plt.show()


# In[37]:


plt.figure(figsize=(6,6))

ax=sns.countplot(data=df, x="EDUCATION", hue="DEFAULT_PAY", palette="magma")
for label in ax.containers:
    ax.bar_label(label)
plt.xticks([0,1,2,3], labels=["Grad School", "University", "High School", "Others"])
plt.title("Education variable distribution by default_pay type")
plt.legend(['No', 'Yes'], title = 'DEFAULT_PAY', loc='upper right', facecolor='white')
plt.show()


# In[38]:


plt.figure(figsize=(5,5))
ax=sns.countplot(data=df, x="MARRIAGE", hue="DEFAULT_PAY", palette="magma")
for label in ax.containers:
    ax.bar_label(label)
plt.xticks([0,1,2], labels=["Married", "Single", "Others"])
plt.title("Marriage variable distribution by default_pay type")
plt.legend(['No', 'Yes'], title = 'DEFAULT_PAY', loc='upper right', facecolor='white', shadow=True)
plt.show()


# In[39]:


df.info()


# In[40]:


CorrMat = df.corr()
plt.figure(figsize = (20,20))
sns.heatmap(CorrMat, annot=True)


# In[41]:


Corr_DEFPAY=CorrMat["DEFAULT_PAY"].sort_values(ascending=False)
print(Corr_DEFPAY.to_string())


# In[42]:


q = df.dtypes!='object'

for i in list(q[q].index):
    m = CorrMat[i]>0.8
    print('Feature',i,'highly corelated with',list(m[m].index))


# In[43]:


x = df.drop(columns='DEFAULT_PAY')
y = df['DEFAULT_PAY']


# In[44]:


y


# In[45]:


train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.7)
train_x.shape


# In[46]:


train_y.shape


# In[47]:


test_x.shape


# In[48]:


test_y.shape


# In[49]:


from sklearn import tree

model_1 = tree.DecisionTreeClassifier(random_state=0)
model_1.fit(train_x, train_y)


# In[50]:


from sklearn.tree import export_text


# export decision rules
tree_rules = export_text(model_1)
test_predict_dtree = model_1.predict(test_x)


# In[51]:


print(tree_rules)


# In[52]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import seaborn as sns

confusion_matrix = metrics.confusion_matrix(test_y,test_predict_dtree) #Predicted test values


matrix_df = pd.DataFrame(confusion_matrix) #plot the result
#matrix_df

ax = plt.axes()
sns.set(font_scale=1.3)
#plt.figure(figsize=(3,2))
sns.heatmap(matrix_df, annot=True, fmt="g", ax=ax)


# In[53]:


print('Correct Predictions: ', confusion_matrix[0,0]+confusion_matrix[1,1])
print('Incorrect Predictions: ', confusion_matrix[0,1]+confusion_matrix[1,0])


# In[54]:


from sklearn.metrics import classification_report

report = classification_report(y_true=test_y, y_pred=test_predict_dtree)
print(report)


# In[55]:


from sklearn.metrics import accuracy_score
accuracy_score(test_y, test_predict_dtree)


# In[ ]:





# In[ ]:





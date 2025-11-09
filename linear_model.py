import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/PankajMishra99/OOPS-QUESTION/main/new_insurance_data.csv'
df = pd.read_csv(url)

# df['region'].mode()[0]

# df.isnull().sum()

# df.info()

for col in df.columns:
  if (df[col].dtype=='object'):
    df[col]=df[col].fillna(df[col].mode()[0])
  else:
    df[col]=df[col].fillna(df[col].mean())
df.isnull().sum()

# print(df.head())

from sklearn.preprocessing import LabelEncoder
def encoder(new_df):
  cate_col = ['sex','smoker','region']
  le=LabelEncoder()
  for col in cate_col:
    new_df[col] = le.fit_transform(new_df[col])
  return new_df
df=encoder(df)
df.head()

# check outliers
for col in df.columns:
  sns.boxplot(df[col])
  plt.xlabel(col)
  plt.show()

outlier_col = ['bmi','Anual_Salary','Hospital_expenditure','past_consultations']
def remove_outlier(new_df:pd.DataFrame):
  for col in outlier_col:
    q1=new_df[col].quantile(0.25)
    q3=new_df[col].quantile(0.75)
    iqr = q3-q1
    lb=q1-1.5*iqr
    ub=q3+1.5*iqr
    new_df=new_df[(new_df[col]>lb) & (new_df[col]<ub)]
  return new_df
df=remove_outlier(df)

# check outliers
for col in outlier_col:
  sns.boxplot(df[col])
  plt.xlabel(col)
  plt.show()

df.columns

x=df.drop('charges',axis=1)
y=df['charges']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=0)
print('x_train_shape : ',x_train.shape)
print('x_test_shape : ',x_test.shape)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
model

help(model.predict)

y_pred= model.predict(x_test)
y_pred

new_df=pd.DataFrame(columns=['Actual_Values','Predict_Values'])

new_df['Actual_Values']=y_test
new_df['Predict_Values']=y_pred

new_df = new_df.reset_index(drop='index')
new_df

from sklearn.metrics import *
acc= r2_score(y_test,y_pred)
acc

sns.regplot(x=y_pred,y=y_test)
plt.title('Regression plot')
plt.xlabel('Predicted charges')
plt.ylabel('Actual charges')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose               
 
import csv
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelBinarizer
from pandas import Timestamp
import datetime
import seaborn as sns
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

import pickle


# #### Importing the Dataset



dataset = pd.read_csv('data_daily.csv')
dataset.tail()




dataset.describe()


# #### Checking for null values



dataset.isna().sum().sum() 


# #### Visualizing the data



# plt.plot(dataset.iloc[:, 1])




# decompose time series
# plt.rcParams["figure.figsize"] = (10,6)
# result = seasonal_decompose(dataset['Receipt_Count'], model='multiplicative', period = 12)
# result.plot()
# plt.show()





temp = dataset['# Date']
dataset['# Date'] = pd.to_datetime(dataset['# Date'])





week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]





weekd = []
for i in range(len(dataset.iloc[:, :])):
    date = dataset.iloc[i, :][0]
   
    weekd.append(date.weekday())

dataset.insert(2,'weekday', weekd, True)
encoder = LabelBinarizer()

month = []
for i in range(len(dataset.iloc[:, :])):
    date = dataset.iloc[i, :][0]
dataset['month'] = pd.DatetimeIndex(dataset['# Date']).month

dataset.head()




cal = calendar()
holidays = cal.holidays(start=dataset['# Date'].min(), end=dataset['# Date'].max())


dataset['holiday'] = dataset['# Date'].isin(holidays)





# funciton code from medium website by author: Naina Chaturvedi
def days_prev_holiday(date, holidays):
    difference=[]
    for item in holidays:
        difference.append(int(((item-date)).days))
    return abs(max([x for x in difference if x<=0]))
def days_next_holiday(date, holidays):
    difference=[]
    for item in holidays:
        difference.append(int(str((item-date).days)))
    return min([x for x in difference if x>=0])



dataset['days_previous_holiday']= dataset.apply(lambda row: days_prev_holiday((row['# Date']), holidays), axis=1)
dataset['days_next_holiday']= dataset.apply(lambda row: days_next_holiday((row['# Date']), holidays), axis=1)





dataset['holiday'] = np.array(encoder.fit_transform(dataset['holiday']))




enc=OneHotEncoder()
 

enc_data=pd.DataFrame(enc.fit_transform(dataset[['weekday']]).toarray())
 
#Merge with main
df=dataset.join(enc_data)
df = df.rename(columns={0: 'M', 1: 'Tu', 2: 'W', 3: 'Th', 4: 'F', 5: 'Sa', 6: 'Su'})

enc_data=pd.DataFrame(enc.fit_transform(df[['month']]).toarray())
 
#Merge with main
df=df.join(enc_data)



df['# Date']=df['# Date'].map(datetime.datetime.toordinal)



from sklearn.model_selection import train_test_split
X = df[['# Date', 'holiday', 'days_previous_holiday','days_next_holiday','month']]
y = df['Receipt_Count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)



df.corr()['Receipt_Count']




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)




class MultipleLinearRegression:
    def __init__(self):
        self.beta = None
    
    def fit(self, X, y):
        # for bias 
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)

        # calculate coefficients using normal equations
        X_transpose = np.transpose(X)
        beta = np.linalg.inv(X_transpose @ X) @ X_transpose @ y
        self.beta = beta
    
    def predict(self, X):
        # to account for bias
        ones = np.ones((X.shape[0], 1))

        X = np.concatenate((ones, X), axis=1)
        y_pred = X @ self.beta

        return y_pred




regressor = MultipleLinearRegression()
regressor.fit(X_train, y_train)




y_pred = regressor.predict(X_test)

# Evaluate the performance of the model
mse = np.mean((y_test - y_pred)**2)
print("Mean Squared Error: ", mse)




y_mean = np.mean(y_test)
ss_tot = np.sum((y_test - y_mean)**2)
ss_res = np.sum((y_test - y_pred)**2)
r2 = 1 - (ss_res / ss_tot)











regressor.predict(np.array([[737791, 1, 0, 0, 4]]))







pickle.dump(regressor, open('model4.pkl','wb'))





model = pickle.load(open('model4.pkl','rb'))
print(model.predict(np.array([[737791, 1, 0, 0, 4]])))

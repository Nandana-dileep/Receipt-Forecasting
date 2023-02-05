# Receipt-Forecasting using Multiple Linear Regression Model
Accuracy - (90-95)%
Predicting the number of the observed scanned receipts from previous 2021 data

To run the model clik link --> https://fetch-model.herokuapp.com/

For detailed code of the model, open file "fetch-model jupyter.pdf"

```python
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

```

## Importing the Dataset


```python
dataset = pd.read_csv('data_daily.csv')
dataset.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># Date</th>
      <th>Receipt_Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>360</th>
      <td>2021-12-27</td>
      <td>10350408</td>
    </tr>
    <tr>
      <th>361</th>
      <td>2021-12-28</td>
      <td>10219445</td>
    </tr>
    <tr>
      <th>362</th>
      <td>2021-12-29</td>
      <td>10313337</td>
    </tr>
    <tr>
      <th>363</th>
      <td>2021-12-30</td>
      <td>10310644</td>
    </tr>
    <tr>
      <th>364</th>
      <td>2021-12-31</td>
      <td>10211187</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Receipt_Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.650000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.826566e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.820089e+05</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.095414e+06</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.142874e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8.799249e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.476970e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.073886e+07</td>
    </tr>
  </tbody>
</table>
</div>



### Checking for null values


```python
dataset.isna().sum().sum() 
```




    0



### Visualizing the data


```python
plt.plot(dataset.iloc[:, 1])
```




    [<matplotlib.lines.Line2D at 0x7f87ea482b50>]




    
![png](output_7_1.png)


    


### Visualizing the data using time-series decomposition


```python
# decompose time series
plt.rcParams["figure.figsize"] = (10,6)
result = seasonal_decompose(dataset['Receipt_Count'], model='multiplicative', period = 12)
result.plot()
plt.show()
```


    
<img src="Screen Shot 2023-02-04 at 8.33.23 PM.png" alt="Alt text" title="Time-series visualization">
    

## Data preprocessing


```python
temp = dataset['# Date']
dataset['# Date'] = pd.to_datetime(dataset['# Date'])
```

### Creating weekday and month columns from # Date


```python

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

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># Date</th>
      <th>Receipt_Count</th>
      <th>weekday</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-01</td>
      <td>7564766</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-02</td>
      <td>7455524</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-03</td>
      <td>7095414</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-04</td>
      <td>7666163</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-05</td>
      <td>7771289</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Feature engineering to extract more data from the dates column


```python
cal = calendar()
holidays = cal.holidays(start=dataset['# Date'].min(), end=dataset['# Date'].max())
holidays
```




    DatetimeIndex(['2021-01-01', '2021-01-18', '2021-02-15', '2021-05-31',
                   '2021-07-05', '2021-09-06', '2021-10-11', '2021-11-11',
                   '2021-11-25', '2021-12-24', '2021-12-31'],
                  dtype='datetime64[ns]', freq=None)



### Creating a holidays column


```python
dataset['holiday'] = dataset['# Date'].isin(holidays)
```


```python
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># Date</th>
      <th>Receipt_Count</th>
      <th>weekday</th>
      <th>month</th>
      <th>holiday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-01</td>
      <td>7564766</td>
      <td>4</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-02</td>
      <td>7455524</td>
      <td>5</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-03</td>
      <td>7095414</td>
      <td>6</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-04</td>
      <td>7666163</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-05</td>
      <td>7771289</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### Creating columns for days from next holiday and days from prev holiday


```python
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
```


```python
dataset['days_previous_holiday']= dataset.apply(lambda row: days_prev_holiday((row['# Date']), holidays), axis=1)
dataset['days_next_holiday']= dataset.apply(lambda row: days_next_holiday((row['# Date']), holidays), axis=1)
```


```python

dataset['holiday'] = np.array(encoder.fit_transform(dataset['holiday']))
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># Date</th>
      <th>Receipt_Count</th>
      <th>weekday</th>
      <th>month</th>
      <th>holiday</th>
      <th>days_previous_holiday</th>
      <th>days_next_holiday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-01</td>
      <td>7564766</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-02</td>
      <td>7455524</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-03</td>
      <td>7095414</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-04</td>
      <td>7666163</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-05</td>
      <td>7771289</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>



### Encoding the data using OneHotEncoder


```python
enc=OneHotEncoder()
 

enc_data=pd.DataFrame(enc.fit_transform(dataset[['weekday']]).toarray())
 
df=dataset.join(enc_data)
df = df.rename(columns={0: 'M', 1: 'Tu', 2: 'W', 3: 'Th', 4: 'F', 5: 'Sa', 6: 'Su'})

enc_data=pd.DataFrame(enc.fit_transform(df[['month']]).toarray())
 
df=df.join(enc_data)
```


```python
df['# Date']=df['# Date'].map(datetime.datetime.toordinal)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th># Date</th>
      <th>Receipt_Count</th>
      <th>weekday</th>
      <th>month</th>
      <th>holiday</th>
      <th>days_previous_holiday</th>
      <th>days_next_holiday</th>
      <th>M</th>
      <th>Tu</th>
      <th>W</th>
      <th>...</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>737791</td>
      <td>7564766</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>737792</td>
      <td>7455524</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>16</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>737793</td>
      <td>7095414</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>737794</td>
      <td>7666163</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>14</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>737795</td>
      <td>7771289</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>13</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



### Splitting the data into testing and training sets


```python
from sklearn.model_selection import train_test_split
X = df[['# Date', 'holiday','days_previous_holiday', 'days_next_holiday','month']]
y = df['Receipt_Count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
```


```python
df.corr()['Receipt_Count']
```




    # Date                   0.960658
    Receipt_Count            1.000000
    weekday                 -0.005646
    month                    0.957785
    holiday                  0.033517
    days_previous_holiday   -0.160776
    days_next_holiday       -0.418415
    M                        0.010168
    Tu                      -0.017177
    W                        0.006578
    Th                       0.007219
    F                        0.000666
    Sa                       0.007838
    Su                      -0.015298
    0                       -0.464196
    1                       -0.357376
    2                       -0.314796
    3                       -0.180769
    4                       -0.131781
    5                       -0.052883
    6                        0.014503
    7                        0.129866
    8                        0.208832
    9                        0.281154
    10                       0.399651
    11                       0.457127
    Name: Receipt_Count, dtype: float64



### Scaling the X values for better prediction


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
```

### Creating the model class


```python
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

```

### Fit the model


```python
regressor = MultipleLinearRegression()
regressor.fit(X_train, y_train)
```

### Checking model accuracy using R^2


```python
y_pred = regressor.predict(X_test)
y_mean = np.mean(y_test)
ss_tot = np.sum((y_test - y_mean)**2)
ss_res = np.sum((y_test - y_pred)**2)
r2 = 1 - (ss_res / ss_tot)
print("R-squared:", r2)
```

    R-squared: 0.9245610773495255


### Storing the model to be used by the web app


```python
pickle.dump(regressor, open('model2.pkl','wb'))
```


```python

model = pickle.load(open('model2.pkl','rb'))
print(model.predict(np.array([[737791, 1, 0, 0, 4]])))
```

    [7599193.48061987]


### Visualizing the predicting against the real values


```python
plt.scatter(X_test["# Date"], y_test,color='red')
plt.scatter(X_test["# Date"], y_pred,color='blue')
```




    <matplotlib.collections.PathCollection at 0x7f87ef75fac0>




    
![png](output_41_1.png)
    





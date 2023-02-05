
import flask
import pickle
import holidays
import pandas as pd
import numpy as np
from pandas import Timestamp
import datetime



# redeclaring for pickle
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


model = pickle.load(open('model4.pkl','rb'))



app = flask.Flask(__name__, template_folder='static')


@app.route('/', methods=['GET', 'POST '])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main2.html'))
    
    

@app.route('/predict', methods=['POST'])
def predict():
    # Get the date from the request
    date = flask.request.form['date']
    orig_date = date
    date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    month = date.month
  
    holiday = 0
    for ptr in holidays.US(years = date.year).items():
        print(ptr[0])
        if ptr[0] == date:
            holiday = 1

    difference=[]
    

    for ptr in holidays.US(years = date.year).items():
        difference.append(int((ptr[0]-date).days))
    next_hol = abs(min([x for x in difference if x>=0]))
    
    print(holiday, next_hol)

    difference=[]
    
    for ptr in holidays.US(years = date.year).items():
        difference.append(int((ptr[0]-date).days))
    prev_hol = abs(max([x for x in difference if x<=0]))
    date = date.toordinal()
        
    input_variables = np.array([[date, holiday, prev_hol, next_hol, float(month)]])
    prediction = model.predict(input_variables)[0]
    return flask.render_template('main.html',
                                original_input={'Date':orig_date},
                                result=prediction,
                                )



    
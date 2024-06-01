import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from flask import Flask,redirect,url_for,render_template,request
app=Flask(__name__)
data=pd.read_csv('Linear_regression//Salary_Data.csv')
x=data[['YearsExperience']]
y=data['Salary']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/')
def first():
    return render_template('index.html')
@app.route('/submit',methods=['POST','GET'])
def submit():
    if request.method=='POST':
        exp=float(request.form['exp'])
        prediction=model.predict([[exp]])
        return render_template('index.html',exp=exp,prediction=prediction[0])
if __name__=='__main__':
    app.run(debug=True)
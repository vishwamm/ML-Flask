import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from flask import Flask,redirect,url_for,render_template,request
app=Flask(__name__)
data=pd.read_csv("KNN//regressiondataset.csv")
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=KNeighborsRegressor(n_neighbors=5, metric='euclidean')
model.fit(x_train,y_train)
@app.route('/')
def first():
    return render_template('index1.html')
@app.route('/submit',methods=['POST','GET'])
def submit():
    if request.method=='POST':
        age=float(request.form['age'])
        exp=float(request.form['exp'])
        prediction=model.predict([[age,exp]])
        return render_template('index1.html',prediction=prediction[0],age=age,exp=exp)
if __name__=='__main__':
    app.run(debug=True)
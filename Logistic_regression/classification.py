import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from flask import Flask,redirect,url_for,render_template,request
app=Flask(__name__)
data=pd.read_csv("Logistic_regression//diabetes2.csv")
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression(max_iter=100) #for bionomial class
#for multinomial class
#model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=100)
#model.fit(x_train, y_train)
model.fit(x_train,y_train)
@app.route('/')
def first():
    return render_template('index.html')
@app.route('/submit',methods=['POST','GET'])
def submit():
    if request.method=='POST':
        preg=float(request.form['preg'])
        glu=float(request.form['glu'])
        blood=float(request.form['blood'])
        skin=float(request.form['skin'])
        ins=float(request.form['ins'])
        bmi=float(request.form['bmi'])
        dia=float(request.form['dia'])
        age=float(request.form['age'])
        prediction=model.predict([[preg,glu,blood,skin,ins,bmi,dia,age]])
        return render_template('index.html',glu=glu,age=age,prediction=prediction[0])
if __name__=='__main__':
    app.run(debug=True)
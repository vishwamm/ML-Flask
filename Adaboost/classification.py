import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier  
from flask import Flask,redirect,url_for,render_template,request
app=Flask(__name__)
df=pd.read_csv("Random_forest//car_evaluation.csv")
df=df.tail(100)
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names
x = df.drop(['class'], axis=1)
y = df['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)
encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)
model= AdaBoostClassifier() 
model.fit(x_train,y_train)
@app.route('/')
def first():
    return render_template('index.html')
@app.route('/submit',methods=['POST','GET'])
def submit():
    if request.method=='POST':
        buy=float(request.form['buy'])
        maint=float(request.form['maint'])
        doors=float(request.form['doors'])
        per=float(request.form['per'])
        lug=float(request.form['lug'])
        safe=float(request.form['safe'])
        prediction=model.predict([[buy,maint,doors,per,lug,safe]])
        return render_template('index.html',buy=buy,prediction=prediction[0])
if __name__=='__main__':
    app.run(debug=True)
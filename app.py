#Importing the libraries
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

df=pd.read_csv('heart.csv')
x=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train=scaler.fit_transform(x_train)

#Global variables
app = Flask(__name__)
model_Logres = pickle.load(open("H_attack_Logres.pkl", "rb"))
model_gnd = pickle.load(open("H_attack_gnd.pkl", "rb"))
model_knn = pickle.load(open("H_attack_knn.pkl", "rb"))


#User-defined functions
@app.route("/", methods=["GET"])
def Home():
    return render_template("index.html")

@app.route("/prediction",methods=['POST'])
def prediction():
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trestbps = int(request.form['trestbps'])
    chol = int(request.form['chol'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalach = int(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = int(request.form['slope'])
    ca = int(request.form['ca'])
    thal = int(request.form['thal'])
    lst=[age, sex, cp, trestbps, chol, fbs, restecg, thalach,exang, oldpeak, slope, ca, thal]
    lst=np.array(lst).reshape(1,-1)
    ans=model_knn.predict(scaler.transform(lst))[0]

    if(ans==0):ans="This person has less chances of a heart attack"
    else:ans="This Person has high chances of a heart attack"

    return render_template("index.html",prediction_output=ans)



#Main function
if __name__ == "__main__":app.run(debug=True)
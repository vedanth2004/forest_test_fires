from flask import Flask,Request,jsonify,render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

ridge_model=pickle.load(open('models/ridge.pkl','rb'))
scaler=pickle.load(open('models/scaler.pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict",methods=["GET","POST"])
def predict():
    if Request.method=="POST":
        Temperature=float(Request.form.get('Temperature'))
        RH=float(Request.form.get('RH'))
        Ws=float(Request.form.get('Ws'))
        Rain=float(Request.form.get('Rain'))
        FFMC=float(Request.form.get('FFMC'))
        DMC=float(Request.form.get('DMC'))
        ISI=float(Request.form.get('ISI'))
        Classes=float(Request.form.get('Classes'))
        Region=float(Request.form.get('Region'))

        # Scale the input data
        data=np.array([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        scaled_data=scaler.transform(data)
        
        # Make prediction
        result=ridge_model.predict(scaled_data)[0]
        
        return render_template("home.html",result=result)
    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(host="0.0.0.0")
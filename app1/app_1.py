# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 22:16:45 2021

@author: Norwin
"""

    # -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 19:10:11 2021

@author: Norwin
"""



from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("classifier_1.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    
    """Bank Default Prediction
    Poc for MLOPS pipeline
    ---
    parameters:  
      - name: Employed
        in: query
        type: number
        required: true
      - name: Bank_Balance
        in: query
        type: number
        required: true
      - name: Annual_Salary
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    Employed=request.args.get("Employed")
    Bank_Balance=request.args.get("Bank_Balance")
    Annual_Salary=request.args.get("Annual_Salary")
    prediction=classifier.predict([[Employed,Bank_Balance,Annual_Salary]])
    print(prediction)
    return "predicted value is"+str(prediction)



if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)
    


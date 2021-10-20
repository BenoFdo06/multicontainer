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

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Banks Default  Prediction
    POC for MLOPS pipeline
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=classifier.predict(df_test)
    
    return str(list(prediction))

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5001)
    



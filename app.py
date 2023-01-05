from flask import Flask, send_file, render_template
from distutils.log import debug
from fileinput import filename
from flask import *  
import pickle
import pandas as pd
import io
import os
import numpy as np

  
app = Flask(__name__)

@app.route('/')  
def main():  
    return render_template("index.html")  
  
@app.route('/success', methods = ['POST'])
def success(): 
    if request.method == 'POST':  
        f = request.files['file']
        f.save(f.filename)
        
        
        with open('model_pkl', 'rb') as fe:
                loaded_model = pickle.load(fe)
       
        

        df = pd.read_csv(f.filename)

        # print(df.head)
        #df = df.to_numpy()
        y_pred = loaded_model.predict(df)
        ypred = pd.DataFrame(y_pred)
        ypred.to_csv('generated.csv')
        # actual_mean = pd.DataFrame(y_test.mean(axis=0))
        pred_mean = pd.DataFrame(y_pred.mean(axis=0))

        # act=actual_mean.values.flatten()
        pred=pred_mean.values.flatten()
 

        
        # #df = df.to_numpy()
        # y_pred = loaded_model.predict(df)
        # # actual_mean = pd.DataFrame(y_test.mean(axis=0))
        # pred_mean = pd.DataFrame(y_pred.mean(axis=0))

        # # act=actual_mean.values.flatten()
        # pred_1=pred_mean.values
        # pred_1.to_csv('result.csv')
        # pred = pred1.flatten()
        # print(pred_1)

        



        return render_template("Acknowledgement.html", name = f.filename, prediction = pred, csv_file="generated.csv")  
  
@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':  
    app.run(debug=True)

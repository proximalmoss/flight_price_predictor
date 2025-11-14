from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('website.html')
@app.route('/predict', methods=['POST'])
def predict_datapoint():
    try:
        data= CustomData(
            airline=request.form.get('airline'),
            source_city=request.form.get('source_city'),
            destination_city=request.form.get('destination_city'),
            departure_time=request.form.get('departure_time'),
            arrival_time=request.form.get('arrival_time'),
            stops=request.form.get('stops'),
            class_type=request.form.get('class'),
            duration=float(request.form.get('duration')),
            days_left=int(request.form.get('days_left'))
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)

        return jsonify({'price':int(results[0])})
    
    except Exception as e:
        return jsonify({'error':str(e)}), 400
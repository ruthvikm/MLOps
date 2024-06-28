import json
import numpy as np
import joblib
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('azureml_quirky_card_4wsv0mt3v2_output_mlflow_log_model_1400287630')
    model = joblib.load(model_path)

def run(data):
    try:
        data = json.loads(data)
        data = np.array(data['data'])
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error

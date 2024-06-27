import json
import pandas as pd
from azureml.core.model import Model
import joblib


def init():
    global model
    model_path = Model.get_model_path('diabetes_model')
    model = joblib.load(model_path)


def run(data):
    try:
        data = json.loads(data)['data']
        data = pd.DataFrame.from_dict(data)
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})

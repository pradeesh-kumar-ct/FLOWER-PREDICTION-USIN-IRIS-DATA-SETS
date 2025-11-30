from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)
with open('iris.pkl', 'rb') as f:
    model= pickle.load(f)
@app.route('/')
def index():
    return jsonify({
        'message': 'Welcome to my API!',
        'version': '1.0',
        'endpoint':{
            '/':"this page API_info",
        '/predict': 'POST to get an method to predict',
        '/health':'Check API health'
        }
    })
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':'healthy',
        'model':'loaded'
    })
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data=request.get_json()
        features=np.array(data['features']).reshape(1,-1)
        predictions=model.predict(features)
        probability=model.predict_proba(features)

        class_names=['setosa','virginica','versicolor']

        return jsonify({
            'prediction':class_names[predictions[0]],
            'probability':{class_names[i]:float(probability[0][i])
            for i in range(3)
            }
        })
    except Exception as e:
        return jsonify({'error':str(e)}),400
if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)
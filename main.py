# from flask import Flask, request, jsonify
# from predict import predict_iris
# import os

# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()  # Get data as JSON
#     sepal_length = float(data['sepal_length'])
#     sepal_width = float(data['sepal_width'])
#     petal_length = float(data['petal_length'])
#     petal_width = float(data['petal_width'])

#     print(sepal_length, sepal_width, petal_length, petal_width)

#     prediction = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
#     return jsonify({'prediction': prediction})

# if __name__ == '__main__':
#     app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

# modifications
# main.py (in project root)
from flask import Flask, request, jsonify
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predict import predict_wine, FEATURE_NAMES

app = Flask(__name__)

@app.route('/')
def health():
    return "Wine API is live! POST to /predict", 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        for col in FEATURE_NAMES:
            if col not in data:
                return jsonify({'error': f'Missing {col}'}), 400
        prediction = predict_wine(**data)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
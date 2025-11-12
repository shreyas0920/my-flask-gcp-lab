# import numpy as np
# import joblib
# import os
# from train import run_training

# # Load the trained model
# model = joblib.load("model/model.pkl")

# def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
#     input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
#     prediction = model.predict(input_data)
#     return prediction[0]

# if __name__ == "__main__":
#     if os.path.exists("model/model.pkl"):
#         print("Model loaded successfully")
#     else:
#         os.makedirs("model", exist_ok=True)
#         run_training()

# modifications


import joblib
import os
import numpy as np

# Load from project root/model/
root_dir = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(root_dir, "model", "wine_model.pkl")
scaler_path = os.path.join(root_dir, "model", "wine_scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

FEATURE_NAMES = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
]

def predict_wine(**features):
    input_data = np.array([[features[col] for col in FEATURE_NAMES]])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    return int(pred)
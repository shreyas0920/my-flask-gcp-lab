# import joblib
# import os
# import pandas as pd 
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split

# def run_training(): 
#     """
#     Train the model
#     """
#     # FIXED PATH: Works no matter where you run from
#     csv_path = os.path.join(os.path.dirname(__file__), 'data', 'IRIS.csv')
#     dataset = pd.read_csv(csv_path)

#     X = dataset.drop("species", axis=1).copy()
#     y = dataset["species"].copy()

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.25, random_state=26)

#     model = LogisticRegression(random_state=26, max_iter=200)
#     model.fit(X_train, y_train)
#     model.feature_names = X.columns

#     os.makedirs("../model", exist_ok=True)
#     joblib.dump(model, "../model/model.pkl")
#     print("Model trained and saved to ../model/model.pkl")

# if __name__ == "__main__":
#     run_training()

# modifications

import joblib
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_training():
    # DEBUG: Print current location
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {os.path.dirname(__file__)}")

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'wine.csv')
    print(f"Loading data from: {data_path}")
    dataset = pd.read_csv(data_path)
    
    X = dataset.drop("target", axis=1)
    y = dataset["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=5000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # SAVE TO PROJECT ROOT/model/
    root_dir = os.path.dirname(os.path.dirname(__file__))  # Go up from src/
    model_dir = os.path.join(root_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "wine_model.pkl")
    scaler_path = os.path.join(model_dir, "wine_scaler.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"MODEL SAVED TO: {model_path}")
    print(f"SCALER SAVED TO: {scaler_path}")

if __name__ == "__main__":
    run_training()
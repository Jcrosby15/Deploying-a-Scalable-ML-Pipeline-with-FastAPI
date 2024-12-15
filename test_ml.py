import os
import pandas as pd
from sklearn.model_selection import KFold
import sklearn
import numpy as np
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    train_model
)

def test_model():
    """
    test model used for training the data
    """
    project_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_path, "data", "census.csv")
    #print(data_path)
    data = pd.read_csv(data_path)

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state = 42)
    for tr,te in kf.split(data):
        train, test = data.iloc[tr], data.iloc[te]

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train, 
        categorical_features= cat_features,
        label="salary",
     training= True
    )

    model = train_model(X_train,y_train)
    assert type(model) == sklearn.ensemble._forest.RandomForestClassifier
    



def test_valid_metrics():
    """
    test metrics returned are valid
    """
    project_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_path, "data", "census.csv")
    #print(data_path)
    data = pd.read_csv(data_path)

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state = 42)
    for tr,te in kf.split(data):
        train, test = data.iloc[tr], data.iloc[te]

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train, 
        categorical_features= cat_features,
        label="salary",
     training= True
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    metrics = compute_model_metrics(y_train, preds)
    for m in metrics:
        assert m >= 0 and m <= 1

def test_inference():
    """
    test whether inference values are all True False (0,1)
    """
    project_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_path, "data", "census.csv")
    #print(data_path)
    data = pd.read_csv(data_path)

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state = 42)
    for tr,te in kf.split(data):
        train, test = data.iloc[tr], data.iloc[te]

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train, 
        categorical_features= cat_features,
        label="salary",
     training= True
    )
    model = train_model(X_train, y_train)
    inf = inference(model, X_train)
    assert np.all((inf==0)|(inf==1)) == True

if __name__ == "__main__":
    test_model()
    test_valid_metrics()
    test_inference()

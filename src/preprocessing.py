import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

def preprocess_data(df, target='CH4', use_poly=False, poly_degree=2):

    df = df.copy()  # Don't modify original
    df = df.dropna(subset=[target])
    df = df.fillna(df.median())  # Don't use inplace
    
    X = df.drop(columns=[target])
    y = df[target].values
    feature_names = X.columns.tolist()
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    if use_poly:
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ('scaler', StandardScaler())
        ])
    else:
        pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])
    
    X_train_scaled = pipeline.fit_transform(X_train)
    X_val_scaled = pipeline.transform(X_val)
    X_test_scaled = pipeline.transform(X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'pipeline': pipeline,
        'feature_names': feature_names
    }

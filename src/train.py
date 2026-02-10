# train.py (at root level)
"""
Train methane production prediction models.

Usage:
    python train.py
"""
from src.preprocessing import preprocess_data
from src.models import train_gam, train_rf, train_knn, MeanPredictor
from src.evaluation import evaluate
from src.s3_utils import load_csv_from_s3


def main():
    # Load data
    print("Loading data...")
    df = load_csv_from_s3(bucket_name='my-bucket', key='methane_data.csv')
    
    # Preprocess
    print("Preprocessing...")
    X_train, X_val, X_test, y_train, y_val, y_test, pipeline = preprocess_data(df)
    
    # Train models
    print("Training models...")
    models = {
        'baseline': MeanPredictor(),
        'gam': None,
        'rf': None,
        'knn': None
    }
    
    models['baseline'].fit(X_train, y_train)
    models['gam'] = train_gam(X_train, y_train)
    models['rf'] = train_rf(X_train, y_train)
    models['knn'] = train_knn(X_train, y_train)
    
    # Evaluate
    print("\nResults on validation set:")
    print("-" * 40)
    for name, model in models.items():
        y_pred = model.predict(X_val)
        metrics = evaluate(y_val, y_pred)
        print(f"{name:12} | MAE: {metrics['MAE']:.2f} | R²: {metrics['R2']:.3f}")


if __name__ == "__main__":
    main()

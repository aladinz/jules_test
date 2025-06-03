import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Assuming feature_engineering and data_ingestion are accessible.
# Adjust PYTHONPATH or use relative imports if running as part of a larger package structure.
try:
    from machine_learning.feature_engineering import prepare_features_for_ml
    from data_ingestion.stock_data import get_historical_data
except ImportError:
    print("Warning: Could not import custom modules for ML model testing. Standalone functionality might be limited.")
    # Define dummy functions or classes if needed for basic syntax checking if imports fail
    def get_historical_data(ticker, start, end): return pd.DataFrame()
    def prepare_features_for_ml(df, **kwargs): return pd.DataFrame(), pd.Series()


def train_model(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> LogisticRegression:
    """
    Trains a Logistic Regression model.

    Args:
        X (pd.DataFrame): DataFrame of features.
        y (pd.Series): Series of target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        LogisticRegression: The trained Logistic Regression model.
                            Returns None if training fails (e.g., insufficient data).
    """
    if X.empty or y.empty:
        print("Error: Input features (X) or target (y) are empty. Cannot train model.")
        return None

    if len(X) != len(y):
        print(f"Error: X and y have mismatched lengths: {len(X)} vs {len(y)}. Cannot train model.")
        return None

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if sum(y) >=2 else None)

        if len(X_train) < 1 or len(X_test) < 1:
             print(f"Warning: Not enough samples to train or test after split. X_train: {len(X_train)}, X_test: {len(X_test)}")
             # Fallback: Train on all data if split results in empty sets (though less ideal for evaluation)
             X_train, y_train = X, y # Use all data for training
             X_test, y_test = X, y # And for testing (will give inflated accuracy)


        model = LogisticRegression(solver='liblinear', random_state=random_state)
        model.fit(X_train, y_train)

        # Evaluate on the test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained. Test Set Accuracy: {accuracy:.4f}")

        return model

    except ValueError as e:
        print(f"Error during model training or evaluation: {e}")
        print("This can happen if data is too small for stratification or other issues.")
        # Attempt to train without stratification if that was the issue.
        try:
            print("Attempting to train without stratification...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            model = LogisticRegression(solver='liblinear', random_state=random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model trained (without stratification). Test Set Accuracy: {accuracy:.4f}")
            return model
        except Exception as final_e:
            print(f"Error during model training fallback: {final_e}")
            return None
    except Exception as e:
        print(f"An unexpected error occurred during model training: {e}")
        return None

def make_prediction(model: LogisticRegression, X_latest: pd.DataFrame) -> tuple[int | None, float | None]:
    """
    Makes a prediction using the trained model for the latest features.

    Args:
        model (LogisticRegression): The trained scikit-learn model.
        X_latest (pd.DataFrame): DataFrame containing the most recent features
                                 (should have the same columns as training X).
                                 Expected to be a single row.

    Returns:
        tuple[int | None, float | None]:
            - Prediction (0 or 1), or None if prediction fails.
            - Probability of the prediction being class 1, or None if fails.
    """
    if model is None:
        print("Error: Model is not trained or provided.")
        return None, None

    if not isinstance(X_latest, pd.DataFrame):
        print("Error: X_latest must be a pandas DataFrame.")
        return None, None

    if X_latest.shape[0] != 1:
        print(f"Warning: X_latest contains {X_latest.shape[0]} rows. Prediction is typically for a single instance. Using first row.")
        X_latest = X_latest.head(1)

    # Ensure columns match training data columns (order and names)
    # This is a common source of errors if not handled.
    # We assume X_latest has been prepared with the same feature engineering pipeline.
    # A more robust check would involve passing expected column names from training.
    # For now, we rely on the caller to ensure column consistency.

    try:
        prediction = model.predict(X_latest)[0]
        probability = model.predict_proba(X_latest)[0, 1] # Probability of class 1
        return int(prediction), float(probability)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

if __name__ == '__main__':
    print("Testing ML Model Training and Prediction...")
    # Fetch sample data
    # Using a different ticker/dates to avoid conflicts if app.py is running or caching issues.
    ticker_model_test = "GOOGL"
    start_date_model_test = "2021-01-01"
    end_date_model_test = "2023-01-01" # 2 years of data

    raw_data_mt = pd.DataFrame()
    # Check if the imported functions are the dummy ones or the real ones
    is_dummy_get_historical_data = get_historical_data.__module__ == __name__

    if not is_dummy_get_historical_data:
        raw_data_mt = get_historical_data(ticker_model_test, start_date_model_test, end_date_model_test)

    if raw_data_mt.empty:
        print(f"Could not fetch data for {ticker_model_test} or using dummy function. Using fallback data for model training test.")
        num_days_fallback_mt = 200 # More data for robust training
        dates_fallback_mt = pd.to_datetime([pd.Timestamp('2022-01-01') + pd.Timedelta(days=i) for i in range(num_days_fallback_mt)])
        prices_mt = [100 + (i/10) + (i%10 - 5) * 0.2 + (i%3) for i in range(num_days_fallback_mt)] # Some trend and noise, and binary signal
        raw_data_mt = pd.DataFrame({'Close': prices_mt}, index=dates_fallback_mt)
        for col in ['Open', 'High', 'Low', 'Volume']: # Add other necessary columns
            raw_data_mt[col] = prices_mt if col != 'Volume' else 10000

    is_dummy_prepare_features = prepare_features_for_ml.__module__ == __name__
    if not raw_data_mt.empty and not is_dummy_prepare_features:
        # For testing make_prediction, we need to ensure X_latest is not part of training data.
        # One way: train on data up to T-1, predict for T.
        # Or, simply use the last row of X_sample if shuffle=False in train_test_split, but default is True.
        # For simplicity in this example, we'll train, then take the true last row of X features
        # (which might have been part of test or train set).
        # A more rigorous approach would be to set aside the last day's data *before* splitting.

        X_sample_full, y_sample_full = prepare_features_for_ml(raw_data_mt, sma_window=10, rsi_window=14, num_lags=5)

        if not X_sample_full.empty and not y_sample_full.empty:
            print(f"Full Features (X_sample_full shape): {X_sample_full.shape}, Full Target (y_sample_full shape): {y_sample_full.shape}")

            # Take the last row of features for later prediction *before* splitting
            # Ensure X_latest_example is a DataFrame with the same columns as X_sample_full
            if len(X_sample_full) > 1:
                X_latest_example = X_sample_full.iloc[[-1]].copy()
                # Train on data excluding the very last sample (if we want to be strict, not strictly necessary for this test)
                # X_train_data = X_sample_full.iloc[:-1]
                # y_train_data = y_sample_full.iloc[:-1]
                # For this example, we'll just train on all available X_sample_full, y_sample_full for simplicity of test setup
                X_train_data, y_train_data = X_sample_full, y_sample_full
            else:
                print("Not enough data to create X_latest_example. Skipping prediction test.")
                X_latest_example = None


            if len(y_train_data.value_counts()) < 2 and len(y_train_data) > 0 : # Check if y_sample has enough variety for stratification
                print(f"Warning: Target variable y_train_data has only one class: {y_train_data.value_counts()}. Stratification might fail.")

            trained_logistic_model = train_model(X_train_data, y_train_data)

            if trained_logistic_model:
                print("Logistic Regression model trained successfully.")

                if X_latest_example is not None:
                    print(f"\nLatest features for prediction (X_latest_example):\n{X_latest_example}")
                    prediction, probability = make_prediction(trained_logistic_model, X_latest_example)
                    if prediction is not None:
                         print(f"\nPrediction for latest features: {prediction} (0: Down/Same, 1: Up)")
                         print(f"Probability of price going up (class 1): {probability:.4f}")
                    else:
                        print("Failed to make a prediction.")
                else:
                    print("Skipping prediction as X_latest_example could not be created.")
            else:
                print("Failed to train Logistic Regression model.")
        else:
            print("Sample X or y is empty after feature engineering. Cannot train model.")
    else:
        print("Could not prepare data for model training test due to missing functions, empty raw data, or using dummy functions.")

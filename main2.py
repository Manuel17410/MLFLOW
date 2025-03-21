import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Load dataset
file_path = r"C:\Users\manue\Desktop\DataScience\Datasets\clean_laptop_data2.csv"
df = pd.read_csv(file_path)

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
numerical_cols.remove('Price')  # Remove target variable

# Convert numerical columns to float64
df[numerical_cols] = df[numerical_cols].astype('float64')

# Define features and target
X = df.drop(columns=['Price'])
y = np.log1p(df['Price'])  # Apply log transformation to target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow Setup
mlflow.set_tracking_uri("sqlite:///C:/Users/manue/Desktop/DataScience/Laptop_2/mlflow_new.db")
mlflow.set_experiment("Laptop_Price_Prediction")

# Models to evaluate
models = {
    "RandomForestRegressor": RandomForestRegressor,
    "XGBRegressor": XGBRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor
}

# Hyperparameter grid
param_grid = [
    {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2},
    {"n_estimators": 200, "max_depth": 20, "min_samples_split": 5},
    {"n_estimators": 300, "max_depth": 30, "min_samples_split": 10}
]


R2_THRESHOLD = 0.83  # R² validation threshold

# Iterate over models and hyperparameters
for model_name, model_class in models.items():
    for params in param_grid:
        with mlflow.start_run():
            mlflow.set_tag("Model", model_name)
            mlflow.set_tag("Author", "Manuel Contreras")
            mlflow.set_tag("Dataset", "Clean_Laptop_Prices")

            # Define preprocessing
            preprocessor = ColumnTransformer([
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
            ])

            # Define pipeline
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model_class(**params, random_state=42))
            ])

            # Train model
            pipeline.fit(X_train, y_train)

            # Predict and revert log transformation
            y_pred_log = pipeline.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            y_test_original = np.expm1(y_test)

            # Evaluate model
            mse = mean_squared_error(y_test_original, y_pred)
            r2 = r2_score(y_test_original, y_pred)
            mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100
            rmspe = np.sqrt(np.mean(((y_test_original - y_pred) / y_test_original) ** 2)) * 100

            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("R2 Score", r2)
            mlflow.log_metric("MAPE", mape)
            mlflow.log_metric("RMSPE", rmspe)

            # Log model
            signature = infer_signature(X_train, pipeline.predict(X_train))
            mlflow.sklearn.log_model(pipeline, model_name, signature=signature, input_example=X_test.iloc[:5])

            # Skip run if R² is below threshold
            if r2 < R2_THRESHOLD:
                print(f"Skipping {model_name} with params {params} (R²: {r2:.4f})")
                mlflow.end_run(status="KILLED")
                continue

            print(f"{model_name} - Params: {params} -> R²: {r2:.4f}, MSE: {mse:.4f}, MAPE: {mape:.2f}%, RMSPE: {rmspe:.2f}%")

            # Plot feature importance (if applicable)
            if model_name in ["RandomForestRegressor", "XGBRegressor"]:
                feature_names = numerical_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
                importances = pipeline.named_steps["model"].feature_importances_

                plt.figure(figsize=(10, 6))
                sns.barplot(x=importances, y=feature_names, palette="viridis", hue=feature_names, legend=False)
                plt.xlabel("Feature Importance")
                plt.ylabel("Features")
                plt.title(f"{model_name} Feature Importance")
                plt.tight_layout()
                feature_importance_path = f"mlflow_artifacts/{model_name}_feature_importance.png"
                plt.savefig(feature_importance_path)
                plt.close()
                mlflow.log_artifact(feature_importance_path)

            # Save residual plot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=y_test_original, y=y_pred, alpha=0.6)
            plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], linestyle="--", color="red")
            plt.xlabel("Actual Price")
            plt.ylabel("Predicted Price")
            plt.title(f"{model_name} - Actual vs. Predicted")
            plt.tight_layout()
            residual_plot_path = f"mlflow_artifacts/{model_name}_residual_plot.png"
            plt.savefig(residual_plot_path)
            plt.close()
            mlflow.log_artifact(residual_plot_path)

            # Error histogram
            errors = y_test_original - y_pred
            plt.figure(figsize=(8, 6))
            sns.histplot(errors, bins=30, kde=True, color="blue")
            plt.xlabel("Prediction Error")
            plt.ylabel("Frequency")
            plt.title(f"{model_name} Error Distribution")
            plt.tight_layout()
            error_hist_path = f"mlflow_artifacts/{model_name}_error_distribution.png"
            plt.savefig(error_hist_path)
            plt.close()
            mlflow.log_artifact(error_hist_path)

            # Ensure the directory exists
            feature_importance_path = "mlflow_artifacts/RandomForestRegressor_feature_importance.png"
            os.makedirs(os.path.dirname(feature_importance_path), exist_ok=True)
            plt.savefig(feature_importance_path)



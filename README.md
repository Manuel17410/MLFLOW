# MLflow Powered Laptop Price Prediction Regression Pipeline

### Project Overview:

The process began with cleaning the data and performing feature engineering to extract more insightful information from the variables. This involved breaking down variables into numerical and categorical components and changing data types accordingly. Next, a pipeline to one-hot encode categorical variables and standard scale the numerical ones was built in order to prepare the data for modeling.

Once the data was ready, the four main components of **MLflow** were used to manage and streamline the machine learning workflow:

1. **MLflow Tracking** was used to log and track various experiments, including model parameters, metrics, and model performance, ensuring we could monitor the progress of each trial.
2. **MLflow Projects** allowed us to package the entire codebase in a reusable and reproducible format by defining the required environment and dependencies through the `MLproject` file.
3. **MLflow Models** enables to manage and deploy models efficiently, supporting different model formats and allowing us to log, save, and load models for future use.
4. **MLflow Registry** served as a centralized repository for tracking the model lifecycle, including versioning and the management of stage transitions (e.g., from staging to production).

Additionally, the **MLflow UI** was employed to compare various runs and register the best model based on performance. An experiment was conducted using different regression models: **RandomForestRegressor**, **XGBRegressor**, and **GradientBoostRegressor**, having a total of 9 runs. The model with the highest R² score was selected, and a pickle file of the best model was created for future use.

This approach ensured that the project was reproducible, well-documented, and ready for deployment.

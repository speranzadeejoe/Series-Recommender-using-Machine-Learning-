# Series-Recommender-using-Machine-Learning-
# Movie Recommendation System

This project implements a simple movie recommendation system using a linear regression model. It predicts movie ratings based on a dataset (assumed to contain features related to movies).

## Overview

The notebook performs the following steps:

1.  **Data Loading and Preparation:**
    * Loads the movie dataset (the specific loading method and dataset are not shown in the provided snippet).
    * Separates the data into features (`X`) and the target variable (`y`, assumed to be 'IMDB\_Rating').
    * Splits the data into training and testing sets.

2.  **Model Training:**
    * Initializes a linear regression model.
    * Trains the model using the training data.
    * Makes predictions on the test data.

3.  **Model Evaluation:**
    * Evaluates the model's performance using Mean Squared Error (MSE) and R-squared score on the test set.
    * Prints the calculated MSE and R-squared values.

4.  **Full Dataset Processing for Prediction:**
    * Encodes categorical features (the specific encoding method is not fully shown but involves a `transformer` and `OneHotEncoder`).
    * Scales numerical features using a `StandardScaler`.
    * Combines the processed categorical and numerical features.

5.  **Rating Prediction on Full Dataset:**
    * Uses the trained linear regression model to predict ratings for all movies in the full dataset.

6.  **Generating Recommendations:**
    * Creates a new DataFrame containing the 'Series\_Title' and the predicted ratings.
    * Sorts this DataFrame in descending order based on the 'Predicted\_Rating'.
    * Drops duplicate 'Series\_Title' entries, keeping the one with the highest predicted rating.
    * Displays the top 10 recommended movies based on their predicted ratings.

## Libraries Used

* pandas
* scikit-learn (`sklearn`)
    * `model_selection.train_test_split`
    * `linear_model.LinearRegression`
    * `metrics.mean_squared_error`
    * `metrics.r2_score`
    * `compose.ColumnTransformer`
    * `preprocessing.OneHotEncoder`
    * `preprocessing.StandardScaler`

## Setup

1.  **Install required libraries:**
    ```bash
    pip install pandas scikit-learn
    ```

2.  **Obtain the dataset:**
    * Place your movie dataset file in the same directory as the notebook or provide the correct path in the data loading step (not shown in the snippet). The dataset should contain a column named 'IMDB\_Rating' which is used as the target variable, and a column named 'Series\_Title' for movie titles. It should also have categorical and numerical features that the model can learn from.

## Usage

1.  Open and run the provided Python notebook.
2.  The output will show:
    * Mean Squared Error and R-squared score of the model on the test set.
    * A table of the top 10 recommended movies with their predicted ratings.

## Potential Improvements

* **Explore different machine learning models:** Experiment with other regression algorithms (e.g., Random Forest Regressor, Gradient Boosting Regressor) to potentially improve prediction accuracy.
* **Feature Engineering:** Create new relevant features from the existing data to enhance the model's learning capability.
* **More sophisticated encoding:** Investigate other encoding techniques for categorical features, such as target encoding or embedding.
* **Hyperparameter tuning:** Optimize the hyperparameters of the chosen model using techniques like GridSearchCV or RandomizedSearchCV.
* **Collaborative filtering or content-based filtering:** For a more advanced recommendation system, consider implementing collaborative filtering or content-based filtering techniques.
* **Handling missing data:** Implement strategies to handle missing values in the dataset.
* **Evaluation metrics:** Explore other relevant evaluation metrics for regression tasks.

## Note

* This is a basic recommendation system based on predicting movie ratings. The quality of recommendations heavily depends on the features available in the dataset and the performance of the trained model.
* The specific data loading and preprocessing steps might need adjustments based on the actual structure and content of your movie dataset.

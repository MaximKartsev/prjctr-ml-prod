import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load the dataset
uploaded = files.upload()
train_df = pd.read_csv('train.csv')

# Split the training data into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(train_df['excerpt'], train_df['target'], test_size=0.2, random_state=42)

# Build a pipeline for text processing and regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('reg', SGDRegressor(random_state=42, max_iter=1000, tol=1e-3))
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred_test = pipeline.predict(X_test)

# Print evaluation metrics
mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f'Test Mean Squared Error: {mse}')
print(f'Test Mean Absolute Error: {mae}')
print(f'R^2 Score: {r2}')

# Save the trained model
joblib.dump(pipeline, 'trained_model.pkl')

# Save the comparison dataframe to a CSV file for further inspection
comparison_df = pd.DataFrame({
    'Excerpt': X_test,
    'Actual': y_test,
    'Predicted': y_pred_test
})
comparison_df.to_csv('test_predictions_comparison.csv', index=False)
print("Comparison CSV file created successfully.")
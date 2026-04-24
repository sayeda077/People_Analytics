# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
# Step 1: Load and clean dataset
df = pd.read_csv("C:/Users/syeda/OneDrive/Documents/Semester 3/People Analytics/wellbeing_of_employees.csv")  # Replace with your file path

# Convert TODO_COMPLETED to numeric, coerce errors to NaN
df['TODO_COMPLETED'] = pd.to_numeric(df['TODO_COMPLETED'], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

# Drop unnecessary column
df.drop(columns=['EMPLOYEE_ID'], inplace=True)

# Step 2: Feature-target separation
X = df.drop(columns=['WORK_LIFE_BALANCE_SCORE'])
y = df['WORK_LIFE_BALANCE_SCORE']

# Step 3: Identify categorical features
categorical_cols = ['JOB_ROLE', 'AGE', 'GENDER']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Step 4: Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(drop="first"), categorical_cols)
], remainder='passthrough')

# Step 5: Modeling pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])
# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 7: Fit the model
model.fit(X_train, y_train)

# Step 8: Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Step 9: Show real vs predicted in table
comparison_df = pd.DataFrame({
    "Real WLB": y_test.values,
    "Predicted WLB": np.round(y_pred, 2),
    "Difference": np.round(y_test.values - y_pred, 2)
}).reset_index(drop=True)

print("\nReal vs Predicted (Sample):")
print(comparison_df.head())

# Step 10: Predict new employee
new_employee = pd.DataFrame([{
    'JOB_ROLE': 'Train Conductor',
    'TEAM_SIZE': 6,
    'MONTHLY_EXTRA_HOURS': 5,
    'EXTRA_HOLIDAYS': 3,
    'TODO_COMPLETED': 7,
    'BMI_RANGE': 1,
    'DAILY_STEPS_IN_THOUSAND': 6,
    'SLEEP_HOURS': 7,
    'DAILY_STRESS': 3,
    'TIME_FOR_HOBBY': 4,
    'HEALTHY_MEALS_PER_WEEK': 7,
    'SUFFICIENT_INCOME': 2,
    'AGE': '21 to 35',
    'GENDER': 'Male'
}])

predicted_wlb = model.predict(new_employee)[0]
print(f"\nPredicted WLB Score for New Employee: {predicted_wlb:.2f}")

# Step 11: Scatter plot: Real vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Real WLB Score')
plt.ylabel('Predicted WLB Score')
plt.title('Real vs Predicted WLB Score')
plt.grid(True)
plt.tight_layout()
plt.savefig("Real vs Predicted WLB Score.png")
plt.show()

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("C:/Users/syeda/OneDrive/Documents/Semester 3/People Analytics/turnover.csv")

# Copy the data
df_encoded = df.copy()

# Encode 'job_role' and 'salary'
df_encoded['job_role'] = LabelEncoder().fit_transform(df_encoded['job_role'])
df_encoded['salary'] = df_encoded['salary'].map({'low': 0, 'medium': 1, 'high': 2})

# Define features and target
X = df_encoded.drop('left', axis=1)
y = df_encoded['left']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", conf_matrix)

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("Confusion Matrix.png")
plt.show()

# Feature Importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar')
plt.title("Feature Importance for Predicting Employee Turnover")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("Feature Importance for Predicting Employee Turnover.png") 
plt.show()

# Convert report to DataFrame if needed
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("classification_report.csv", index=True)
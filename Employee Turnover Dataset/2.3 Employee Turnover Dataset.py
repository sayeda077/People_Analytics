import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("C:/Users/syeda/OneDrive/Documents/Semester 3/People Analytics/turnover.csv")

# Step 1: Convert Categorical Data
# Map salary to numeric and job_role to numeric using mapping
df['salary'] = df['salary'].map({'low': 0, 'medium': 1, 'high': 2}) 
job_role_mapping = {role: idx for idx, role in enumerate(df['job_role'].unique())}
reverse_job_role_mapping = {v: k for k, v in job_role_mapping.items()}
df['job_role'] = df['job_role'].map(job_role_mapping)

# Step 2: Salary by Job Role
df_viz = df.copy()
df_viz['job_role'] = df_viz['job_role'].map(reverse_job_role_mapping)
df_viz['salary'] = df_viz['salary'].map({0: 'low', 1: 'medium', 2: 'high'})

# Plot grouped bar chart
plt.figure(figsize=(12, 6))
sns.countplot(data=df_viz, x='job_role', hue='salary', palette='coolwarm', order=sorted(df_viz['job_role'].unique()))
plt.title('Salary by Job Role')
plt.xlabel('Job Role')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Salary by Job Role.png")
plt.show()

# Step 3: Job Satisfaction of Employees Who Left
avg_satisfaction_left = df[df['left'] == 1]['job_satisfaction_rate'].mean()
print(f"Average Job Satisfaction of Employees Who Left: {avg_satisfaction_left:.2f}")

# Step 4: Tenure of Employees Who Left
avg_time_spent_left = df[df['left'] == 1]['time_spend_company'].mean()
print(f"Average Time Spent by Employees Who Left: {avg_time_spent_left:.2f} years")

# Step 5: Correlation Matrix
correlation_matrix = df.corr(numeric_only=True)
print("\nCorrelation with 'left':\n", correlation_matrix['left'].sort_values(ascending=False))

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig("Correlation Matrix.png")
plt.show()

# Step 6: Missing Value Check
missing_values = df.isnull().sum()
print("\nMissing Values in Dataset:")
print(missing_values)

# Step 7: View Final Preprocessed Dataset
print("\nPreprocessed Dataset Sample:")
print(df.head())
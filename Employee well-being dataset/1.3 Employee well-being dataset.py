
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("C:/Users/syeda/OneDrive/Documents/Semester 3/People Analytics/wellbeing_of_employees.csv")

# Data Preprocessing
data['DAILY_STRESS'] = pd.to_numeric(data['DAILY_STRESS'], errors='coerce')
data.dropna(subset=['DAILY_STRESS'], inplace=True)
data.drop(columns=['BMI_RANGE'], inplace=True)

# Mapping categorical to numeric
age_mapping = {"Under 20": 0, "21 to 35": 1, "36 to 50": 2, "51 or more": 3}
gender_mapping = {"Female": 0, "Male": 1}
data['AGE'] = data['AGE'].map(age_mapping)
data['GENDER'] = data['GENDER'].map(gender_mapping)

# Plot: Average Daily Stress by Gender
plt.figure(figsize=(6, 4))
sns.barplot(x='GENDER', y='DAILY_STRESS', data=data, palette='coolwarm')
plt.title('Average Daily Stress by Gender')
plt.xticks([0, 1], ['Female', 'Male'])
plt.xlabel('Gender')
plt.ylabel('Average Daily Stress')
plt.tight_layout()
plt.savefig("Average Daily Stress by Gender.png")
plt.show()

# Plot: Average Daily Stress by Age Group
plt.figure(figsize=(6, 4))
sns.barplot(x='AGE', y='DAILY_STRESS', data=data, palette='viridis')
plt.title('Average Daily Stress by Age Group')
plt.xticks(ticks=[0, 1, 2, 3], labels=["<20", "21-35", "36-50", "51+"])
plt.xlabel('Age Group')
plt.ylabel('Average Daily Stress')
plt.tight_layout()
plt.savefig("Average Daily Stress by Age Group.png")
plt.show()

# Plot: Daily Steps by Gender
plt.figure(figsize=(6, 4))
sns.barplot(x='GENDER', y='DAILY_STEPS_IN_THOUSAND', data=data, palette='Set2')
plt.title('Average Daily Steps by Gender')
plt.xticks([0, 1], ['Female', 'Male'])
plt.xlabel('Gender')
plt.ylabel('Average Daily Steps (in Thousands)')
plt.tight_layout()
plt.savefig("Daily Steps by Gender.png")
plt.show()

# Plot: Correlation Matrix with WLB Score
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix with WLB Score')
plt.tight_layout()
plt.savefig("Correlation Matrix with WLB Score.png")
plt.show()

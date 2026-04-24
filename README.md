# 📊 People Analytics

A data-driven project focused on analyzing **employee well-being** and **employee turnover**, and building machine learning models to **predict work-life balance (WLB)** and **employee attrition**.

---

## 📌 Project Overview

This project explores two key HR analytics domains:

### 1. Employee Well-Being Analysis
- Understand factors affecting work-life balance
- Analyze stress, lifestyle, and productivity metrics
- Build a predictive model for WLB score

### 2. Employee Turnover Analysis
- Identify reasons why employees leave
- Perform exploratory data analysis (EDA)
- Build a classification model to predict turnover

---

## 📂 Project Structure
People_Analytics/
│
├── Employee well-being dataset/
│   └── 1.3 Employee well-being dataset.py
│
├── Predictive well-being algorithm/
│   └── 1.4 Predictive well-being algorithm.py
│
├── Employee Turnover Dataset/
│   └── 2.3 Employee Turnover Dataset.py
│
├── Turnover prediction/
│   └── 2.4 Turnover prediction.py
│
├── Project Report - Sayeda Fatema Tuj Zohura.pdf
└── README.md


---

## 📊 Datasets

### Employee Well-Being Dataset
Includes features such as:
- Daily stress
- Sleep hours
- Daily steps
- Time for hobbies
- Healthy meals per week
- Monthly extra hours

Target:
- `WORK_LIFE_BALANCE_SCORE`

---

### Employee Turnover Dataset
Includes:
- Job satisfaction
- Salary level (low, medium, high)
- Time spent at company
- Job role
- Promotion history
- Number of colleague friends

Target:
- `left` (0 = Stayed, 1 = Left)

---

## Exploratory Data Analysis (EDA)

### Well-Being Insights
- Females report slightly higher daily stress than males
- Stress peaks in the **21–35 age group**
- Males take slightly more daily steps than females

### Key Correlations with Work-Life Balance
**Positive correlations:**
- Time for hobby (+0.52)
- Healthy meals per week (+0.51)
- Extra holidays (+0.46)
- Daily steps (+0.42)

**Negative correlation:**
- Daily stress (-0.37)

---

### Turnover Insights
- Employees who leave have **low job satisfaction (~0.44)**
- Average tenure before leaving: **~3.9 years**
- Strongest factor affecting turnover:
  - Job satisfaction (negative correlation: -0.39)

---

##  Machine Learning Models

### 1. Work-Life Balance Prediction

Model: **Linear Regression**

Pipeline:
- Data preprocessing
- One-hot encoding for categorical variables
- Train-test split

Performance:
- **R² Score ≈ 0.86**

Capabilities:
- Predict WLB score for employees
- Compare real vs predicted values

---

### 2. Employee Turnover Prediction

Model: **Random Forest Classifier**

Features used:
- Job satisfaction
- Monthly working hours
- Salary
- Job role
- Social and workplace factors

Performance:
- **Accuracy ≈ 98.67%**
- High precision, recall, and F1-score

---

##  Key Visualizations

- Correlation heatmaps
- Salary distribution by job role
- Average stress by gender and age
- Daily steps comparison
- Real vs predicted WLB scatter plot
- Confusion matrix
- Feature importance bar chart

---

##  Feature Importance (Turnover Model)

Most influential features:
1. Job satisfaction rate
2. Time spent at company
3. Number of colleague friends
4. Average monthly hours

Less influential:
- Salary
- Promotion
- Job role

---

##  Key Insights

### Employee Well-Being
- More hobby time leads to better work-life balance
- Healthy lifestyle improves employee well-being
- High stress significantly reduces WLB

### Employee Turnover
Employees are more likely to leave due to:
- Low job satisfaction
- Weak workplace relationships
- Long tenure without growth
- Imbalanced workload

---


##  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

##  How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/People_Analytics.git
cd People_Analytics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_students = 200
data = {
    'Midterm_Score': np.random.randint(0, 101, num_students),
    'Homework_Avg': np.random.randint(0, 101, num_students),
    'Attendance': np.random.randint(0, 101, num_students),
    'Prior_GPA': np.random.uniform(0, 4.0, num_students),
    'Is_First_Gen': np.random.choice([0, 1], num_students), # 0: No, 1: Yes
    'At_Risk': np.random.choice([0, 1], num_students, p=[0.8, 0.2]) # 80% not at risk, 20% at risk
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Preparation ---
# (In a real-world scenario, this would involve handling missing values, outliers, etc.)
# For this synthetic data, no cleaning is explicitly needed.
# --- 3. Predictive Modeling ---
X = df.drop('At_Risk', axis=1)
y = df['At_Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000) # Increased max_iter to ensure convergence
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# --- 4. Model Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
# --- 5. Visualization ---
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not at Risk', 'At Risk'],
            yticklabels=['Not at Risk', 'At Risk'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
# Save the plot to a file
output_filename = 'confusion_matrix.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
plt.figure(figsize=(10,6))
sns.countplot(x='At_Risk', data=df)
plt.title('Distribution of At-Risk Students')
plt.xlabel('At Risk (0: No, 1: Yes)')
plt.ylabel('Count')
plt.tight_layout()
output_filename2 = 'at_risk_distribution.png'
plt.savefig(output_filename2)
print(f"Plot saved to {output_filename2}")
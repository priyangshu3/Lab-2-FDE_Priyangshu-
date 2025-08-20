# DA.py - Data Analysis for Healthcare Data

import pandas as pd
import matplotlib.pyplot as plt

# Path
processed_file = "data_warehouse/processed_healthcare_data.csv"

# 1. Load processed data
df = pd.read_csv(processed_file)

# ✅ Step 1: Aggregate feedback at patient level (avoid duplication)
patient_feedback = (
    df.groupby('patient_id')['patient_feedback_score']
    .mean()
    .reset_index()
)

# Merge aggregated feedback back into df
df = df.drop(columns=['patient_feedback_score'], errors='ignore')
df = df.merge(patient_feedback, on='patient_id', how='left')

# 2. Analysis: Top doctors by revenue + avg patient feedback score
report = (
    df.groupby(['doctor_id', 'specialty'])
    .agg(total_revenue=('total_revenue','sum'),
         avg_feedback=('patient_feedback_score','mean'))
    .sort_values(by='total_revenue', ascending=False)
    .head(5)
    .reset_index()
)

print("\nTop 5 Doctors by Revenue & Feedback")
print(report)

# 3. Visualization

# Bar chart for revenue
plt.figure(figsize=(8,5))
plt.bar(report['doctor_id'], report['total_revenue'], color='skyblue')
plt.title("Top 5 Doctors by Revenue")
plt.xlabel("Doctor ID")
plt.ylabel("Total Revenue")
plt.tight_layout()
plt.show()

# Line chart for average patient feedback score
plt.figure(figsize=(8,5))
plt.plot(report['doctor_id'], report['avg_feedback'], marker='o', color='green')
plt.title("Average Patient Feedback Score for Top 5 Doctors")
plt.xlabel("Doctor ID")
plt.ylabel("Average Feedback Score")
plt.tight_layout()
plt.show()

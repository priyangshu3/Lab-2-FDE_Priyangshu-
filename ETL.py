# ETL.py - Extract, Transform, Load for Healthcare Data

import pandas as pd
import os

# Paths
patient_file = "raw_data/patients_data_with_doctor.csv"
doctor_file = "raw_data/doctors_info.csv"
feedback_file = "raw_data/patient_feedback.json"
output_dir = "data_warehouse"
processed_file = f"{output_dir}/processed_healthcare_data.csv"

# Ensure directory exists
os.makedirs(output_dir, exist_ok=True)

# 1. Ingestion
patients = pd.read_csv(patient_file)
doctors = pd.read_csv(doctor_file)
feedback = pd.read_json(feedback_file)

# 2. Cleansing
patients['treatment_cost'] = pd.to_numeric(patients['treatment_cost'], errors='coerce').fillna(0)
patients['room_cost'] = pd.to_numeric(patients['room_cost'], errors='coerce').fillna(0)
patients['treatment_date'] = pd.to_datetime(patients['treatment_date'], errors='coerce')
feedback['review_date'] = pd.to_datetime(feedback['review_date'], errors='coerce')

# 3. Transformation
patients['total_revenue'] = patients['treatment_cost'] + patients['room_cost']

# Merge with doctors info
merged = patients.merge(doctors, on='doctor_id', how='left')

# ✅ Merge with feedback only on patient_id (not treatment_id, since IDs don’t align)
merged = merged.merge(
    feedback[['patient_id','patient_feedback_score']],
    on='patient_id',
    how='left'
)

# 4. Load
merged.to_csv(processed_file, index=False)
print(f"[ETL] Processed healthcare data saved at {processed_file}")
print(merged.head())

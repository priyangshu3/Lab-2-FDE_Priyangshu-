# ML.py - VIP Patient Classification with Reverse ETL

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os

# Paths
processed_file = "data_warehouse/processed_healthcare_data.csv"
output_dir = "reverse_etl_output"
enriched_file = f"{output_dir}/enriched_healthcare_data.csv"

# Ensure output dir exists
os.makedirs(output_dir, exist_ok=True)

# 1. Load data
df = pd.read_csv(processed_file)

# ✅ Step 1: Aggregate feedback at patient level
patient_feedback = (
    df.groupby('patient_id')['patient_feedback_score']
    .mean()
    .reset_index()
)

# Merge aggregated feedback back into df
df = df.drop(columns=['patient_feedback_score'], errors='ignore')
df = df.merge(patient_feedback, on='patient_id', how='left')

# 2. Aggregate patient-level stats
patient_stats = df.groupby('patient_id').agg(
    total_spent=('total_revenue','sum'),
    visits=('treatment_id','count'),
    avg_spent=('total_revenue','mean'),
    avg_feedback=('patient_feedback_score','mean')
).reset_index()

# 3. Preprocess
X = patient_stats[['total_spent','visits','avg_spent','avg_feedback']].fillna(0)

# 4. ML Model (Clustering to separate VIP vs Non-VIP)
kmeans = KMeans(n_clusters=2, random_state=42)
patient_stats['cluster'] = kmeans.fit_predict(X)

# Decide which cluster is VIP (higher spending)
vip_cluster = patient_stats.groupby('cluster')['total_spent'].mean().idxmax()
patient_stats['vip_status'] = np.where(patient_stats['cluster']==vip_cluster, "VIP", "Non-VIP")

# 5. Merge back into main data
df = df.merge(patient_stats[['patient_id','avg_feedback','vip_status']], on='patient_id', how='left')

# 6. Reverse ETL (Save enriched data)
df.to_csv(enriched_file, index=False)
print(f"[ML] Enriched healthcare data with VIP status saved at {enriched_file}")
print(df[['patient_id','doctor_id','total_revenue','avg_feedback','vip_status']].head())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv("bs140513_032310.csv")

# ----------------------------
# Objective 1: Data Handling & Processing
# ----------------------------

# Clean column names
df.columns = df.columns.str.strip()

# Drop missing values (if any)
df = df.dropna()

# Simulate timestamps for demo purposes
np.random.seed(42)
df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='min')
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()

# High-risk merchants (top 10 with most frauds)
fraud_locations = df[df['fraud'] == 1]['merchant'].value_counts().head(10)

# ----------------------------
# Objective 2: Fraud Analysis
# ----------------------------

# Overall fraud rate
fraud_rate = df['fraud'].mean() * 100

# High-risk transactions
high_amount_frauds = df[(df['fraud'] == 1) & (df['amount'] > 10000)]
odd_hour_frauds = df[(df['fraud'] == 1) & ((df['hour'] < 6) | (df['hour'] > 22))]

# Most fraud-prone merchants
fraud_by_merchant = df[df['fraud'] == 1]['merchant'].value_counts().head(10)

# ----------------------------
# Objective 3: Statistical Insights
# ----------------------------

fraud_percentage_by_category = df.groupby('category')['fraud'].mean() * 100
amount_distribution = df['amount'].describe()

# ----------------------------
# Objective 4: Data Visualization
# ----------------------------

plt.figure(figsize=(10, 5))
sns.histplot(df['amount'], bins=100, kde=True)
plt.title('Transaction Distribution by Amount')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='fraud')
plt.title('Fraud vs Non-Fraud Transactions')
plt.xlabel('Is Fraud')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=df[df['fraud'] == 1], x='hour')
plt.title('Hourly Fraud Trend')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Frauds')
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=df[df['fraud'] == 1], y='merchant', order=fraud_by_merchant.index)
plt.title('Top Fraud-Prone Merchants')
plt.xlabel('Fraud Count')
plt.ylabel('Merchant')
plt.show()

# ----------------------------
# Objective 5: Anomaly Detection Using Rules
# ----------------------------

def rule_based_fraud(row):
    if row['amount'] > 10000:
        return 1
    if row['hour'] < 6 or row['hour'] > 22:
        return 1
    return 0

df['rule_based_flag'] = df.apply(rule_based_fraud, axis=1)

# ----------------------------
# Objective 6: Interactive Dashboard Summary
# ----------------------------

fraud_summary = {
    'Total Transactions': len(df),
    'Total Frauds': df['fraud'].sum(),
    'Fraud Rate (%)': round(fraud_rate, 2),
    'High-Value Frauds (>10k)': len(high_amount_frauds),
    'Odd-Hour Frauds': len(odd_hour_frauds),
    'Top Fraud-Prone Merchants': fraud_by_merchant.to_dict()
}

# Print summary
for k, v in fraud_summary.items():
    print(f"{k}: {v}")

# Save cleaned file (optional)
df.to_csv("cleaned_fraud_data.csv", index=False)

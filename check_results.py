# check_results.py
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('data/sustainability_data.db')

# Get extraction count
count = pd.read_sql("SELECT COUNT(*) as count FROM extractions", conn)
print(f"\nTotal extractions: {count['count'][0]}")

# Get all extractions
df = pd.read_sql("""
    SELECT company, indicator_name, value, unit, confidence, 
           source_page, data_quality 
    FROM extractions
""", conn)

print(f"\n{df}")

# Save to CSV
df.to_csv('data/output/sustainability_extractions.csv', index=False)
print("\n✅ CSV saved to: data/output/sustainability_extractions.csv")

conn.close()

import pandas as pd

# Read the CSV files into DataFrames
csv_file1 = 'merged.csv'
csv_file2 = 'air.csv'

df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)

# Select the specific column from the second DataFrame
shared_column = 'SUBDISTRICT'
df2_selected = df2[['SUBDISTRICT', 'PM2.5']]

# Merge the DataFrames on the shared column (e.g., 'shared_column')
merged_df = df1.merge(df2_selected, on=shared_column)

# Save the merged DataFrame to a new CSV file
output_file = 'merged.csv'
merged_df.to_csv(output_file, index=False)

print(f'Merged CSV files saved as {output_file}')

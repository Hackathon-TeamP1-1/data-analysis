#!/usr/bin/env python
# coding: utf-8

# In[63]:


import os
import pandas as pd


# In[64]:


base_path = 'data/nasa'
starting_year = 2020
ending_year = 2025


# In[65]:


all_subdirs = [
    d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
]

year_dirs = []
for d in all_subdirs:
    if d.isdigit():
        year_int = int(d)
        if starting_year <= year_int <= ending_year:
            year_dirs.append(year_int)
year_dirs.sort()


# In[66]:


merged_dir = os.path.join(base_path, 'merged')
os.makedirs(merged_dir, exist_ok=True)


# In[67]:


for var_num in range(1, 36):
    dfs = []

    for year in year_dirs:
        filename = f"POWER_Regional_Daily_{year}0101_{year}1231 ({var_num}).csv"
        file_path = os.path.join(base_path, str(year), filename)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path, skiprows=9)
            dfs.append(df)

    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.sort_values(by="LAT", inplace=True)
        merged_df.reset_index(drop=True, inplace=True)

        out_filename = f"POWER_Regional_Daily_Merged ({var_num}).csv"
        out_path = os.path.join(merged_dir, out_filename)
        merged_df.to_csv(out_path, index=False)

        print(f"Variable ({var_num}) merged and saved to {out_path}")
    else:
        print(f"No files found for variable ({var_num}) in the given year range.")


# In[68]:


def merge_all_variables(
    merged_dir="data/nasa/merged", output_file="all_variables_merged.csv"
):
    """
    Merges all CSV files in `merged_dir` that match the pattern:
    'POWER_Regional_Daily_Merged (*.csv)'.

    Each CSV is expected to have:
        LAT, LON, YEAR, MO, DY, <VARIABLE_COLUMN>
    The script:
        1. Reads each CSV.
        2. Identifies the variable column (anything not in {LAT,LON,YEAR,MO,DY}).
        3. Performs an outer merge on [LAT, LON, YEAR, MO, DY].
        4. Sorts by these key columns and writes the final DataFrame to `output_file`.
    """

    key_cols = ["LAT", "LON", "YEAR", "MO", "DY"]

    # List all "POWER_Regional_Daily_Merged" CSV files in the directory
    all_files = [
        f
        for f in os.listdir(merged_dir)
        if f.startswith("POWER_Regional_Daily_Merged") and f.endswith(".csv")
    ]

    # Sort file names just for consistency (optional)
    all_files.sort()

    merged_df = None

    for csv_file in all_files:
        file_path = os.path.join(merged_dir, csv_file)

        # Read the CSV
        df = pd.read_csv(file_path)

        # Identify the variable column(s)
        var_cols = [c for c in df.columns if c not in key_cols]

        # If there's exactly 1 variable column, we proceed
        if len(var_cols) == 1:
            var_name = var_cols[0]  # e.g. "CLRSKY_SFC_SW_DWN" or "ALLSKY_SFC_SW_DNI"

            if merged_df is None:
                # First file becomes the base DataFrame
                merged_df = df
            else:
                # Outer merge so we keep all rows from both DataFrames
                merged_df = pd.merge(merged_df, df, on=key_cols, how="outer")
        else:
            print(
                f"Warning: {csv_file} has {len(var_cols)} variable columns; skipping."
            )

    # Final sorting by the key columns
    if merged_df is not None:
        merged_df.sort_values(by=key_cols, inplace=True)
        merged_df.reset_index(drop=True, inplace=True)

        # Write to CSV
        output_path = os.path.join(merged_dir, output_file)
        merged_df.to_csv(output_path, index=False)
        print(f"All variables merged. Final file saved at: {output_path}")
    else:
        print("No valid files found to merge or no variable columns detected.")


# Run the merge
merge_all_variables()


# In[69]:


nasa_data = pd.read_csv("data/nasa/merged/all_variables_merged.csv")
nasa_data.head()


# In[70]:


missing_values_before = nasa_data.isnull().sum()

# Display the updated missing values summary
missing_data_summary_before = pd.DataFrame({
    "Missing Values": missing_values_before,
    "Percentage": (missing_values_before / len(nasa_data)) * 100
}).sort_values(by="Missing Values", ascending=False)

missing_data_summary_before.head()


# In[71]:


# Perform group-wise interpolation based on LAT and LON
nasa_data.sort_values(by=["LAT", "LON", "YEAR", "MO", "DY"], inplace=True)

# Apply interpolation to fill missing values using neighboring LAT/LON data
nasa_data.interpolate(method="linear", limit_direction="both", inplace=True)

# Check if there are still missing values
missing_values_after = nasa_data.isnull().sum()

# Display the updated missing values summary
missing_data_summary_after = pd.DataFrame({
    "Missing Values": missing_values_after,
    "Percentage": (missing_values_after / len(nasa_data)) * 100
}).sort_values(by="Missing Values", ascending=False)

missing_data_summary_after.head()


# In[ ]:


get_ipython().system('jupyter nbconvert --to script "nasa_dataset_inspection.ipynb" --output-dir="outputs/scripts"')
get_ipython().system('jupyter nbconvert --to html "nasa_dataset_inspection.ipynb" --output-dir="outputs/html"')


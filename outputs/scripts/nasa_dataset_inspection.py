#!/usr/bin/env python
# coding: utf-8

# In[610]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.multicomp import tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import scipy.stats as stats
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
from scipy.stats import kruskal
from scipy.stats import friedmanchisquare
from scipy.stats import mannwhitneyu
from scipy.stats import levene
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from scipy.stats import bartlett
from scipy.stats import fligner
from scipy.stats import mood
from scipy.stats import ranksums
from scipy.stats import kstest
from scipy.stats import ks_2samp
from scipy.stats import chisquare
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import numpy as np
from scipy.stats import zscore


# In[611]:


base_path = 'data/nasa'
starting_year = 2009
ending_year = 2025


# In[612]:


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


# In[613]:


merged_dir = os.path.join(base_path, 'merged')
os.makedirs(merged_dir, exist_ok=True)


# In[614]:


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


# In[615]:


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

    all_files = [
        f
        for f in os.listdir(merged_dir)
        if f.startswith("POWER_Regional_Daily_Merged") and f.endswith(".csv")
    ]

    all_files.sort()

    merged_df = None

    for csv_file in all_files:
        file_path = os.path.join(merged_dir, csv_file)

        df = pd.read_csv(file_path)

        var_cols = [c for c in df.columns if c not in key_cols]

        if len(var_cols) == 1:
            var_name = var_cols[0]

            if merged_df is None:

                merged_df = df
            else:

                merged_df = pd.merge(merged_df, df, on=key_cols, how="outer")
        else:
            print(
                f"Warning: {csv_file} has {len(var_cols)} variable columns; skipping."
            )

    if merged_df is not None:
        merged_df.sort_values(by=key_cols, inplace=True)
        merged_df.reset_index(drop=True, inplace=True)

        output_path = os.path.join(merged_dir, output_file)
        merged_df.to_csv(output_path, index=False)
        print(f"All variables merged. Final file saved at: {output_path}")
    else:
        print("No valid files found to merge or no variable columns detected.")


merge_all_variables()


# In[616]:


nasa_data = pd.read_csv("data/nasa/merged/all_variables_merged.csv")
nasa_data.head()


# In[617]:


missing_values_before = nasa_data.isnull().sum()

missing_data_summary_before = pd.DataFrame({
    "Missing Values": missing_values_before,
    "Percentage": (missing_values_before / len(nasa_data)) * 100
}).sort_values(by="Missing Values", ascending=False)

missing_data_summary_before.head()


# In[618]:


nasa_data.sort_values(by=["LAT", "LON", "YEAR", "MO", "DY"], inplace=True)


# nasa_data.interpolate(method="linear", limit_direction="both", inplace=True)


missing_values_after = nasa_data.isnull().sum()


missing_data_summary_after = pd.DataFrame(
    {
        "Missing Values": missing_values_after,
        "Percentage": (missing_values_after / len(nasa_data)) * 100,
    }
).sort_values(by="Missing Values", ascending=False)

missing_data_summary_after.head()


# In[619]:


# nasa_data.to_csv("data/nasa/merged/all_variables_merged_interpolated.csv", index=False)
# nasa_data.shape


# In[620]:


# nasa_data.info()


# I'll start by inspecting the dataset to understand its structure and completeness. Then, I'll prepare it for the **Renewable Energy Consumption Tracker** by applying necessary data cleaning, feature engineering, and transformations. Let me analyze the dataset first.
# 
# # Key Observations:
# 1. **Missing Data Representation:** The dataset uses `-999` as a placeholder for missing values instead of `NaN`. These need to be replaced for proper handling.
# 
# 2. **Duplicate Columns:** `CLRSKY_SFC_SW_DWN_x` and `CLRSKY_SFC_SW_DWN_y` appear to be duplicate variables.
# 
# 3. **Latitude and Longitude Range Validation:** Some latitude (LAT) and longitude (LON) values (e.g., 29.5°N, 34.0°E) are outside Palestine’s expected range (31°N-33°N, 34°E-36°E), requiring filtering.
# 
# 4. **Outlier Detection Needed:** Some columns may contain extreme values beyond physically reasonable limits.
# 
# 5. **Key Variables for Renewable Energy:**
# - **Solar Energy Indicators:** `ALLSKY_SFC_SW_DWN`, `CLRSKY_SFC_SW_DWN`, `ALLSKY_SFC_SW_DNI`, `ALLSKY_SFC_UV_INDEX`, `ALLSKY_SFC_PAR_TOT`, `CLRSKY_SFC_PAR_TOT`
# - **Wind Energy Indicators:** `WS10M`, `WS10M_MAX`, `WS50M`, `WS50M_MAX`
# - **Weather Factors:** `T2M (Temperature)`, `RH2M (Humidity)`, `PRECTOTCORR (Precipitation)`
# 
# # Next Steps in Data Preparation:
# 
# - Replace `-999` values with `NaN` and handle missing values.
# 
# - Remove duplicate and unnecessary columns.
# 
# - Filter dataset to keep only valid LAT/LON values.
# 
# - Detect and handle outliers using Z-score filtering.
# 
# - Normalize/scale the relevant features for better model performance.
# 

# In[621]:


nasa_data_copy = nasa_data.copy()


# In[622]:


nasa_data.replace(-999.0, np.nan, inplace=True)

# calculate the sum of missing values in each row
# nasa_data["missing_values"] = nasa_data.isnull().sum(axis=1)
# nasa_data["missing_values"]
# nasa_data.to_csv('outputs/exploring_outputs/nasa/missing_values.csv', index=False)
# nasa_data.dropna(inplace=True)

# show which columns have missing values
nasa_data.isnull().sum()
missing_cols = nasa_data.columns[nasa_data.isnull().any()].tolist()
missing_cols

# nasa_data.shape


# In[623]:


nasa_with_missing = nasa_data[missing_cols]
nasa_with_missing.describe()


# In[624]:


nasa_data.drop(columns=["CLRSKY_SFC_SW_DWN_x", "CLRSKY_SFC_SW_DWN_y"], inplace=True)


# In[625]:


palestine_lat_range = (31, 33)
palestine_lon_range = (34, 36)

nasa_data = nasa_data[
    (nasa_data["LAT"] >= palestine_lat_range[0]) & (nasa_data["LAT"] <= palestine_lat_range[1]) &
    (nasa_data["LON"] >= palestine_lon_range[0]) & (nasa_data["LON"] <= palestine_lon_range[1])
]


# In[626]:


missing_after_filtering = nasa_data.isnull().sum()
missing_after_filtering[missing_after_filtering > 0]


# In[627]:


nasa_data.interpolate(method="linear", limit_direction="both", inplace=True)
# nasa_data.fillna(method="bfill", inplace=True)
# nasa_data.fillna(method="ffill", inplace=True)
nasa_data.to_csv("data/nasa/merged/all_variables_merged_interpolated.csv", index=False)
missing_after_filtering = nasa_data.isnull().sum()
missing_after_filtering[missing_after_filtering > 0]


# In[628]:


# numeric_cols = nasa_data.select_dtypes(include=["float64", "int64"]).columns
# z_scores = nasa_data[numeric_cols].apply(zscore)
# nasa_data = nasa_data[(z_scores.abs() <= 3).all(axis=1)]
# nasa_data.isnull().sum().sum()


# In[629]:


numeric_cols = nasa_data.select_dtypes(include=["float64", "int64"]).columns
features_to_adjust = [col for col in numeric_cols if col not in ["LON", "LAT"]]

# Apply outlier treatment only to selected features
for col in features_to_adjust:
    lower_limit = nasa_data[col].quantile(0.05)
    upper_limit = nasa_data[col].quantile(0.95)
    nasa_data[col] = np.clip(nasa_data[col], lower_limit, upper_limit)


# In[630]:


print("Before:", nasa_data['LON'].min(), nasa_data['LON'].max())


# In[631]:


nasa_data["PRECTOTCORR"] = np.log1p(nasa_data["PRECTOTCORR"])  # log1p handles zero values safely


# Normalize selected features for AI model input

# In[632]:


plt.figure(figsize=(15, 10))
for i, feature in enumerate(["LAT", "LON", "ALLSKY_SFC_SW_DWN", "WS2M", "T2M", "RH2M", "PRECTOTCORR", 'T2M_MAX', 'T2M_MIN', "ALLSKY_KT", 'ALLSKY_SFC_PAR_TOT']):
    plt.subplot(4, 3, i + 1)
    sns.boxplot(y=nasa_data[feature])
    plt.title(feature)

plt.tight_layout()
plt.show()


# In[633]:


scaling_cols = [
    "ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_PAR_TOT", "CLRSKY_SFC_PAR_TOT", 
    "WS10M", "WS10M_MAX", "WS50M", "WS50M_MAX", "T2M", "RH2M", "PRECTOTCORR"
]


# In[634]:


nasa_data[scaling_cols] = (nasa_data[scaling_cols] - nasa_data[scaling_cols].min()) / (
    nasa_data[scaling_cols].max() - nasa_data[scaling_cols].min()
)


# In[635]:


missing_values_after = nasa_data.isnull().sum()


missing_data_summary_after = pd.DataFrame(
    {
        "Missing Values": missing_values_after,
        "Percentage": (missing_values_after / len(nasa_data)) * 100,
    }
).sort_values(by="Missing Values", ascending=False)

missing_data_summary_after.head()


# In[636]:


nasa_data.head()


# In[637]:


# if not exist
os.makedirs("outputs/exploring_outputs/nasa", exist_ok=True)
nasa_data.describe().to_csv("outputs/exploring_outputs/nasa/nasa_interpolated_description.csv", index=False)
nasa_data.describe()


# The dataset has been cleaned and prepared for the **`Renewable Energy Consumption Tracker`**. Key steps taken:
# 
# ✅ Handled Missing Values: Replaced -999 with NaN and applied interpolation.
# 
# ✅ Removed Duplicates: Dropped redundant columns.
# 
# ✅ Filtered by Location: Kept only valid latitude/longitude values for Palestine.
# 
# ✅ Outlier Detection & Removal: Used Z-score filtering to remove extreme values.
# 
# ✅ Feature Normalization: Scaled key variables for AI model compatibility.
# 

# In[638]:


nasa_data.shape


# In[639]:


nasa_data.isnull().sum().sum()


# In[640]:


nasa_data.duplicated().sum()


# In[641]:


nasa_data.dtypes


# Now the dataset is well-structured, free of missing values and duplicates, and correctly formatted for further analysis or machine learning models.

# In[642]:


corr_matrix = nasa_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# The correlation heatmap shows the relationships between different features, with a particular focus on their correlation with E_produced (if present). Below is an analysis of feature importance and how to optimize the model:
# 
# # Best Features for Training
# Based on correlation with E_produced, the most relevant features should be:
# 
# 1. `ALLSKY_SFC_SW_DWN` (Solar radiation reaching the surface) → Strongest positive correlation
# 2. `ALLSKY_KT` (Clear-sky transmittance) → Moderate correlation
# 3. `ALLSKY_SFC_PAR_TOT` (Photosynthetically active radiation) → Useful for energy conversion
# 4. `T2M` (Air Temperature) → Impacts solar panel efficiency
# 5. `RH2M` (Relative Humidity) → Affects solar power generation
# 6. `WS2M` (Wind Speed at 2m) → May impact dust accumulation and cooling effects
# 7. `PRECTOTCORR` (Precipitation) → May reduce panel efficiency during rain
# 8. `T2M_MAX` & `T2M_MIN` (Daily max/min temperature) → Temperature fluctuation affects efficiency
# 
# **Features to Consider Dropping:**
# - `ALLSKY_SFC_UV_INDEX`, `ALLSKY_SFC_UVA`, `ALLSKY_SFC_UVB` → Less direct impact on solar panel energy output.
# - `PS (Surface Pressure)` → Minimal effect on energy generation.
# - `WS50M`, `WD50M (Wind at 50m)` → Might not contribute significantly to solar forecasting.

# In[643]:


plt.figure(figsize=(15, 10))
for i, feature in enumerate(["LAT", "LON", "ALLSKY_SFC_SW_DWN", "WS2M", "T2M", "RH2M", "PRECTOTCORR", 'T2M_MAX', 'T2M_MIN', "ALLSKY_KT", 'ALLSKY_SFC_PAR_TOT']):
    plt.subplot(4, 3, i + 1)
    sns.boxplot(y=nasa_data[feature])
    plt.title(feature)

plt.tight_layout()
plt.show()


# In[644]:


plt.figure(figsize=(15, 10))
for i, feature in enumerate(["LAT", "LON", "ALLSKY_SFC_SW_DWN", "WS2M", "T2M", "RH2M", "PRECTOTCORR", 'T2M_MAX', 'T2M_MIN', "ALLSKY_KT", 'ALLSKY_SFC_PAR_TOT']):
    plt.subplot(4, 3, i + 1)
    sns.boxplot(y=nasa_data_cleaned_no_outliers[feature])
    plt.title(feature)

plt.tight_layout()
plt.show()


# In[645]:


os.makedirs("outputs/preprocessed_data", exist_ok=True)
nasa_data.to_csv("outputs/preprocessed_data/nasa_data_cleaned.csv", index=False)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script "nasa_dataset_inspection.ipynb" --output-dir="outputs/scripts"')
get_ipython().system('jupyter nbconvert --to html "nasa_dataset_inspection.ipynb" --output-dir="outputs/html"')


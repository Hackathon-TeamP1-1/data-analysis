#!/usr/bin/env python
# coding: utf-8

# # Analysis of the Datasets and Strategy for AI-Powered Renewable Energy Consumption & Forecasting Dashboard

# # Objectives
# 
# - **Understand the Dataset:** Analyze the provided datasets to assess their structure, completeness, and relevance to the project goals.
# 
# - **Explore Renewable Energy Consumption Trends:** Identify historical patterns in renewable energy consumption at national, continental, and global levels.
# 
# - **Assess the Impact of Investments on Renewable Energy Growth:** Examine how public and private investments correlate with the increase in renewable energy production.
# 
# - **Evaluate Policy and Macroeconomic Influences:** Analyze how economic factors, government policies, and global trends impact renewable energy adoption.
# 
# - **Prepatre a aData set for Develop an AI-Powered Interactive Dashboard:** Create a cleaned dataset for a web-based visualization platform to display energy consumption trends, forecasts, and policy insights.
# 
# - **Ensure Data Consistency and Integrity:** Preprocess, clean, and integrate multiple datasets for accurate analysis and forecasting.
# 
# # Questions:
# 
# ## Data Understanding & Preparation
# - What are the key variables in each dataset, and how do they relate to renewable energy consumption and forecasting?
# 
# - Are there any missing values or inconsistencies that need to be addressed before analysis?
# 
# - How can different datasets be merged effectively for better insights and predictions?
# 
# ## Trend Analysis & Forecasting
# - What are the historical trends in renewable energy consumption for different countries and continents?
# 
# - How has the share of renewable energy changed over the past few decades?
# 
# - Which forecasting model (ARIMA, Prophet, LSTM) provides the most accurate predictions for future renewable energy adoption?
# 
# - What are the expected trends in renewable energy adoption for the next 5–10 years?
# 
# ## Investment & Economic Impact
# - How does public and private investment impact renewable energy production and adoption?
# 
# - Is there a correlation between GDP growth and increased renewable energy consumption?
# 
# - Which countries have successfully scaled their renewable energy infrastructure, and what investment patterns support this growth?
# 
# ## Policy & Regulatory Factors
# - What policies have been most effective in promoting renewable energy adoption?
# 
# - How do government incentives and regulations affect renewable energy trends?
# 
# - Are there any observable macroeconomic factors (e.g., inflation, energy prices) that influence renewable energy adoption?
# 
# ## Visualization & Dashboard Development
# - What types of visualizations (heatmaps, line charts, bar charts, dashboards) best represent energy consumption and forecasting insights?
# 
# - How can an interactive dashboard help policymakers, businesses, and researchers make data-driven decisions?
# 
# - What features should be included in the dashboard to allow users to explore data dynamically?

# In[282]:


# the goal is the analyze the data and find the best way to predict the Strategy for AI-Powered Renewable Energy Consumption & Forecasting Dashboard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# ## Datasets Loading
# ### 1. Our World in Data (OWID) - Global Energy Dataset

# In[283]:


# loading the datasets
# owid data
owid_data = pd.read_csv('data/OWID/owid-energy-data.csv')
owid_bookcode = pd.read_csv('data/OWID/owid-energy-codebook.csv')

print(
    owid_data.shape,
    owid_bookcode.shape,
)


# In[284]:


owid_data.head()


# In[285]:


owid_data.sample(10)


# ### 2. Global Energy Consumption & Renewable Generation || Kaggle

# In[286]:


# global energy data
continent_consumption = pd.read_csv("data/global_energy/Continent_Consumption_TWH.csv")
country_consumption = pd.read_csv("data/global_energy/Country_Consumption_TWH.csv")
non_renewable_total_power_generation = pd.read_csv(
    "data/global_energy/nonRenewablesTotalPowerGeneration.csv"
)
renewable_power_generation_97_17 = pd.read_csv(
    "data/global_energy/renewablePowerGeneration97-17.csv"
)
renewable_total_power_generation = pd.read_csv(
    "data/global_energy/renewablesTotalPowerGeneration.csv"
)
top_20_countries_power_generatoin = pd.read_csv(
    "data/global_energy/top20CountriesPowerGeneration.csv"
)

# print the shapes of the datasets
print(
    continent_consumption.shape,
    country_consumption.shape,
    non_renewable_total_power_generation.shape,
    renewable_power_generation_97_17.shape,
    renewable_total_power_generation.shape,
    top_20_countries_power_generatoin.shape,
)


# In[287]:


continent_consumption.head()


# In[288]:


country_consumption.head()


# In[289]:


country_consumption.head()


# In[290]:


non_renewable_total_power_generation.head()


# In[291]:


renewable_power_generation_97_17.head()


# In[292]:


renewable_total_power_generation.head()


# In[293]:


top_20_countries_power_generatoin.head()


# ### Energy Generation & Consumption (from multiple sources)

# In[294]:


electricity_consumption_statistics = pd.read_csv('data/IRR_cleaned/ELECSTAT_CLEANED.csv')
heat_generations = pd.read_csv('data/IRR_cleaned/HEATGEN_CLEANED.csv')
share_of_renewables = pd.read_csv('data/IRR_cleaned/RESHARE_CLEANED.csv')
investment_in_energy_infrastructure = pd.read_csv('data/IRR_cleaned/PUBFIN_CLEANED.csv')

print(
    electricity_consumption_statistics.shape,
    heat_generations.shape,
    share_of_renewables.shape,
    investment_in_energy_infrastructure.shape,
)


# In[295]:


electricity_consumption_statistics.head()


# In[296]:


heat_generations.head()


# In[297]:


share_of_renewables.head()


# In[298]:


investment_in_energy_infrastructure.head()


# # 1. Understanding the Datasets

# After reviewing the three provided datasets, we can categorize them as follows:
# 

# ## 1. Dataset Group 1: Our World in Data (OWID) - Global Energy Dataset
# 
# **Section Summary:**
# - **owid-energy-data.csv**: A dataset covering global energy production, electricity mix, and energy consumption trends from various sources (hydro, wind, solar, fossil fuels, etc.).
# 
# - **owid-energy-codebook.csv**: A codebook detailing column descriptions and data sources for the OWID dataset.

# In[299]:


owid_data.columns


# In[300]:


owid_bookcode.columns


# In[301]:


# save the output on a text file
owid_data.describe().to_csv('outputs/exploring_outputs/owid/owid_data_describe.csv')
owid_data.describe()


# The bookcode provides metadata about the cols, we can benefit from that by visually navigate through the csv. 
# 
# After Reviewing the bookcode CSV, I can benefit from the "column" and the "units" columns through my programatically exploring into this dataset.

# In[302]:


owid_bookcode = owid_bookcode[['column', 'unit']]
owid_bookcode.dropna(inplace=True)
owid_bookcode.head()


# In[303]:


owid_info_df = pd.DataFrame(owid_data.dtypes, columns=['data_type']).reset_index()
owid_info_df.to_csv('outputs/exploring_outputs/owid/owid_info.csv')

# owid_info_df['missing_values'] = owid_data.isnull().sum()


# In[304]:


owid_data.value_counts().to_csv('outputs/exploring_outputs/owid/owid_data_value_counts.csv')   


# In[305]:


missing_df = owid_data.isnull().sum().reset_index()
missing_df.columns = ['column', 'missing_values']
missing_df['total_values'] = owid_data.shape[0]
missing_df['missing_percentage'] = missing_df['missing_values'] / missing_df['total_values'] * 100
missing_df.sort_values('missing_percentage', ascending=False, inplace=True)
missing_df.to_csv('outputs/exploring_outputs/owid/owid_data_missing.csv')


# In[306]:


def basic_info(df, name):
    print(f"\n{name} Dataset Info:")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nSummary Statistics:")
    print(df.describe(include='all'))


# In[307]:


basic_info(owid_data, "Energy Data")


# In[308]:


display(owid_data.head())


# In[309]:


def plot_energy_trends(df, energy_source):
    """Plot trends in energy production/consumption over time."""
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="year", y=energy_source, hue="country", legend=False)
    plt.title(f"Trends in {energy_source}")
    plt.xlabel("Year")
    plt.ylabel(energy_source)
    plt.savefig(f"outputs/exploring_outputs/owid/figures/{energy_source}_trends.png")

def plot_energy_correlation(df):
    """Plot correlation matrix of energy sources."""
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Energy Sources")
    plt.show()
    plt.savefig(f"outputs/exploring_outputs/owid/figures/energy_correlation.png")


def plot_energy_sources(df):
    """Plot energy sources."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="value", y="source", palette="viridis")
    plt.title("Energy Sources")
    plt.xlabel("Value")
    plt.ylabel("Source")
    plt.show()
    plt.savefig(f"outputs/exploring_outputs/owid/figures/energy_sources.png")


def plot_energy_sources_by_country(df, country):
    """Plot energy sources by country."""
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df[df["country"] == country], x="value", y="source", palette="viridis"
    )
    plt.title(f"Energy Sources in {country}")
    plt.xlabel("Value")
    plt.ylabel("Source")
    plt.show()
    plt.savefig(f"outputs/exploring_outputs/owid/figures/{country}_energy_sources.png")


def plot_energy_sources_by_year(df, year):
    """Plot energy sources by year."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df[df["year"] == year], x="value", y="source", palette="viridis")
    plt.title(f"Energy Sources in {year}")
    plt.xlabel("Value")
    plt.ylabel("Source")
    plt.show()
    plt.savefig(f"outputs/exploring_outputs/owid/figures/{year}_energy_sources.png")


def plot_energy_sources_by_continent(df, continent):
    """Plot energy sources by continent."""
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df[df["continent"] == continent], x="value", y="source", palette="viridis"
    )
    plt.title(f"Energy Sources in {continent}")
    plt.xlabel("Value")
    plt.ylabel("Source")
    plt.show()
    plt.savefig(
        f"outputs/exploring_outputs/owid/figures/{continent}_energy_sources.png"
    )


def plot_energy_sources_by_region(df, region):
    """Plot energy sources by region."""
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df[df["region"] == region], x="value", y="source", palette="viridis"
    )
    plt.title(f"Energy Sources in {region}")
    plt.xlabel("Value")
    plt.ylabel("Source")
    plt.show()
    plt.savefig(f"outputs/exploring_outputs/owid/figures/{region}_energy_sources.png")


def plot_energy_sources_by_income_group(df, income_group):
    """Plot energy sources by income group."""
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df[df["income_group"] == income_group],
        x="value",
        y="source",
        palette="viridis",
    )
    plt.title(f"Energy Sources in {income_group}")
    plt.xlabel("Value")
    plt.ylabel("Source")
    plt.show()
    plt.savefig(
        f"outputs/exploring_outputs/owid/figures/{income_group}_energy_sources.png"
    )


# In[310]:


energy_columns = [col for col in owid_data.columns if any(x in col for x in ['energy', 'electricity', 'consumption', 'production'])]
print("Available energy-related columns:")
print(energy_columns)


# In[311]:


for col in energy_columns:
    if owid_data[col].dtype in ['float64', 'int64']:
        plot_energy_trends(owid_data, col)
        


# In[312]:


def plot_energy_distribution(df, energy_source):
    """Plot distribution of energy production/consumption."""
    plt.figure(figsize=(12, 6))
    sns.histplot(df[energy_source], bins=30, kde=True)
    plt.title(f"Distribution of {energy_source}")
    plt.xlabel(energy_source)
    plt.ylabel("Frequency")
    plt.savefig(
        f"outputs/exploring_outputs/owid/figures/{energy_source}_distribution.png"
    )


# In[313]:


plot_energy_distribution(owid_data, "renewables_consumption")


# ## 2. Dataset Group 2: Global Energy Consumption & Renewable Generation
# - **renewablePowerGeneration97-17.csv:** Tracks renewable power generation trends from 1997 to 2017 across different energy types (hydro, wind, biofuel, solar, geothermal).
# 
# - **renewablesTotalPowerGeneration.csv:** Summarizes total renewable power generation in TWh globally.
# 
# - **nonRenewablesTotalPowerGeneration.csv:** Summarizes total non-renewable power generation in TWh globally.
# 
# - **top20CountriesPowerGeneration.csv:** Highlights the top 20 countries' renewable energy generation.
# 
# - **Country_Consumption_TWH.csv:** Records national energy consumption trends.
# 
# - **Continent_Consumption_TWH.csv:** Records energy consumption trends at a continental level.
# 
# 

# ## Dataset Group 3: Energy Generation & Consumption (from multiple sources)
# 
# - **HEATGEN_CLEANED.csv:** Contains cleaned data on heat generation. This dataset could be useful for understanding how different energy sources contribute to overall energy production.
# 
# - **ELECSTAT_CLEANED.csv:** Likely includes statistics on electricity consumption, possibly broken down by country and year.
# - **RESHARE_CLEANED.csv:** Appears to contain information on the share of renewable energy sources in overall energy consumption.
# - **PUBFIN_CLEANED.csv:** May provide financial data related to public investment in energy infrastructure.

# # 2. Data Suitability for Forecasting and Visualization
# 
# Based on the project requirements, we can determine how each dataset fits into the forecasting and visualization strategy:

# ## A. Forecasting Future Renewable Energy Trends (5-10 years)

# In[ ]:





# # Store the jupyter notebook

# In[ ]:


get_ipython().system('jupyter nbconvert --to script "data_inspection.ipynb" --output-dir="outputs/scripts"')
get_ipython().system('jupyter nbconvert --to html "data_inspection.ipynb" --output-dir="outputs/html"')


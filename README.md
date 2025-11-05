# Retail Sales Data Monthly Multivariate Forecasting
This project focuses on forecasting monthly retail sales and revenue across multiple stores, product categories, and regions. Before building forecasting models, the dataset undergoes exploratory data analysis (EDA) to identify key drivers and relationships among variables such as discounts and promotions to net sales and revenue.

## Key Points
- **Problem:** Retail sales and revenue fluctuate due to various factors, making accurate forecasting essential for business planning.
- **Proposed Solution:** Develop a monthly multivariate forecasting model (SARIMAX) that captures both trend and seasonality while considering external factors.
- **Results:** The SARIMAX models achieved high accuracy, effectively predicting both sales and revenue with minimal deviations.

## Process Overview
1. **Data Wrangling:** Gathered, assessed, and cleaned the raw datasets to ensure a usable format. Find the raw data on [Kaggle: Synthetic Retail Sales Data 2022–2024](https://www.kaggle.com/datasets/shivamja/synthetic-retail-sales-data-20222024).
2. **Exploratory Data Analysis (EDA):** Explored cleaned datasets to identify key variables and meaningful patterns.
3. **Forecasting:** Built and evaluated SARIMAX models to predict future sales and revenue using significant exogenous variables derived from EDA.

For detailed documentation, refer to the analysis process [here](https://github.com/gracypresilia/3_Sales_Data_Forecasting/blob/main/analyzation_process.md).

## File Structure
```
│  
├── data/                                       
│   └── retail_sales_synthetic.csv              # Raw dataset file
├── .gitignore
├── analysis_process.md                         # Overview of the project's analysis process
├── eda.ipynb                                   # Data preprocessing and EDA (Notebook)
├── eda.py                                      # Data preprocessing and EDA (Python script)
├── forecast.ipynb                              # Forecasting model development (Notebook)
├── forecast.py                                 # Forecasting model development (Python script)
├── helper.py                                   # Helper functions python script
├── model_net_revenue.pkl                       # Saved forecast model for revenue
├── model_net_units.pkl                         # Saved forecast model for sales
├── models_exog_scaler.pkl                      # Saved scaler for standardized exogenous variables
├── README.md                                   # Project documentation
└── requirements.txt                            # List of dependencies 
```

## Future Work
This project can be further enhanced by implementing the following ideas.
- Add deterministic seasonality to further reduce peak-month bias.
- Deploy a lightweight application for interactive forecast testing.
- Automate monthly retraining and evaluation with updated data.

## Bibliography
- [Geeks for Geeks: Complete Guide To SARIMAX in Python](https://www.geeksforgeeks.org/python/complete-guide-to-sarimax-in-python/)
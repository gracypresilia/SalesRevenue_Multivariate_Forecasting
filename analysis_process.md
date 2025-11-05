# Retail Sales Data Monthly Multivariate Forecasting Analysis Process
This project focuses on forecasting monthly retail sales and revenue across multiple stores, product categories, and regions. Before building forecasting models, the dataset undergoes exploratory data analysis (EDA) to identify key drivers and relationships among variables such as discounts and promotions to net sales and revenue.

## Raw Data
### Source
[Kaggle: Synthetic Retail Sales Data 2022–2024](https://www.kaggle.com/datasets/shivamja/synthetic-retail-sales-data-20222024)
### Columns Explanation
- **Date**
    - `date`: 1096 unique dates from 1st January 2022 to 31st December 2024.
    - `is_holiday`: Indicates if the date is a holiday (1) or not (0).
    - `day_of_week`: Day of the week ranging from 0-6 (Monday-Sunday).
    - `weekend`: Indicates if the day falls on a weekend (1) or not (0).
- **Store Data**
    - `store_id`: 10 unique store IDs across various regions and cities.
    - `store_type`: 3 unique store types for each unique store ID.
    - `region`: 4 unique regions for each unique store ID.
    - `city`: 10 unique cities for each unique store ID.
    - `store_area_sqft`: Various store area (in sqft) for each unique store ID.
- **Main Product Data**
    - `product_id`: 50 unique product IDs.
    - `category`: 5 unique category for each unique product ID.
    - `base_price`: Base price for each unique product ID.
    - `discount_pct`: Discount percentages ranging from 0-30%.
    - `promotion`: Indicates if the product was on promotion (1) or not (0).
    - `online`: Indicates if the product was sold online (1) or offline (0).
These fixed variables define the unique combinations of each row and determine the dependent variables listed below.
- **Sales and Revenue Data (For each product in their each unique conditions (date, store, etc.))**
    - `final_price`: Various final price.
    - `units_sold`: Number of units sold for each product in each unique conditions.
    - `returns`: Number of units returned by customer for each product in each unique conditions.
    - `net_units`: Net units sold (net sales) for each product in each unique conditions.
    - `revenue`: Gross revenue for each product in each unique conditions.
    - `net_revenue`: Net revenue for each product in each unique conditions.
    - `avg_rating`: Customers rating for each product in each unique conditions.

## Tools and Libraries
### Tools
- **Visual Studio Code:** An IDE for writing, debugging, and running Python scripts in a streamlined development environment.
- **Jupyter Notebook:** A code documentation tool that allows the use of both Markdown for text and Python for coding.
- **Python:** A high-level programming language for data analysis.
### Python Libraries
- **importlib:** Library to reload `helper` script.
- **joblib:** Library to export scaled exogenous variables values.
- **matplotlib:** Library for creating data visualizations
- **numpy:** Library to process numerical operations.
- **pandas:** Library to handle, clean, and process DataFrames.
- **scikit_learn:** Library to scale exogenous variables values and calculate models performance.
- **seaborn:** Library to create advanced statistical plots.
- **statsmodels (SARIMAX):** Library to develop models.

**Note:** To install or check the library versions used in this project, refer [here](https://github.com/gracypresilia/Sales_Data_Forecasting/blob/main/requirements.txt).

## Analysis Pipeline
### EDA (Exploratory Data Analysis)
**Goal:** Understand data distribution, detect patterns, and explore relationships between categorical and numerical variables.
1. Does the presence of holiday affect overall sales and revenue, both daily and monthly?
    - **Proposed Solution:** Find mean difference percentages between variables (`is_holiday`, `net_units`, and `net_revenue`).
    - **Results:** Holidays increase overall daily sales by **27.6%** and revenue by **25.45%**. On a monthly level, holidays still have a positive impact, around **7%**, though the effect is less significant compared to the daily view.
2. Is there any change in product's category trend during no-holiday months and holiday months?
    - **Proposed Solution:** Find mode of `category` for the specific dates.
    - **Results:** There is **no change** in category trends between non-holiday and holiday months. In both cases, **Clothing** remains the most popular category.
3. Does the weekend status affect overall daily sales and revenue?
    - **Proposed Solution:** Find mean difference percentages between variables (`weekend`, `net_units`, and `net_revenue`).
    - **Results:** Weekends increase overall daily sales and revenue by approximately **22.7%**.
4. How is the overall day-by-day sales and revenue trend during a week?
    - **Proposed Solution:** Find average values of sales (`net_units`) and revenue (`net_revenue`) for each day of the week (`day_of_week`).
    - **Results:** Sales and revenue are consistently **higher on weekends**, confirming the results from the previous section that weekend status positively affects daily performance.
5. Does the store type and area affect customer experience and, in turn, store sales?
    - **Proposed Solution:** Find average values of customer experiences (`avg_rating`), sales (`net_units`), and revenue (`net_revenue`) for each pair of store type (`store_type`). Also, find correlation coefficients between variables (`store_area_sqft`, `avg_rating`, `net_units`, and `net_revenue`).
    - **Results:** Store type shows **clear differences in sales and revenue**, while customer ratings vary little by type. Store type C has the highest rank, but store type A has the highest net sales and revenue. On the other hand, store area **doesn't significantly affect** customer experience, sales, or revenue. Interestingly, larger store areas exhibit a negative correlation with ratings, sales, and revenue, suggesting that bigger spaces might not necessarily lead to higher customer satisfaction. In addition, customer experience also doesn't significantly affect sales or revenue and even shows a negative correlation, meaning that higher ratings are associated with lower sales and revenue.
6. Which category of product is the most popular in each city month-by-month?
    - **Proposed Solution:** Find mode of `category` for the specific dates and `city`.
    - **Results:** Each city shows different trends and preferences over time. However, across all months and cities, the top categories are consistently **Clothing** or **Home**.
7. How does discount percentages on products affect store's sales and revenue?
    - **Proposed Solution:** Find mean difference percentages between variables (`discount_pct`, `net_units`, and `net_revenue`). The values are calculated for every percentages to the baseline percentage (0%).
    - **Results:** Discount percentages **slightly impact sales and revenue**. The higher the discount percentages, the higher the sales is. However, by percentages, it is inversely proportional to the revenue, though by amount the impact doesn't show much.
8. How does product's promotion affect store's sales and revenue?
    - **Proposed Solution:** Find mean difference percentages between variables (`promotion`, `net_units`, and `net_revenue`).
    - **Results:** Promotion significantly affects sales and revenue. Its impact is far greater than that of discount percentages—around **67.8%**. Moreover, promotion has a directly proportional relationship with both sales and revenue, where the presence of promotion consistently leads to higher values in both.
9. How does the combination of discount and promotion give different effect to store's sales and revenue?
    - **Proposed Solution:** Find average values of sales (`net_units`) and revenue (`net_revenue`) for each combination of discount (`discount_pct`) and promotion (`promotion`).
    - **Results:** The combination of discount and promotion produces **varying effects** on sales and revenue. A high discount combined with promotion results in the highest sales, while no discount with promotion yields the highest revenue. This finding affirms the results from the previous sections that promotion has a stronger positive impact on both sales and revenue.
10. Does online transaction affect the customer experience (returns and rating)?
    - **Proposed Solution:** Find mean difference percentages between variables (`online`, `returns`, and `avg_rating`).
    - **Results:** Online transactions slightly affect both returns and ratings in a **directly proportional** manner. This is a surprising result, as an increase in returns usually corresponds to lower ratings.

**Conclusions:**
- 🆙 Holidays, weekends, and store type significantly boost sales and revenue.
- 📈 Promotions have the largest positive impact, far surpassing discounts.
- 📊 Store area and online transaction do not show meaningful positive correlations with performance metrics.
- 🏷️ Across cities and months, Clothing and Home remain the dominant categories.
### Forecasting
**Goal:** Predict future sales and revenue based on historical data.  
**Proposed Solution:** Build a monthly forecasting model using a multivariate time-series approach (SARIMAX).  
**Results:**
1. Sales Forecasting
    - MAE: 100 | MAPE: **0.50%**
    - The sales model tracked the late-2024 surge closely with only minor deviations in the test window.
    - In the residual plot, the sales model displayed a temporary overprediction/underfitting period but quickly adapted afterward.
2. Revenue Forecasting
    - MAE: 12,912 | MAPE: **1.44%**
    - The revenue model exhibited small errors during peak months, but that's a reasonable outcome given the higher volatility of revenue data.
    - In the residual plot, was slightly more sensitive which led to bias, suggesting the need for additional seasonal flexibility. However, it still adapted well over time.

**Conclusions:** 
- 🎯 The SARIMAX models successfully captured the monthly sales and revenue patterns by achieving strong accuracy on both metrics.
- 🔍 While the sales model performs well in adapting to recent surges, the revenue model still shows mild bias at peak periods due to its higher volatility.

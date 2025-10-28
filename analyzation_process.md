# Retail Sales Data Monthly Multivariate Forecasting
(project description)

## Key Points
- **Problem:** 
- **Proposed Solution:** 

## Raw Data
### Source
- [Kaggle: Synthetic Retail Sales Data 2022–2024](https://www.kaggle.com/datasets/shivamja/synthetic-retail-sales-data-20222024)
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

## Details of Analyzation Process
### EDA (Exploratory Data Analysis)
1. Does the presence of holiday affect overall sales and revenue, both daily and monthly?
    - **Proposed Solution:** Find correlation coefficients between variables (`is_holiday`, `net_units`, and `net_revenue`).
2. Is there any change in product's category trend during no-holiday months and holiday months?
    - **Proposed Solution:** Find mode of `category` for the specific dates.
3. Does the weekend status affect overall daily sales and revenue?
    - **Proposed Solution:** Find correlation coefficients between variables (`weekend`, `net_units`, and `net_revenue`).
4. How is the overall day-by-day sales and revenue trend during a week?
    - **Proposed Solution:** Find average values of sales (`net_units`) and revenue (`net_revenue`) for each day of the week (`day_of_week`).
5. Does the store type and area affect the customer experiences, which lead to store's sales and revenue?
    - **Proposed Solution:** Find average values of customer experiences (`avg_rating`), sales (`net_units`), and revenue (`net_revenue`) for each pair of store type (`store_type`). Also, find correlation coefficients between variables (`store_area_sqft`, `avg_rating`, `net_units`, and `net_revenue`).
6. Which category of product is the most popular in each city month-by-month?
    - **Proposed Solution:** Find mode of `category` for the specific dates and `city`.
7. How does discount percentages on products affect store's sales and revenue?
    - **Proposed Solution:** Find correlation coefficients between variables (`discount_pct`, `net_units`, and `net_revenue`).
8. How does product's promotion affect store's sales and revenue?
    - **Proposed Solution:** Find correlation coefficients between variables (`promotion`, `net_units`, and `net_revenue`).
9. Does the combination of discount and promotion gives different effect to store's sales and revenue?
    - **Proposed Solution:** Find correlation coefficients between variables (`promotion`, `net_units`, and `net_revenue`).
10. Does online transaction affect the customers experience (returns and rating)?
    - **Proposed Solution:** Find correlation coefficients between variables.
### Regression

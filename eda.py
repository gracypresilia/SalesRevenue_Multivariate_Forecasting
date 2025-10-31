# %% [markdown]
# # Import all packages/library.

# %%
import importlib
import helper
importlib.reload(helper)

# %%
from helper import *
import pandas as pd

# %% [markdown]
# # Data Preprocessing

# %% [markdown]
# Read the raw data file.

# %%
raw_df = pd.read_csv('./data/retail_sales_synthetic.csv')
df = raw_df.copy()  # Copy to ensure every change made in this code doesn't affect the raw data.

# %%
# It will be truncated if we printed it as df.head() or df.describe() so we can't check (see) all columns.
# In order to avoid that, we need to print it partially.
n_col = len(df.columns)

# %%
print('\nInitial Dataframe Head:')
print(df.head().iloc[:, :int(n_col/2)])
print(df.head().iloc[:, int(n_col/2):])

# %%
# Check the raw data information before preprocessing.
print('\nInitial Dataframe Information:')
print(df.info())

# %%
# Check the current data numeric stats before preprocessing.
print('\nInitial Dataframe Numeric Stats:')
print(df.describe().iloc[:, :int(n_col/3)])
print(df.describe().iloc[:, int(n_col/3):])

# %% [markdown]
# As shown above, the data is already clean and ready to use.

# %% [markdown]
# # EDA (Exploratory Data Analysis)

# %% [markdown]
# ## 1. Does the presence of holiday affect overall sales and revenue, both daily and monthly?

# %% [markdown]
# Get the necessary columns from the preprocessed data.

# %%
# Each row is confirmed unique from the previous step so we can exclude the ID columns in this section.
# Copy to ensure every change made in this section doesn't affect the main data.
df_1 = df[['date','is_holiday','net_units','net_revenue']].copy()
print(df_1.head())

# %% [markdown]
# ### Holiday Effect Towards Daily Sales and Revenue

# %% [markdown]
# Sum sales (units) and revenue of all stores and products each day.

# %%
df_1 = df_1.groupby(df_1['date']).sum()
# The line above sums all numeric data except `date` so `is_holiday` were also summed.
# However, `is_holiday` is conditional data and is better represented as binary (0 or 1) in a day-by-day data.
df_1.loc[df_1['is_holiday'] > 0, 'is_holiday'] = 1
print(df_1.head())

# %% [markdown]
# Find the correlation coefficients between variables (`is_holiday`, `net_units`, and `net_revenue`) to analyze the effect of holiday to daily sales and revenue.

# %%
corr_1_1 = df_1[['is_holiday', 'net_units', 'net_revenue']].corr()
print(corr_1_1)

# %% [markdown]
# Based on the results, the presence of holiday doesn't significantly affect the overall daily sales and revenue.

# %% [markdown]
# ### Holiday Effect Towards Monthly Sales and Revenue

# %% [markdown]
# Every dates in the current data are already unique so we can erase the day in the `date` data to support monthly-based analysis process.

# %%
df_1.index = dtm(df_1.index)
print(df_1.head())

# %% [markdown]
# Group (sum) the numeric data based on `date` to earn monthly net sales and revenue.

# %%
df_1 = df_1.groupby(df_1.index).sum()
# In this section, `is_holiday` is no longer conditional and is expected to be summed as a representation for total holiday-days in a month.
print(df_1.head())

# %% [markdown]
# Find the correlation coefficients between variables (`is_holiday`, `net_units`, and `net_revenue`).

# %%
corr_1_2 = df_1[['is_holiday', 'net_units', 'net_revenue']].corr()
print(corr_1_2)

# %% [markdown]
# Based on the results, the presence of holiday doesn't significantly affect the overall monthly sales and revenue. However, the effect shows more than the daily analysis in the previous section.

# %% [markdown]
# ## 2. Is there any change in product's category trend during no-holiday months and holiday months?

# %% [markdown]
# Get the necessary columns from the preprocessed data.

# %%
# Each row is confirmed unique from the previous step so we can exclude the ID columns in this section.
# Copy to ensure every change made in this section doesn't affect the main data.
df_2 = df[['date','is_holiday','category','net_units']].copy()
print(df_2.head())

# %% [markdown]
# Every dates in the current data are already uniquely paired with each category so we can erase the day in the `date` data to support monthly-based analysis process.

# %%
df_2['date'] = dtm(df_2['date'])
print(df_2.head())

# %% [markdown]
# Count `net_units` by month and product's category while keeping the `is_holiday` properties.

# %%
df_2 = df_2.groupby(['date', 'category'], as_index=False)[['is_holiday','net_units']].sum()
# The above line will sum `is_holiday` and `net_units` data based on unique pairs of `date` and `category`.
# However, in this section, we need `is_holiday` as a conditional data so it is better represented as binary (0 or 1).
df_2.loc[df_2['is_holiday'] > 0, 'is_holiday'] = 1
print(df_2.head())

# %% [markdown]
# Only returns the highest sales product's category for each month.

# %%
df_2 = df_2.loc[df_2.groupby('date')['net_units'].idxmax()]
print(df_2.head())

# %% [markdown]
# Only returns the mode of the highest sales product's category across all of the no-holiday months and all of the holiday months.

# %%
df_2 = df_2.groupby('is_holiday')['category'].agg(lambda x: x.mode()[0])
print(df_2.head())

# %% [markdown]
# As seen in the two latest dataframes, there's no change in product's category trend during no-holiday months and holiday months. Both product's category trends are clothing.

# %% [markdown]
# ## 3. Does the weekend status affect overall daily sales and revenue?

# %% [markdown]
# Get the necessary columns from the preprocessed data.

# %%
# Each row is confirmed unique from the previous step so we can exclude the ID columns in this section.
# Copy to ensure every change made in this section doesn't affect the main data.
df_3 = df[['date','weekend','net_units','net_revenue']].copy()
print(df_3.head())

# %% [markdown]
# Sum sales (units) and revenue of all stores and products each day.

# %%
df_3 = df_3.groupby(df_3['date']).sum()
# The line above sums all numeric data except `date` so `weekend` were also summed.
# However, `weekend` is conditional data and is better represented as binary (0 or 1) in a day-by-day data.
df_3.loc[df_3['weekend'] > 0, 'weekend'] = 1
print(df_3.head())

# %% [markdown]
# Find the correlation coefficients between variables (`weekend`, `net_units`, and `net_revenue`) to analyze the effect of weekend to daily sales and revenue.

# %%
corr_2 = df_3[['weekend', 'net_units', 'net_revenue']].corr()
print(corr_2)

# %% [markdown]
# Based on the results, the weekend status quite significantly affect the overall daily sales and revenue.

# %% [markdown]
# ## 4. How is the overall day-by-day sales and revenue trend during a week?

# %% [markdown]
# Get the necessary columns from the preprocessed data.

# %%
# Each row is confirmed unique from the previous step so we can exclude the ID columns in this section.
# Because this section analyze day or `day_of_week` instead of `date`, we can also exclide the `date` column.
# Copy to ensure every change made in this section doesn't affect the main data.
df_4 = df[['day_of_week','net_units','net_revenue']].copy()
print(df_4.head())

# %% [markdown]
# Find the average values of sales and revenue for each day of the week.

# %%
df_4 = df_4.groupby(df_4['day_of_week']).mean()
print(df_4)

# %% [markdown]
# The results show that sales and revenue are higher during weekends. This shows a consistent result between this section and previous section, the weekend status quite significantly affect the overall daily sales and revenue.

# %% [markdown]
# ## 5. Does the store type and area affect the customer experience, which lead to store's sales and revenue?

# %% [markdown]
# Get the necessary columns from the preprocessed data.

# %%
# Each row is confirmed unique from the previous step so we can exclude the `date` and ID columns in this section.
# Copy to ensure every change made in this section doesn't affect the main data.
df_5 = df[['store_type','store_area_sqft','avg_rating','net_units','net_revenue']].copy()
print(df_5.head())

# %% [markdown]
# Find the average values of customer experiences, sales, and revenue for each store type and area.

# %%
df_5 = df_5.groupby(['store_type', 'store_area_sqft'], as_index=False)[['avg_rating','net_units','net_revenue']].mean()
print(df_5)

# %% [markdown]
# ### Store Type Effect Towards Average Customer Experiences, Sales, and Revenue

# %% [markdown]
# Find the average values of customer experiences, sales, and revenue for each store type only.

# %%
df_5_type = df_5.drop(columns='store_area_sqft').copy()
df_5_type = df_5_type.groupby(df_5_type['store_type']).mean()
print(df_5_type)

# %% [markdown]
# The result above shows that store type, though it doesn't significantly affect customer experiences, quite significantly affect store's sales and revenue. Store type C has the highest rank, but store type A has the highest net sales and revenue.

# %% [markdown]
# ### Store Area (in sqft) Effect Towards Average Customer Experiences, Sales, and Revenue

# %% [markdown]
# Find the correlation coefficients between `store_area_sqft`, `avg_rating`, `net_units`, and `net_revenue`.

# %%
df_5_area = df_5[['store_area_sqft', 'avg_rating', 'net_units', 'net_revenue']].corr()
print(df_5_area)

# %% [markdown]
# The result above shows that store area doesn't significantly affect either customer experiences, sales, nor revenue. Interestingly, larger store areas show a negative correlation with ratings, sales, and revenue suggesting that bigger spaces might not necessarily improve customer satisfaction. In addition, surprisingly, customer experiences also doesn't significantly affect either sales nor revenue and is on negative correlation, which means a higher rating results to lower sales and revenue.

# %% [markdown]
# ## 6. Which category of product is the most popular in each city month-by-month?

# %% [markdown]
# Get the necessary columns from the preprocessed data.

# %%
# Each row is confirmed unique from the previous step so we can exclude the `date` and ID columns in this section.
# Copy to ensure every change made in this section doesn't affect the main data.
df_6 = df[['date','city','category','net_units']].copy()
print(df_6.head())

# %% [markdown]
# Every dates in the current data are already uniquely paired with each category so we can erase the day in the `date` data to support monthly-based analysis process.

# %%
df_6['date'] = dtm(df_6['date'])
print(df_6)

# %% [markdown]
# Count `net_units` by month and product's category while keeping the `city` data.

# %%
df_6 = df_6.groupby(['date', 'category','city'], as_index=False)[['net_units']].sum()
# The above line will sum `net_units` data based on unique pairs of `date`, `category`, and 'city'.
print(df_6.head())

# %% [markdown]
# Separate data by city to support city-based analysis process and to ensure every store carries equal weights.

# %%
city_dfs = {}
separate(df_6, city_dfs, 'city')

# %% [markdown]
# Returns the highest sales product's category for each pair of month and place.

# %%
for city, df_city in city_dfs.items():
    df_city = df_city.loc[df_city.groupby(['date'])['net_units'].idxmax()]
    city_dfs[city] = df_city
    print(city,':\n',df_city.head())

# %% [markdown]
# To simplify the pattern analyzation process, we can group by continuous segments with the same category as below.

# %%
for city, df_city in city_dfs.items():
    # df_city = series_group(df_city)
    df_city = series_group(df_city, 'category', 'net_units')
    print(city,':\n',df_city)

# %% [markdown]
# As shown above, each city displays different trends and preferences over time. However, the top product's category across those months and cities are always whether **Clothing** or **Home**.

# %% [markdown]
# ## 7. How does discount percentages on products affect store's sales and revenue?

# %% [markdown]
# Get the necessary columns from the preprocessed data.

# %%
# Each row is confirmed unique from the previous step so we can exclude the `date` in this section.
# Discount percentage applies to one specific product and the product is not always on discount.
# Therefore, we need to include product ID data and analyze the effect for each product
# Copy to ensure every change made in this section doesn't affect the main data.
df_7 = df[['product_id','discount_pct','net_units','net_revenue']].copy()
print(df_7.head())

# %% [markdown]
# Separate data by product ID to support product-based analysis process and to ensure every product carries equal weights.

# %%
disc_dfs = {}
separate(df_7, disc_dfs, 'product_id')

# %%
# Output examples.
print('prod_031:\n',disc_dfs['prod_031'].head())
print('prod_041:\n',disc_dfs['prod_041'].head())
print('prod_022:\n',disc_dfs['prod_022'].head())

# %% [markdown]
# Find the correlation coefficients between `discount_pct`, `net_units`, and `net_revenue` for each product.

# %%
for prod_id, df_disc in disc_dfs.items():
    df_disc = df_disc[['discount_pct', 'net_units', 'net_revenue']].corr()
    disc_dfs[prod_id] = df_disc
    # print('\n',prod_id,':\n',disc_dfs[prod_id])

# %% [markdown]
# Average correlation values to find general insights.

# %%
disc_mat = []

avg_disc_df = avg_corr(prod_id, df_disc, disc_dfs, disc_mat)
print(avg_disc_df)

# %% [markdown]
# As shown above, discount percentages slightly impact sales and revenue. The higher the discount percentages, the higher the sales is. However, it is inversely proportional to the revenue.

# %% [markdown]
# ## 8. How does product's promotion affect store's sales and revenue?

# %% [markdown]
# Get the necessary columns from the preprocessed data.

# %%
# Each row is confirmed unique from the previous step so we can exclude the `date` in this section.
# Promotion applies to one specific product and the product is not always on promotion.
# Therefore, we need to include product ID data and analyze the effect for each product
# Copy to ensure every change made in this section doesn't affect the main data.
df_8 = df[['product_id','promotion','net_units','net_revenue']].copy()
print(df_8.head())

# %% [markdown]
# Separate data by product ID to support product-based analysis process and to ensure every product carries equal weights.

# %%
promo_dfs = {}
separate(df_8, promo_dfs, 'product_id')

# %%
# Output examples.
print('prod_031:\n',promo_dfs['prod_031'].head())
print('prod_041:\n',promo_dfs['prod_041'].head())
print('prod_022:\n',promo_dfs['prod_022'].head())

# %% [markdown]
# Find the correlation coefficients between `promotion`, `net_units`, and `net_revenue` for each product.

# %%
for prod_id, df_promo in promo_dfs.items():
    df_promo = df_promo[['promotion', 'net_units', 'net_revenue']].corr()
    promo_dfs[prod_id] = df_promo
    # print(prod_id,':\n',df_promo)

# %% [markdown]
# Average correlation values to find general insights.

# %%
promo_mat = []

avg_promo_df = avg_corr(prod_id, df_promo, promo_dfs, promo_mat)
print(avg_promo_df)

# %% [markdown]
# As shown above, promotion affects sales and revenue. The impact is far greater than the impact of discount percentages. Moreover, promotion has directly proportional relations with both sales and revenue. The presence of promotion triggers higher sales and revenue.

# %% [markdown]
# ## 9. How does the combination of discount and promotion give different effect to store's sales and revenue?

# %% [markdown]
# Get the necessary columns from the preprocessed data.

# %%
# Each row is confirmed unique from the previous step so we can exclude the `date` in this section.
# Promotion applies to one specific product and the product is not always on promotion.
# Therefore, we need to include product ID data and analyze the effect for each product
# Copy to ensure every change made in this section doesn't affect the main data.
df_9 = df[['product_id','discount_pct','promotion','net_units','net_revenue']].copy()
print(df_9.head())

# %% [markdown]
# To analyze how the combination of discount and promotion gives different effect to store's sales and revenue, we need to pair each discount and promotion in categories.
# 
# From the data preprocessing section, we already knew that the discount percentage value ranges from 0-30%. We will group these value into four categories: `no_disc`, `low_disc`, `mid_disc`, and `high_disc`. Meanwhile, the promotion value represents in a binary condition so there are only two categories: `no_promo` and `promo`.

# %%
df_9['disc_group'] = pd.cut(df_9['discount_pct'], bins=[-1, 0, 10, 20, 30], labels=['no_disc', 'low_disc', 'mid_disc', 'high_disc'])
df_9['promo_group'] = df_9['promotion'].map({0: 'no_promo', 1: 'promo'})
print(df_9.head())

# %% [markdown]
# Separate data by product ID to support product-based analysis process and to ensure every product carries equal weights.

# %%
comb_dfs = {}
separate(df_9, comb_dfs, 'product_id')

for prod_id, df_comb in comb_dfs.items():
    df_comb = df_comb.drop(columns=['discount_pct','promotion'])
    # We will do a cross group analysis process for each product, averaging values of sales and revenue for each combination of discount and promotion.
    df_comb = df_comb.groupby(['disc_group', 'promo_group'], observed=True)[['net_units', 'net_revenue']].mean()
    comb_dfs[prod_id] = df_comb

# %%
# Output examples.
print('prod_031:\n',comb_dfs['prod_031'])
print('prod_041:\n',comb_dfs['prod_041'])
print('prod_022:\n',comb_dfs['prod_022'])

# %% [markdown]
# Average correlation values to find general insights.

# %%
comb_mat = []

avg_comb_df = cross_avg_mean(prod_id, df_comb, comb_dfs, comb_mat)
print(avg_comb_df)

# %% [markdown]
# As shown above, combination of discount and promotion gives different effect to sales and revenue. High discount with promotion results in highest sales, but no discount with promotion results in highest revenue. This affirm the two previous sections' result that promotion has greater positive impact to sales and revenue.

# %% [markdown]
# ## 10. Does online transaction affect the customer experience (returns and rating)?

# %% [markdown]
# Get the necessary columns from the preprocessed data.

# %%
# Each row is confirmed unique from the previous step so we can exclude the `date` in this section.
# Customer experience varies between stores.
# Therefore, we need to include store ID data to analyze online transaction effect to customer experience.
# Copy to ensure every change made in this section doesn't affect the main data.
df_10 = df[['store_id', 'online','returns','avg_rating']].copy()
print(df_10.head())

# %% [markdown]
# Separate data by store ID to support store-based analysis process and to ensure every store carries equal weights.

# %%
cx_dfs = {}
separate(df_10, cx_dfs, 'store_id')

# %%
# Output examples.
print('store_09:\n',cx_dfs['store_09'].head())
print('store_02:\n',cx_dfs['store_02'].head())
print('store_06:\n',cx_dfs['store_06'].head())

# %% [markdown]
# Find the correlation coefficients between `online`, `returns`, and `avg_rating` for each store.

# %%
for store_id, df_cx in cx_dfs.items():
    df_cx = df_cx[['online', 'returns', 'avg_rating']].corr()
    cx_dfs[store_id] = df_cx
    print(store_id,':\n',df_cx)

# %% [markdown]
# Average correlation values to find general insights.

# %%
cx_mat = []

avg_cx_df = avg_corr(store_id, df_cx, cx_dfs, cx_mat)
print(avg_cx_df)

# %% [markdown]
# As shown above, the correlation values is too low so we can say online transaction doesn't affect customer experience.



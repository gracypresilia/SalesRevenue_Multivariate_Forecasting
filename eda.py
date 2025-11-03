# %% [markdown]
# # Import All Packages/Library

# %%
import importlib
import helper
importlib.reload(helper)

# %%
from helper import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# %% [markdown]
# # Data Preprocessing

# %% [markdown]
# Read the raw data file.

# %%
raw_df = pd.read_csv('./data/retail_sales_synthetic.csv')
df = raw_df.copy()  # Copy to ensure every change made in this code doesn't affect the raw data.

# %%
# df.head()/df.describe() truncates columns.
# In order to inspect everything, we need to print in parts.
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
# Find the mean difference percentages between variables (`is_holiday`, `net_units`, and `net_revenue`) to analyze the effect of holiday to daily sales and revenue.

# %%
bmd_holiday_daily = binary_mean_diff(df_1, 'is_holiday', 'both')
print(f"Effect of holiday on daily sales: {bmd_holiday_daily['sales_pct_diff']:.2f}%")
print(f"Effect of holiday on daily revenue: {bmd_holiday_daily['rev_pct_diff']:.2f}%")

# %%
# Visualize results as barplot comparison.
fig, ax = plt.subplots(1, 2, figsize=(10,4))
plot_bar(df_1, 'is_holiday', 'net_units', 'Average Daily Sales', 'Sales', 'Not Holiday', 'Holiday', ax=ax[0])
plot_bar(df_1, 'is_holiday', 'net_revenue', 'Average Daily Revenue', 'Revenue', 'Not Holiday', 'Holiday', ax=ax[1])
plt.tight_layout()
plt.show()

# %% [markdown]
# Based on the results, holidays increase overall daily sales by 27,6% and revenue by 25,45%.

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
# Find the mean difference percentages between variables (`is_holiday`, `net_units`, and `net_revenue`) to analyze the effect of holiday to monthly sales and revenue.

# %%
bmd_holiday_monthly = binary_mean_diff(df_1, 'is_holiday', 'both')
print(f"Effect of holiday on monthly sales: {bmd_holiday_monthly['sales_pct_diff']:.2f}%")
print(f"Effect of holiday on monthly revenue: {bmd_holiday_monthly['rev_pct_diff']:.2f}%")

# %%
# Visualize results as barplot comparison.
fig, ax = plt.subplots(1, 2, figsize=(10,4))
plot_bar(df_1, 'is_holiday', 'net_units', 'Average Monthly Sales', 'Sales', 'Not Holiday', 'Holiday', ax=ax[0])
plot_bar(df_1, 'is_holiday', 'net_revenue', 'Average Monthly Revenue', 'Revenue', 'Not Holiday', 'Holiday', ax=ax[1])
plt.tight_layout()
plt.show()

# %% [markdown]
# Based on the results, holidays increase overall monthly sales and revenue by around 7%, less significant compared to its impact towards daily view.

# %% [markdown]
# ## 2. Is there any change in product category trend during no-holiday months and holiday months?

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
# Count `net_units` by month and product category while keeping the `is_holiday` properties.

# %%
df_2 = df_2.groupby(['date', 'category'], as_index=False)[['is_holiday','net_units']].sum()
# The above line will sum `is_holiday` and `net_units` data based on unique pairs of `date` and `category`.
# However, in this section, we need `is_holiday` as a conditional data so it is better represented as binary (0 or 1).
df_2.loc[df_2['is_holiday'] > 0, 'is_holiday'] = 1
print(df_2.head())

# %% [markdown]
# Only returns the highest sales product category for each month.

# %%
df_2 = df_2.loc[df_2.groupby('date')['net_units'].idxmax()]
print(df_2.head())

# %% [markdown]
# Return the mode of top categories across holiday and non-holiday months.

# %%
df_2 = df_2.groupby('is_holiday')['category'].agg(lambda x: x.mode()[0])
print(df_2.head())

# %% [markdown]
# As seen in the two latest dataframes, there's no change in product category trend during no-holiday months and holiday months. Both product category trends are clothing.

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
# Find the mean difference percentages between variables (`weekend`, `net_units`, and `net_revenue`) to analyze the effect of weekend to daily sales and revenue.

# %%
bmd_weekend = binary_mean_diff(df_3, 'weekend', 'both')
print(f"Effect of weekend on daily sales: {bmd_weekend['sales_pct_diff']:.2f}%")
print(f"Effect of weekend on daily revenue: {bmd_weekend['rev_pct_diff']:.2f}%")

# %%
# Visualize results as barplot comparison.
fig, ax = plt.subplots(1, 2, figsize=(10,4))
plot_bar(df_3, 'weekend', 'net_units', 'Average Weekend Sales', 'Sales', 'Weekday', 'Weekend', ax=ax[0])
plot_bar(df_3, 'weekend', 'net_revenue', 'Average Weekend Revenue', 'Revenue', 'Weekday', 'Weekend', ax=ax[1])
plt.tight_layout()
plt.show()

# %% [markdown]
# Based on the results, the weekend increase overall daily sales and revenue by around 22.7%, a big number but still less significant than holiday.

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

# %%
# Visualize results as barplot comparison.
fig, ax = plt.subplots(1, 2, figsize=(10,4))

sns.barplot(
        data=df, x='day_of_week', y='net_units', ax=ax[0],
        errorbar=None, legend=False,
        hue='day_of_week', palette='crest'
    )
ax[0].set_title('Average Sales of A Week')
ax[0].set_xlabel('Day of The Week')
ax[0].set_ylabel('Sales')
ax[0].set_xticks(
    [0, 1, 2, 3, 4, 5, 6], 
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    rotation=45
)

sns.barplot(
        data=df, x='day_of_week', y='net_revenue', ax=ax[1],
        errorbar=None, legend=False,
        hue='day_of_week', palette='crest'
    )
ax[1].set_title('Average Revenue of A Week')
ax[1].set_xlabel('Day of The Week')
ax[1].set_ylabel('Revenue')
ax[1].set_xticks(
    [0, 1, 2, 3, 4, 5, 6], 
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    rotation=45
)

plt.tight_layout()
plt.show()

# %% [markdown]
# The results show that sales and revenue are higher during weekends. This shows a consistent result between this section and previous section, the weekend status affects the overall daily sales and revenue.

# %% [markdown]
# ## 5. Does the store type and area affect customer experience and, in turn, store sales?

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

# %%
# Visualize results as barplot comparison.
fig, ax = plt.subplots(1, 3, figsize=(15,4))
plot_bar(df_5_type, 'store_type', 'avg_rating', 'Average Customer Ratings', xlabel='Store Type', ylabel='Customer Ratings', ax=ax[0])
plot_bar(df_5_type, 'store_type', 'net_units', 'Average Store Sales', xlabel='Store Type', ylabel='Sales', ax=ax[1])
plot_bar(df_5_type, 'store_type', 'net_revenue', 'Average Store Revenue', xlabel='Store Type', ylabel='Revenue', ax=ax[2])
plt.tight_layout()
plt.show()

# %% [markdown]
# Store type shows clear differences in sales and revenue, while customer ratings vary little by type. Store type C has the highest rank, but store type A has the highest net sales and revenue.

# %% [markdown]
# ### Store Area (in sqft) Effect Towards Average Customer Experiences, Sales, and Revenue

# %% [markdown]
# Find the correlation coefficients between `store_area_sqft`, `avg_rating`, `net_units`, and `net_revenue`.

# %%
df_5_area = df_5[['store_area_sqft', 'avg_rating', 'net_units', 'net_revenue']].corr()
print(df_5_area)

# %%
# Visualize results as heatmap correlation.
plt.figure(figsize=(5,4))
sns.heatmap(df_5_area, annot=True, cmap='crest', fmt=".2f")
plt.title('Correlation Heatmap')
plt.xticks(np.arange(df_5_area.shape[1]) + 0.5, ['Store Area (sqft)', 'Customer Ratings', 'Sales', 'Revenue'], rotation=45, ha='right', rotation_mode='anchor')
plt.yticks(np.arange(df_5_area.shape[0]) + 0.5, ['Store Area (sqft)', 'Customer Ratings', 'Sales', 'Revenue'], va='center')
plt.show()

# %% [markdown]
# The results above show that store area doesn't significantly affect customer experience, sales, or revenue. Interestingly, larger store areas exhibit a negative correlation with ratings, sales, and revenue, suggesting that bigger spaces might not necessarily lead to higher customer satisfaction. In addition, customer experience also doesn't significantly affect sales or revenue and even shows a negative correlation, meaning that higher ratings are associated with lower sales and revenue.

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
print(df_6.head())

# %% [markdown]
# Count `net_units` by month and product category while keeping the `city` data.

# %%
df_6 = df_6.groupby(['date', 'category','city'], as_index=False)[['net_units']].sum()
df_6['city'] = df_6['city'].str.replace(r'city_(\d)$', r'city_0\1', regex=True)
# The above line will sum `net_units` data based on unique pairs of `date`, `category`, and 'city'.
print(df_6.head())

# %% [markdown]
# Separate data by city to support city-based analysis process and to ensure every store carries equal weights.

# %%
city_dfs = {}
separate(df_6, city_dfs, 'city')

# %% [markdown]
# Returns the highest sales product category for each pair of month and place.

# %%
for city, df_city in city_dfs.items():
    df_city = df_city.loc[df_city.groupby(['date'])['net_units'].idxmax()]
    city_dfs[city] = df_city
    # print(city,':\n',df_city.head())

# %% [markdown]
# To simplify pattern analysis, we can group by continuous segments with the same category as below.

# %%
for city, df_city in city_dfs.items():
    df_city_series = series_group(df_city, 'category', 'net_units')
    print(city,':\n',df_city_series)

# %%
# Visualize results as heatmap correlation.
# Regroup dictionary to a DataFrame.
df_cities = dfs_to_df(city_dfs, 'city')
months = sorted(
    df_cities['date'].unique(),
    key=lambda s: pd.to_datetime(s + '-01', errors='coerce')
)

# Category mapping.
# Use complete category from initial data.
cats = list(pd.unique(df_6['category']))
cat_dtype = pd.api.types.CategoricalDtype(categories=cats, ordered=False)
df_cities['category'] = df_cities['category'].astype(cat_dtype)
df_cities['cat_code'] = df_cities['category'].cat.codes  

pivot = (
    df_cities
      .pivot(index='city', columns='date', values='cat_code')
      .reindex(columns=months)           
      .sort_index()
)
n_cats = len(cats)
n_city = len(pivot)

# Color palette settings.
main_palette = sns.color_palette('Set3', 12)
palette = [main_palette[i] for i in [1,0,2,4,7]]
cmap = mcolors.ListedColormap(palette)
bounds = np.arange(n_cats + 1) - 0.5
norm = mcolors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(15, 5))
ax = sns.heatmap(
    pivot, 
    cmap=cmap, norm=norm, cbar=True,
    linewidths=0.5, linecolor='white'
)
ax.set_title('Top Category per City (Month-by-Month)')
ax.set_xlabel('Month')
ax.set_ylabel('')

# Month label (xticks) settings.
for i, tick in enumerate(ax.get_xticklabels()):
    show_every = max(1, len(months)//12)
    tick.set_visible(i % show_every == 0)

# Legend settings.
cbar = ax.collections[0].colorbar
cbar.set_ticks(np.arange(n_cats))
cbar.set_ticklabels(cats)

# Ticks settings.
plt.xticks(rotation=45, ha='right')
plt.yticks(
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + 0.5,
    np.arange(n_city) + 0.5,
    ['City 1','City 2', 'City 3', 'City 4', 'City 5', 'City 6', 'City 7', 'City 8', 'City 9', 'City 10'],
    rotation='horizontal')

plt.tight_layout()
plt.show()

# %% [markdown]
# As shown above, each city shows different trends and preferences over time. However, the top product category across those months and cities are always whether **Clothing** or **Home**.

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

# %%
# Check number of unique discount percentages to determine analysis approach.
print(df_7[['discount_pct']].nunique())

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
# Calculate average sales and revenue for each level discount in each product.

# %%
for prod_id, df_disc in disc_dfs.items():
    df_disc = df_disc.groupby('discount_pct')[['net_units', 'net_revenue']].mean()
    disc_dfs[prod_id] = df_disc
    # print('\n',prod_id,':\n',disc_dfs[prod_id])

# %% [markdown]
# Find the mean difference percentages between variables (`discount_pct`, `net_units`, and `net_revenue`) to analyze the effect of discount to product sales and revenue.  
# **Note:** The mean difference percentages are compared to baseline.

# %%
disc_dfs_mdiff = disc_dfs.copy()
# print('Mean Difference Pencentages of Each Product:')
for prod_id, df_disc in disc_dfs_mdiff.items():
    df_disc = df_base_diff(df_disc)
    disc_dfs_mdiff[prod_id] = df_disc
    # print('\n',prod_id,':\n',disc_dfs_mdiff[prod_id])

# %%
# Visualize results as heatmap (Part 1).
# Regroup dictionary to a DataFrame.
df_discs = dfs_to_df(disc_dfs, 'prod_id', index_name='discount_pct')

print(df_discs.head())

# %%
# Visualize results as heatmap (Part 2).
pivot_sales = (
    df_discs
      .pivot(index='discount_pct', columns='prod_id', values='net_units')
)
pivot_rev = (
    df_discs
      .pivot(index='discount_pct', columns='prod_id', values='net_revenue')
)

fig, ax = plt.subplots(1, 2, figsize=(25,5))
# Sales Heatmap
sns.heatmap(
    pivot_sales, ax=ax[0],
    cmap='crest', 
    cbar=True, cbar_kws={'label': 'Sales'},
    linewidths=0.5, linecolor='white',
)
ax[0].set_title('Sales by Product and Discount')
ax[0].set_xlabel('Product ID')
ax[0].set_ylabel('Discount (%)')
# Ticks settings.
ax[0].set_xticks(np.arange(50) + 0.5)
ax[0].set_xticklabels(np.arange(1, 51), rotation='horizontal', ha='center')
ax[0].set_yticks(
    np.arange(len(pivot_sales.index)),
    pivot_sales.index,
    rotation='horizontal', va='center')

# Revenue Heatmap
sns.heatmap(
    pivot_rev, ax=ax[1],
    cmap='crest', 
    cbar=True, cbar_kws={'label': 'Revenue'},
    linewidths=0.5, linecolor='white'
)
ax[1].set_title('Revenue by Product and Discount')
ax[1].set_xlabel('Product ID')
ax[1].set_ylabel('Discount (%)')
# Ticks settings.
ax[1].set_xticks(np.arange(50) + 0.5)
ax[1].set_xticklabels(np.arange(1, 51), rotation='horizontal', ha='center')
ax[1].set_yticks(
    np.arange(len(pivot_rev.index)),
    pivot_rev.index,
    rotation='horizontal', va='center')

plt.tight_layout()
plt.show()

# %% [markdown]
# Average mean difference percentages to find general insights.

# %%
disc_mat = []

avg_disc_df = avg_dfitem(prod_id, df_disc, disc_dfs_mdiff, disc_mat)
print(avg_disc_df)

# %%
# Visualize results as barplot comparison.
fig, ax = plt.subplots(1, 2, figsize=(10,4))
plot_bar(avg_disc_df, 'discount_pct', 'net_units', 'Average Product Sales', 'Sales', ax=ax[0])
plot_bar(avg_disc_df, 'discount_pct', 'net_revenue', 'Average Product Revenue', 'Revenue', ax=ax[1])
plt.tight_layout()
plt.show()

# %% [markdown]
# As shown above, discount percentages slightly impact sales and revenue. The higher the discount percentages, the higher the sales is. However, by percentages, it is inversely proportional to the revenue, though by amount the impact doesn't show much.

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
# Find the mean difference percentages between variables (`promotion`, `net_units`, and `net_revenue`) to analyze the effect of promotion to product sales and revenue.

# %%
promo_dfs_mdiff = promo_dfs.copy()
print("Effect of Promotion to Each Product's Sales and Revenue by Mean Difference Percentages:")
for prod_id, df_promo in promo_dfs_mdiff.items():
    df_promo = binary_mean_diff(df_promo, 'promotion', 'both')
    df_promo = pd.DataFrame([df_promo])
    promo_dfs_mdiff[prod_id] = df_promo

promo_mdiffs = dfs_to_df(promo_dfs_mdiff, 'prod_id')
print(promo_mdiffs.head())

# %%
# Visualize results as stacked bar plot.
pivot_sales = (
    df_8.groupby(['product_id', 'promotion'])[['net_units']]
      .mean()
      .unstack(fill_value=0)
)
pivot_sales.columns = ['No Promotion', 'Promotion']

pivot_rev = (
    df_8.groupby(['product_id', 'promotion'])[['net_revenue']]
      .mean()
      .unstack(fill_value=0)
)
pivot_rev.columns = ['No Promotion', 'Promotion']

fig, ax = plt.subplots(1, 2, figsize=(25,5))

# Sales stacked bar plot.
plot_stackedbar(
    pivot_sales,
    'Sales by Product and Promotion', 
    'Product ID', 'Sales',
    50, ax=ax[0])

# Revenue stacked bar plot.
plot_stackedbar(
    pivot_rev,
    'Revenue by Product and Promotion', 
    'Product ID', 'Revenue',
    50, ax=ax[1])

plt.tight_layout()
plt.show()

# %% [markdown]
# Calculate average mean difference percentages for each product.

# %%
promo_mat = []

avg_promo_df = avg_dfitem(prod_id, df_promo, promo_dfs_mdiff, promo_mat)
print(f"Average effect of promotion on product sales: {avg_promo_df['sales_pct_diff'][0]:.2f}%")
print(f"Average effect of promotion on product revenue: {avg_promo_df['rev_pct_diff'][0]:.2f}%")

# %% [markdown]
# As shown above, promotion significantly affects sales and revenue. Its impact is far greater than that of discount percentages—around 67.8%. Moreover, promotion has a directly proportional relationship with both sales and revenue, where the presence of promotion consistently leads to higher values in both.

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
# Average mean values for each category to find general insights.

# %%
comb_mat = []

avg_comb_df = avg_dfitem(prod_id, df_comb, comb_dfs, comb_mat, mode='cross')
print(avg_comb_df)

# %%
# Visualize results as catplot.
fig, ax = plt.subplots(1, 2, figsize=(15,5))

# Sales catplot.
sns.barplot(
    avg_comb_df, ax=ax[0],
    x='disc_group', y='net_units', hue='promo_group',
    palette='crest'
)
ax[0].set_title('Sales by Discount and Promotion')
ax[0].set_xlabel('')
ax[0].set_ylabel('Sales')
ax[0].set_xticks([0, 1, 2, 3], ['No Discount (0%)', 'Low Discount (0-10%)', 'Mid Discount (10-20%)', 'High Discount (20-30%)'])
ax[0].legend(title='', labels=['No Promotion', 'Promotion'])
pad_ylim(ax[0])

# Revenue catplot.
sns.barplot(
    avg_comb_df, ax=ax[1],
    x='disc_group', y='net_revenue', hue='promo_group',
    palette='crest'
)
ax[1].set_title('Revenue by Discount and Promotion')
ax[1].set_xlabel('')
ax[1].set_ylabel('Revenue')
ax[1].set_xticks([0, 1, 2, 3], ['No Discount (0%)', 'Low Discount (0-10%)', 'Mid Discount (10-20%)', 'High Discount (20-30%)'])
ax[1].legend(title='', labels=['No Promotion', 'Promotion'])
pad_ylim(ax[1])

plt.tight_layout()
plt.show()

# %% [markdown]
# As shown above, the combination of discount and promotion produces varying effects on sales and revenue. A high discount combined with promotion results in the highest sales, while no discount with promotion yields the highest revenue. This finding affirms the results from the previous sections that promotion has a stronger positive impact on both sales and revenue.

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
# Find the mean difference percentages between variables (`online`, `returns`, and `avg_rating`) to analyze the effect of online transaction to customer experience.

# %%
cx_dfs_mdiff = cx_dfs.copy()
print("Effect of Online Transaction to Each Store's Customer Experience by Mean Difference Percentages:")
for store_id, df_cx in cx_dfs_mdiff.items():
    df_cx = {
        'returns': binary_mean_diff(df_cx, condition='online', mode='returns'), 
        'avg_rating': binary_mean_diff(df_cx, condition='online', mode='avg_rating')}
    df_cx = pd.DataFrame([df_cx])
    cx_dfs_mdiff[store_id] = df_cx

cx_mdiff = dfs_to_df(cx_dfs_mdiff, 'store_id')
print(cx_mdiff.head())

# %%
# Visualize results as stacked bar plot.
pivot_returns = (
    df_10.groupby(['store_id', 'online'])[['returns']]
      .mean()
      .unstack(fill_value=0)
)
pivot_returns.columns = ['Offline', 'Online']

pivot_ratings = (
    df_10.groupby(['store_id', 'online'])[['avg_rating']]
      .mean()
      .unstack(fill_value=0)
)
pivot_ratings.columns = ['Offline', 'Online']

fig, ax = plt.subplots(1, 2, figsize=(10,5))

# Returns stacked bar plot.
plot_stackedbar(
    pivot_returns, 
    'Returns by Store and Transaction Method',
    'Store ID', 'Returns',
    10, ax=ax[0])

# Ratings stacked bar plot.
plot_stackedbar(
    pivot_ratings, 
    'Ratings by Store and Transaction Method',
    'Store ID', 'Ratings',
    10, ax=ax[1])

plt.tight_layout()
plt.show()

# %% [markdown]
# Calculate average mean difference percentages for each store.

# %%
cx_mat = []

avg_cx_df = avg_dfitem(store_id, df_cx, cx_dfs_mdiff, cx_mat)
print(f"Average effect of online transaction on returns number: {avg_cx_df['returns'][0]:.2f}%")
print(f"Average effect of online transaction on ratings: {avg_cx_df['avg_rating'][0]:.2f}%")

# %% [markdown]
# As shown above, online transactions slightly affect both returns and ratings in a directly proportional manner. This is a surprising result, as an increase in returns usually corresponds to lower ratings.



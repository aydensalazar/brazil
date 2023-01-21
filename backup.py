# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

import geobr
import mapsmx as mp
import plotly.express as px 
import geopandas as gpd


# %%
data_path = 'data/'
customers_df = pd.read_csv(data_path + "olist_customers_dataset.csv")
geolocation_df = pd.read_csv(data_path + "olist_geolocation_dataset.csv")
order_items_df = pd.read_csv(data_path + "olist_order_items_dataset.csv")
order_payments_df = pd.read_csv(data_path + "olist_order_payments_dataset.csv")
order_reviews_df = pd.read_csv(data_path + "olist_order_reviews_dataset.csv")
orders_df = pd.read_csv(data_path + "olist_orders_dataset.csv")
products_df = pd.read_csv(data_path + "olist_products_dataset.csv")
sellers_df = pd.read_csv(data_path + "olist_sellers_dataset.csv")
product_category_name_translation_df = pd.read_csv(data_path + "product_category_name_translation.csv")

full_df = orders_df.merge(order_reviews_df, on='order_id')\
                   .merge(order_payments_df, on='order_id')\
                   .merge(customers_df, on='customer_id')\
                   .merge(order_items_df, on='order_id')\
                   .merge(products_df, on='product_id')\
                   .merge(sellers_df, on='seller_id')

full_df = full_df.drop_duplicates()

full_df.head()

# %%
full_df = full_df.assign(
    purchase_date = pd.to_datetime(full_df['order_purchase_timestamp']).dt.date,
    purchase_year = pd.to_datetime(full_df['order_purchase_timestamp']).dt.year,
    purchase_month = pd.to_datetime(full_df['order_purchase_timestamp']).dt.month,
    purchase_MMYYYY= pd.to_datetime(full_df['order_purchase_timestamp']).dt.strftime('%b-%y'),
    purchase_day = pd.to_datetime(full_df['order_purchase_timestamp']).dt.day_name(),
    purchase_hour = pd.to_datetime(full_df['order_purchase_timestamp']).dt.hour)
full_df.head()

# %%
# Create an empty dataframe to store the results
na_counts = pd.DataFrame(columns=['column_name', 'na_count'])

# Iterate through each column of full_df
for col in full_df.columns:
    # Count the number of NA values in the column
    na_count = full_df[col].isna().sum()
    # Append a new row to na_counts with the column name and na_count
    #na_counts = na_counts.append({'column_name': col, 'na_count': na_count}, ignore_index=True)
    insert_row = {
    "column_name": col,
    "na_count": na_count }
    na_counts = pd.concat([na_counts, pd.DataFrame([insert_row])])



na_counts = full_df.agg(lambda x: x.isna().sum()).reset_index()
na_counts.columns = ['column_name', 'na_count']

def highlight_NaN(x):
    return ['color:darkblue;background-color:pink' if v > 0 else '' for v in x]

na_counts = na_counts.style.apply(highlight_NaN, subset=pd.IndexSlice[:, ["na_count"]])

na_counts

# %%
# Grab the columns we want to focus on to draw scatterplots and histograms
filtered_df = full_df[['payment_value', 'price', 'freight_value', 'product_weight_g', 'review_score']].sample(500, random_state=69)

# Define the list of columns you want to calculate the IQR for
columns_to_remove_outliers = ['payment_value', 'price', 'freight_value', 'product_weight_g']

for column in columns_to_remove_outliers:
    # Calculate the quartiles of the column
    q1 = filtered_df[column].quantile(0.25)
    q3 = filtered_df[column].quantile(0.75)

    # Calculate the IQR
    iqr = q3 - q1

    # Calculate the lower and upper bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Remove rows that fall outside the IQR
    filtered_df = filtered_df[(filtered_df[column] >= lower_bound) & (filtered_df[column] <= upper_bound)]
# This code first defines a list of columns columns_to_remove_outliers that you want to calculate the IQR for. Then, it uses a for loop to iterate over the list of columns. For each column, it calculates the first (q1) and third quart


# Leverage Seaborn's PairGrid method to create a grid pattern for the plots
g = sns.PairGrid(filtered_df, hue="review_score", palette="Set2")
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()

# %%
full_df.columns

# %%
# Grab the columns we want to focus on to draw scatterplots and histograms
filtered_df = full_df[['product_weight_g', 'product_length_cm',
       'product_height_cm', 'product_width_cm', "review_score"]].sample(500, random_state=69)

# Define the list of columns you want to calculate the IQR for
columns_to_remove_outliers = ['product_weight_g', 'product_length_cm',
       'product_height_cm', 'product_width_cm']

for column in columns_to_remove_outliers:
    # Calculate the quartiles of the column
    q1 = filtered_df[column].quantile(0.25)
    q3 = filtered_df[column].quantile(0.75)

    # Calculate the IQR
    iqr = q3 - q1

    # Calculate the lower and upper bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Remove rows that fall outside the IQR
    filtered_df = filtered_df[(filtered_df[column] >= lower_bound) & (filtered_df[column] <= upper_bound)]
# This code first defines a list of columns columns_to_remove_outliers that you want to calculate the IQR for. Then, it uses a for loop to iterate over the list of columns. For each column, it calculates the first (q1) and third quart


# Leverage Seaborn's PairGrid method to create a grid pattern for the plots
g = sns.PairGrid(filtered_df, palette="Set2", hue="review_score")
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()

# %%
from datetime import datetime

#full_df = full_df.dropna()

full_df['order_delivered_customer_date'] = pd.to_datetime(full_df['order_delivered_customer_date'])#full_df['order_delivered_customer_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
full_df['order_estimated_delivery_date'] = pd.to_datetime(full_df['order_estimated_delivery_date'])
# full_df['elapsed_time_est_deliv'] = 
full_df['elapsed_deliv_estim_days'] = (full_df['order_estimated_delivery_date'] - full_df['order_delivered_customer_date']).dt.days


# %%
sns.histplot(full_df['elapsed_deliv_estim_days'])

# %%
full_df.columns

# %%
full_df.groupby('seller_id').size().sort_values(ascending=False)[0:10].to_frame().reset_index()['seller_id']

# %%
# Create a list of unique seller IDs
#seller_ids = full_df["seller_id"].unique()
seller_ids = full_df.groupby('seller_id').size().sort_values(ascending=False)[0:10].to_frame().reset_index()['seller_id']

# Set up the grid of histograms
fig, axes = plt.subplots(len(seller_ids)//2 + len(seller_ids) % 2, 2, figsize=(10,10))
axes = axes.ravel()

# Plot a histogram for each unique seller ID
for i, seller_id in enumerate(seller_ids):
    data = full_df[full_df["seller_id"] == seller_id]["elapsed_deliv_estim_days"]
    axes[i].hist(data, bins=20)
    axes[i].set_title(f"Seller ID: {seller_id}")

plt.tight_layout()
plt.show()

# %%
import plotly.express as px

# Assume the DataFrame is called 'full_df' and contains a column called 'state' with the state names
# and a column called 'review_value' with the review values

# Calculate the average review value for each state
state_reviews = full_df.groupby('seller_state')['review_score'].mean()

# Create a choropleth map
fig = px.choropleth(state_reviews,
                    locations='review_score',
                    locationmode='country names',
                    color='review_score',
                    title='Average Review Value by State in Brazil',
                    hover_name='review_score',
                    range_color=[state_reviews.min(), state_reviews.max()],
                    color_continuous_scale='Viridis')

fig.show()

# %%
states = geobr.read_state(year=2016)
# municipalities = geobr.read_municipality(year=2016)
# all_muni = geobr.read_municipality(code_muni="RJ", year=2016)
# sp_muni = geobr.read_municipality(code_muni="SP", year=2016)

# %%
#states["abbrev_state"] = states["abbrev_state"].str.lower()
customers_df_state=customers_df.groupby(by='customer_state')[['customer_id']].count().reset_index()
#customers_df_state['customer_state'] = customers_df_state['customer_state'].str.lower()

# join geo with data
states_n = states.merge(customers_df_state, left_on="abbrev_state", right_on="customer_state")
states_n=states_n.rename(columns={'customer_id':'Numbers'})
fig = px.choropleth(states_n, geojson=states_n['geometry'], 
                    locations=states_n.index, color="Numbers",
                    height=500,
                    hover_name='abbrev_state',
                    labels = {'name_state' : 'Numbers'}, 
                    title = 'Numbers of customers across states',
                   color_continuous_scale="Viridis")
fig.update_geos(fitbounds="locations", visible=True)
fig.update_layout(
    title={'text':'Numbers of customers across states',
            'x':0.5,
            'xanchor': 'center'}
)
fig.show()

# %%
customers_df_state

# %%
states

# %%
full_df.groupby('customer_state').mean()['review_score'].to_frame().reset_index()

# %%
#states["abbrev_state"] = states["abbrev_state"].str.lower()
avg_rating_df_state = full_df.groupby('customer_state').mean()['review_score'].to_frame().reset_index()
avg_rating_df_state = avg_rating_df_state.rename(columns={'review_score':'avg_review_score'})

#customers_df_state['customer_state'] = customers_df_state['customer_state'].str.lower()

# join geo with data
states_n = states.merge(avg_rating_df_state, left_on="abbrev_state", right_on="customer_state")
fig = px.choropleth(states_n, geojson=states_n['geometry'], 
                    locations=states_n.index, color="avg_review_score",
                    height=500,
                    hover_name='abbrev_state',
                    labels = {'name_state' : 'avg_review_score'}, 
                    title = 'Average Review Score For Each State',
                   color_continuous_scale="Viridis")
fig.update_geos(fitbounds="locations", visible=True)
fig.update_layout(
    title={'text':'Average Review Score For Each State',
            'x':0.5,
            'xanchor': 'center'}
)
fig.show()

# %%
full_df

# %%
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Create a sample DataFrame
df = pd.DataFrame({'seller_id': [1, 2, 3, 4], 'customer_id': [5, 6, 7, 8]})

# Create a Graph object
G = nx.Graph()

# Add edges to the graph
for i, row in df.iterrows():
    G.add_edge(row['seller_id'], row['customer_id'])

# Draw the graph
nx.draw(G, with_labels=True)
plt.show()

# %%
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Create a sample DataFrame
seller_list = full_df.groupby('seller_id').size().sort_values(ascending=False)[0:10].to_frame().reset_index()['seller_id'].to_list()
filtered_seller_df = full_df[full_df['seller_id'].isin(seller_list)]
sample_df = filtered_seller_df.sample(50, random_state=69)
# customer_list = full_df.groupby('customer_id').size().sort_values(ascending=False)[0:10].to_frame().reset_index()['customer_id'].to_list()
# filtered_customer_df = full_df[full_df['customer_id'].isin(customer_list)]
# sample_df = filtered_customer_df.sample(50, random_state=69)

# Create a Graph object
G = nx.Graph()

# Add edges to the graph
for i, row in sample_df.iterrows():
    G.add_edge(row['seller_id'], row['customer_id'])

colors = []
for node in G:
    if node in sample_df["seller_id"].values:
        colors.append("lightblue")
    else:
        colors.append("lightgreen")

# Draw the graph
nx.draw(G, with_labels=False, node_color=colors, node_size=100)
plt.show()

# %%
#seller_ids = full_df.groupby('seller_id').size().sort_values(ascending=False)[0:10].to_frame().reset_index()['seller_id']
customer_list = full_df.groupby('customer_id').size().sort_values(ascending=False)[0:10].to_frame().reset_index()['customer_id'].to_list()
filtered_customer_df = full_df[full_df['customer_id'].isin(customer_list)]
filtered_customer_df


# %%
full_df['product_category_name'].value_counts()

# %%
pivot_revenue =full_df.pivot_table(values = ['order_id', 'price']
                              , index=['purchase_year','purchase_month','purchase_MMYYYY']
                              , aggfunc={'order_id':'nunique','price':'sum'})
pivot_revenue = pivot_revenue.reset_index().iloc[3:23]                              
pivot_revenue

# %%
import seaborn as sns
import matplotlib.ticker as ticker

sns.set_theme(style="darkgrid")

# Plot the responses for different events and regions
g = sns.barplot(x="purchase_MMYYYY", y="price",
             data=pivot_revenue, width=0.5)
#g.set(yscale='log')
plt.xticks(rotation=45)
plt.ylabel("Revenue")
plt.xlabel("Month-Year")
plt.title("Revenue Per Month in 2017-2018")
g.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))


# %%
purchase_day = full_df.groupby("purchase_day").size().to_frame()

day_of_wk = ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday']
purchase_day = purchase_day.reindex(day_of_wk, axis=0).reset_index()

# from pandas.api.types import CategoricalDtype
# cat_type = CategoricalDtype(categories=cats, ordered=True)
# purchase_day['purchase_day'] = purchase_day['purchase_day'].astype(cat_type)




sns.set_theme(style="darkgrid")

# Plot the responses for different events and regions
g = sns.barplot(x="purchase_day", y=0,
             data=purchase_day, width=0.5)

plt.xlabel("Day of the Week")
plt.ylabel("Number of Orders")
plt.title("Total Number of Orders By Day")

# %% [markdown]
# Note:
# 
# Morning = 5:00 to before 12:00
# Afternoon = 12:00 to before 17:00
# Evening = 17:00 to before 21:00
# Night = 21:00 to before 4:00

# %%
# Note:
# Morning = 5:00 to before 12:00
# Afternoon = 12:00 to before 17:00
# Evening = 17:00 to before 21:00
# Night = 21:00 to before 4:00


# Create the bins for each time range
bins = [pd.to_datetime('0:00').hour, pd.to_datetime('5:00').hour, pd.to_datetime('12:00').hour, pd.to_datetime('17:00').hour, pd.to_datetime('21:00').hour, pd.to_datetime('23:59:59.999999').hour]

# Create the labels for each bin
labels = ['Night', 'Morning', 'Afternoon', 'Evening', 'Night']

# Convert the order_time column to datetime if it's not already
full_df['order_purchase_timestamp_hour'] = full_df['order_purchase_timestamp'].apply(lambda x: pd.to_datetime(x).hour)

# Create the new column with the binned time ranges
full_df['time_range'] = pd.cut(full_df['order_purchase_timestamp_hour'], bins=bins, labels=labels, ordered=False)

# %%
pivot_order =full_df.pivot_table(values = ['order_id']
                              , index=['purchase_day','time_range']
                              , aggfunc={'order_id':'nunique'})
pivot_order                            

# %%
pivot_order = pivot_order.reset_index().pivot("purchase_day", "time_range", "order_id")
# cat_type = CategoricalDtype(categories=cats, ordered=True)
# pivot_revenue['purchase_day'] = pivot_revenue['purchase_day'].astype(cat_type)

#.pivot("purchase_day", "time_range", "order_id")

pivot_order


# %%
day_of_wk = ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday']
time_of_day = ['Morning', 'Afternoon', 'Evening', 'Night']
pivot_order = pivot_order.reindex(day_of_wk, axis=0)
pivot_order = pivot_order.reindex(time_of_day, axis=1)
pivot_order

# %%
sns.heatmap(pivot_order, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), fmt='g')
plt.title("Number of Orders By Time For Each Day of the Week")
plt.xlabel("Time of Day")
plt.ylabel("Day of Week")

# %%
pivot_revenue_day_time = full_df.pivot_table(values = ['price']
                              , index=['purchase_day','time_range']
                              , aggfunc={'price':'sum'})

pivot_revenue_day_time = pivot_revenue_day_time.reset_index().pivot("purchase_day", "time_range", "price")

day_of_wk = ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday']
time_of_day = ['Morning', 'Afternoon', 'Evening', 'Night']
pivot_revenue_day_time = pivot_revenue_day_time.reindex(day_of_wk, axis=0)
pivot_revenue_day_time = pivot_revenue_day_time.reindex(time_of_day, axis=1)


sns.heatmap(pivot_revenue_day_time, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), fmt='g')
plt.title("Total Revenue By Time For Each Day of the Week (Dollars)")
plt.xlabel("Time of Day")
plt.ylabel("Day of Week")
 

# %%
most_pop_categories = full_df.groupby('product_category_name').size().to_frame().reset_index().sort_values(0, ascending=False).iloc[0:20, :]

sns.set_theme(style="darkgrid")

# Plot the responses for different events and regions
g = sns.barplot(x="product_category_name", y=0,
             data=most_pop_categories, width=0.5)

plt.xlabel("Product Name")
plt.ylabel("Number of Orders")
plt.title("Total Number of Orders By Product")
plt.xticks(rotation=90)

# %%


# %%
list_of_prod_cats = full_df.groupby('product_category_name').size().to_frame().reset_index().sort_values(0, ascending=False).iloc[0:20, :]['product_category_name'].to_numpy()

list_of_prod_cats = list_of_prod_cats[0]

for prod_cat in list_of_prod_cats:
    temp_df = full_df[full_df['product_category_name'] == prod_cat]
    pivot_prod_order = temp_df.pivot_table(values = ['order_id']
                              , index=['purchase_day','time_range', "product_category_name"]
                              , aggfunc={'order_id':'nunique'})

    pivot_prod_order = pivot_prod_order.reset_index().pivot("purchase_day", "time_range", "order_id")

    day_of_wk = ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday']
    time_of_day = ['Morning', 'Afternoon', 'Evening', 'Night']
    pivot_prod_order = pivot_prod_order.reindex(day_of_wk, axis=0)
    pivot_prod_order = pivot_prod_order.reindex(time_of_day, axis=1)


    sns.heatmap(pivot_prod_order, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), fmt='g')
    plt.title("Total Revenue By Time For Each Day of the Week (Dollars)")
    plt.xlabel("Time of Day")
    plt.ylabel("Day of Week")
    
                          

# %%
list_of_prod_cats = full_df.groupby('product_category_name').size().to_frame().reset_index().sort_values(0, ascending=False).iloc[0:20, :]['product_category_name'].to_numpy()

# %%




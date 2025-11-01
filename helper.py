import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def avg_dfitem(item, df, dfs, mat, mode=''):
    # Check if `dfs` is empty.
    if not dfs:
        raise ValueError('Please pass a non-empty dict of DataFrames (dfs).')
    
    for item, df in dfs.items():
        mat.append(df)

    stack = np.stack([c.values for c in mat])
    avg_data = stack.mean(axis=0)

    if mode=='cross':
        cross_group = next(iter(dfs.values()))
        avg_df = pd.DataFrame(avg_data, 
                                index=cross_group.index, 
                                columns=cross_group.columns)
    else:
        avg_df = pd.DataFrame(avg_data, 
                            index=df.index, 
                            columns=df.columns)

    return avg_df

def binary_mean_diff(df, condition, mode):
    mean_pos_sales = df.loc[df[condition] == 1, 'net_units'].mean() if mode=='sales' or mode=='both' else 0
    mean_neg_sales = df.loc[df[condition] == 0, 'net_units'].mean() if mode=='sales' or mode=='both' else 0
    pct_sales = (mean_pos_sales - mean_neg_sales) / mean_pos_sales * 100 if mode=='sales' or mode=='both' else 0

    mean_pos_rev = df.loc[df[condition] == 1, 'net_revenue'].mean() if mode=='revenue' or mode=='both' else 0
    mean_neg_rev = df.loc[df[condition] == 0, 'net_revenue'].mean() if mode=='revenue' or mode=='both' else 0
    pct_rev = (mean_pos_rev - mean_neg_rev) / mean_pos_rev * 100 if mode=='revenue' or mode=='both' else 0
    
    if mode=='sales':
        return pct_sales
    elif mode=='revenue':
        return pct_rev
    elif mode=='both':
        return {'sales_pct_diff': pct_sales, 'rev_pct_diff': pct_rev}
    else:
        mean_pos_mode = df.loc[df[condition] == 1, mode].mean()
        mean_neg_mode = df.loc[df[condition] == 0, mode].mean()
        pct_mode = (mean_pos_mode - mean_neg_mode) / mean_pos_mode * 100
        return pct_mode

def df_base_diff(df):
    baseline = df.loc[0.0]
    df_diff = df - baseline
    pct_diff = df_diff / baseline * 100
    return pct_diff

def dfs_to_df(dfs, cat, index_name=None):    
    if index_name!=None:
        for item, df in dfs.items():
            df = df.reset_index().rename(columns={'index': index_name})
            dfs[item] = df
    df_group = pd.concat(dfs, names=[cat]).reset_index(level=0).reset_index(drop=True)
    return df_group

def dtm(date):
    date = date.astype(str)
    date = date.str.replace(r'-\d{2}$', '', regex=True)
    return date

def plot_bar(df, x, y, title, xlabel=None, ylabel=None, xtick1=None, xtick2=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4))
    sns.barplot(
        data=df, x=x, y=y, ax=ax, estimator='mean',
        errorbar=None, legend=False,
        hue=x, palette='crest'
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # For plot with binary x label.
    if xtick1!=None and xtick2!=None:
        ax.set_xlabel('')
        ax.set_xticks([0, 1], [xtick1, xtick2])
    return ax

def separate(df, dfs, category):
    # Sort data first for neat visuals.
    df = df.sort_values(category).reset_index(drop=True)

    for item in df[category].unique():
        df_item = df[df[category] == item].drop(columns=category)
        dfs[item] = df_item

def series_group(df, category, item):
    # Ensure the dates are sorted,
    df = df.sort_values('date').reset_index(drop=True)

    # Add column to flag group and group data.
    df['group'] = (df[category] != df[category].shift()).cumsum()
    df_group = (
        df.groupby(['group', category]).agg(
            start_date=('date', 'first'),
            end_date=('date', 'last'),
            total_item=(item, 'sum')
        )
        .reset_index()
    )

    df_group['date'] = df_group['start_date'] + ' - ' + df_group['end_date']
    df_group = df_group[['date', category, 'total_item']]
    return df_group
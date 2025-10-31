import numpy as np
import pandas as pd

def avg_corr(item, df, dfs, mat):
    for item, df in dfs.items():
        mat.append(df)

    stack = np.stack([c.values for c in mat])
    avg_corr = stack.mean(axis=0)

    avg_df = pd.DataFrame(avg_corr, 
                            index=df.columns, 
                            columns=df.columns)

    return avg_df

def cross_avg_mean(item, df, dfs, mat):
    for item, df in dfs.items():
        mat.append(df)

    stack = np.stack([c.values for c in mat])
    avg_mean = stack.mean(axis=0)

    cross_group = next(iter(dfs.values()))
    avg_df = pd.DataFrame(avg_mean, 
                            index=cross_group.index, 
                            columns=cross_group.columns)
    
    return avg_df

def dtm(date):
    date = date.astype(str)
    date = date.str.replace(r'-\d{2}$', '', regex=True)
    return date

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
import pandas as pd

def dtm(df, col):
    """
    Convert YYYY-MM-DD object data-type date format to YYYY-MM.
    This function is dedicated for date in a dataframe.
    """
    # if not isinstance(date, pd.Series):
    #     date = pd.Series(date)
    # date = date.astype(str)
    # result = date.str.replace(r'-\d{2}$', '', regex=True)
    # return pd.Series(result, index=date.index)
    df[col] = df[col].astype(str).str.replace(r'-\d{2}$', '', regex=True)
    return df
import pandas as pd
import numpy as np

# import tabulate as tab

# import pdb

# import os
# import inspect
# try:
#     currpath = os.path.dirname(os.path.abspath(__file__))
# except NameError:
#     currpath = os.path.dirname(
#         os.path.abspath(inspect.getfile(inspect.currentframe())))
#     rootpath = os.path.dirname(os.path.dirname(currpath))
#     datapath = rootpath + '\\tests\\data\\'


def _count(df, row, col, weight, missing, margins=True):
    '''
    Count total number of observations by variable (or by two variables),
    given policy on missing data and survey weights.
    '''
    df = df.copy()

    # create dropna for functions
    dropna = (not missing)
    '''
    Note that certain aggregation techniques (groupby, pivot_table, etc.)
    are not able to treat missings as normal values when they are grouped
    by a categorical variable. To see this, try running:
    >df.groupby(row, dropna=False).size()
    For a row variable that is (1) categorical (2) has missing values, and
    (3) missing values are not in the cateogries. Below is a workaround for
    these cases.
    '''
    if missing:
        # current bug; need to add missing dummy to category first
        if df[row].dtype.name == 'category':
            df[row] = df[row].cat.add_categories(['missing_dummy'])
            df[row] = df[row].fillna('missing_dummy')
        # df[row] = df[row].fillna('missing_dummy')
        if col is not None:
            if df[col].dtype.name == 'category':
                df[col] = df[col].cat.add_categories(['missing_dummy'])
                df[col] = df[col].fillna('missing_dummy')
            # df[col] = df[col].fillna('missing_dummy')

    group_cols = [row]
    if col is not None:
        group_cols.append(col)

    grouper = df.groupby(group_cols, dropna=dropna)
    if col is None:
        if weight is None:
            n_hat_rc = grouper.size()
        else:
            n_hat_rc = grouper[weight].sum()
    elif col is not None:
        if weight is None:
            n_hat_rc = grouper.size().unstack(level=1)
        else:
            n_hat_rc = grouper[weight].sum().unstack(level=1)

    # if keeping missing values, replace dummies here
    if missing:
        # pdb.set_trace()
        n_hat_rc = (n_hat_rc.reset_index().replace('missing_dummy', np.nan)
                    .set_index(row)
                    .squeeze()  # squeeze since this could make series DF
                    )
        if col is not None:
            n_hat_rc.rename(columns={'missing_dummy': np.nan}, inplace=True)

    if margins:
        # Very weird error when I run n_hat_rc.loc['All']=n_hat_rc.sum(axis=0)
        if type(n_hat_rc) is pd.Series:
            append_val = pd.Series(n_hat_rc.sum(axis=0), index=['All'])
        elif type(n_hat_rc) is pd.DataFrame:
            append_val = (n_hat_rc.sum(axis=0)
                          .to_frame(name='All')
                          .unstack().unstack())
        n_hat_rc = pd.concat([n_hat_rc, append_val], axis=0)
        if col is not None:
            n_hat_rc.loc[:, 'All'] = n_hat_rc.sum(axis=1)

    return n_hat_rc

import pandas as pd
import numpy as np

# import tabulate as tab
import pdb


def _apply_group_func(df, group_var, var, weight, func, missing=False):
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
        if df[group_var].dtype.name == 'category':
            df[group_var] = df[group_var].cat.add_categories(['missing_dummy'])
            df[group_var] = df[group_var].fillna('missing_dummy')
    grouper = df.groupby(group_var, dropna=dropna)
    n_hat = grouper.apply(func)
    # if keeping missing values, replace dummies here
    # check that this command is in line with the _count command below
    if missing:
        n_hat.rename(index={'missing_dummy': np.nan}, inplace=True)
    return n_hat


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
        # Replace the missing dummy for the row
        n_hat_rc.rename(index={'missing_dummy': np.nan}, inplace=True)
        # This workaround might be necessary for the row rename, but I
        # forgot to document why.
        # n_hat_rc = (n_hat_rc.reset_index().replace('missing_dummy', np.nan)
        #             .set_index(row)
        #             .squeeze()  # squeeze since this could make series DF
        #             )
        # Replace the missing dummy for the column(s)
        if col is not None:
            n_hat_rc.rename(columns={'missing_dummy': np.nan}, inplace=True)

    if margins:
        # Very weird error when I run n_hat_rc.loc['All']=n_hat_rc.sum(axis=0)
        if type(n_hat_rc) is pd.Series:
            append_val = pd.Series(n_hat_rc.sum(axis=0), index=['All'])
        elif type(n_hat_rc) is pd.DataFrame:
            append_val = (n_hat_rc.sum(axis=0)
                          .to_frame(name='All')
                          .transpose())
        n_hat_rc = pd.concat([n_hat_rc, append_val], axis=0)
        if col is not None:
            # if columns are categorical, this step encounters a TypeError
            if n_hat_rc.columns.dtype.name == 'category':
                n_hat_rc.columns = n_hat_rc.columns.add_categories(['All'])
            # if multiindex (like when calucating BRR std. errors), there
            # need to sum differently
            if n_hat_rc.columns.nlevels == 1:
                n_hat_rc.loc[:, 'All'] = n_hat_rc.sum(axis=1)
            else:
                n_hat_rc_all = n_hat_rc.sum(axis=1, level=0)
                # Set new column level with totals
                n_hat_rc_all.columns = (pd.MultiIndex.from_product(
                    [n_hat_rc_all.columns, ['All']]))
                n_hat_rc = pd.concat([n_hat_rc, n_hat_rc_all], axis=1)

    return n_hat_rc

import pandas as pd
import numpy as np

import pdb

from _agg import _count

import _descriptive_stats


def _se(
    df, var, weight, theta='count',
    vce='brr', brrweight=None, mse=True, **kwargs
):
    theta_hat, brr_var = _var(
        df=df, var=var, weight=weight, theta=theta,
        vce=vce, brrweight=brrweight, mse=mse, **kwargs
    )
    brr_se = np.sqrt(brr_var)
    return theta_hat, brr_se


def _var(
    df, var, weight, theta='count',
    vce='brr', brrweight=None, mse=True, **kwargs
):
    '''
    Calculate the variance of a dataframe using BRR errors.

    Required:

        * df:
        * var:
        * weight: survey weight
        * vce: currently only allows 'brr'
        * brrweight: vector of BRR weights
        * mse: Calculate SE using MSE; currently only True

    Optional:

        * theta: point estimate to calculate variance for. Currently
                 only possible for 'count'; 'mean'; and 'proportion'
    '''

    if vce == 'brr':
        if brrweight is not None:
            pass
        else:
            print('Pass BRR weights')
            return
    else:
        print('Only have capability for vce=brr')
        return

    # theta_hat is the vector of point estimates computed using sampling
    # weights for a given stratified survey design
    # For total variables, we use the '_count' function
    if ((theta == 'count') | (theta == 'proportion')):
        if type(var) is list:
            assert type(var) == list, 'Please pass var = [row, col]'
            assert len(var) == 2, 'Please pass var = [row, col]'
            row, col = var
        else:
            row, col = var, None
        if 'missing' not in kwargs:
            # default is to ignore missing
            missing = False
        else:
            missing = kwargs['missing']

        theta_hat = _count(
            df=df, row=row, col=col, weight=weight,
            missing=missing, margins=True)
        # theta_hat_b is the vector of point estimates from the ith
        # replication
        # pdb.set_trace()
        theta_hat_b = _count(
            df=df, row=row, col=col, weight=brrweight,
            missing=missing, margins=True)
        if theta == 'proportion':
            # If doing a twoway tabulate, scale by total obs
            if type(var) is list:
                tot_idx = ('All', 'All')
                tot_idx_b = ('All', pd.IndexSlice[:, 'All'])
            else:
                tot_idx = 'All'
                tot_idx_b = 'All'
            theta_hat_all = theta_hat.loc[tot_idx]
            theta_hat = (theta_hat / theta_hat_all)

            theta_hat_all_b = theta_hat_b.loc[tot_idx_b]
            # if a twoway tabulate, drop the level to better broadcast
            if col is not None:
                theta_hat_all_b = theta_hat_b.loc[tot_idx_b].droplevel(col)
                theta_hat_b = theta_hat_b.div(theta_hat_all_b, axis=1, level=0)
            else:
                theta_hat_all_b = theta_hat_b.loc[tot_idx_b]
                theta_hat_b = (theta_hat_b / theta_hat_all_b)

        # broadcasting on columns breaks with np.nan as a name
        if missing:
            # To rename things, you can try renaming as a dictionary
            # Sometimes this runs into issues though; because in pandas,
            # np.nan == np.nan returns False, so dictionary lookup may not
            # work.
            # pdb.set_trace()
            theta_hat_b.rename(columns={np.nan: 'NA'}, inplace=True)
            if type(theta_hat) is pd.DataFrame:
                # theta_hat.rename(columns={np.nan: 'NA'}, inplace=True)
                theta_hat.columns = theta_hat.columns.fillna('NA')
            # to keep things easier, also rename index. But code should work
            # if you only rename the columns
            theta_hat_b.index = theta_hat_b.index.fillna('NA')
            theta_hat.index = theta_hat.index.fillna('NA')
            # theta_hat.rename(index={np.nan: 'NA'}, inplace=True)
            # theta_hat_b.rename(index={np.nan: 'NA'}, inplace=True)
            # Need to change fill_value in pd.DataFrame.sub from default None
            # to 0 for variance to be correctly calculated.
            # Note: need to check this; if this dataframe is a dataframe
            # of brr values for a series, fill_value breaks sum
            # fill_value = 0
            fill_value = None
        else:
            fill_value = None
    elif theta == 'mean':
        # over can be used to calculate mean over different values
        if 'over' not in kwargs:
            # default is to calculate mean over the whole dataset:
            over = None
        else:
            over = kwargs['over']
        # missing treats missing like other values; really only relevant if
        # calculating mean over something
        if 'missing' not in kwargs:
            # default is to ignore missing
            missing = False
        else:
            missing = kwargs['missing']
        theta_hat = _descriptive_stats._w_mean(
            df=df, var=var, weight=weight,
            over=over, missing=missing)
        theta_hat_b = _descriptive_stats._w_mean(
            df=df, var=var, weight=brrweight,
            over=over, missing=missing)
        fill_value = None
    # pdb.set_trace()
    # Rename the brr values in theta_hat_b
    # The BRR values will be series if calculating SE for a single point
    # estimate (like for the mean of a variable)
    if type(theta_hat_b) == pd.Series:
        theta_hat_b.name = 'brr'
    elif type(theta_hat_b) == pd.DataFrame:
        # The BRR values will be the columns in a dataframe if calculating
        # the value counts for a single variable, like in a one way tabulate
        if theta_hat_b.columns.nlevels == 1:
            theta_hat_b.columns.rename('brr', inplace=True)
        # The BRR values will be the first level of columns in a dataframe
        # with a two level column index if calculatingthe value counts for
        # a singlevariable, like in a one way tabulate
        elif theta_hat_b.columns.nlevels == 2:
            theta_hat_b.columns.rename('brr', level=0, inplace=True)

    if mse:
        # Calculate MSE (theta_hat - theta_hat_b)**2 for each BRR value
        mse = theta_hat_b.sub(theta_hat, axis=0, fill_value=fill_value) ** 2
        # Appropriate way to sum MSE depends on data type (similar to
        # renaming BRR values above).
        # MSE are a series if theta is a single point estimate; then BRR
        # values are a series
        if type(theta_hat_b) == pd.Series:
            mse_sum = mse.sum()
        elif type(theta_hat_b) == pd.DataFrame:
            # MSE are a dataframe with a single level of columns if theta
            # is a series, like in a one way tabulate
            if theta_hat_b.columns.nlevels == 1:
                mse_sum = mse.sum(axis=1)
            # MSE are a dataframe with two levels of columns if theta is a
            # dataframe, like in a twoway tabulates
            else:
                mse_sum = mse.sum(axis=1, level=1)
        brr_var = (mse_sum / len(brrweight)).squeeze()
        # If brr_var is a number if calculating a point estimate for a
        # single variable. Otherwise it's a series or dataframe that
        # should have a name
        if not isinstance(brr_var, (int, float)):
            brr_var.name = 'Var'
    else:
        print('Only have MSE capability')
        return
    return theta_hat, brr_var


def _se_old(df, row, col, weight, missing,
        theta='count',
        vce='brr', brrweight=None, mse=True,
        floatfmt='{:,.0f}', printtab=False):
    '''
    Produces a dataframe with a point estimate and standard errors,
    where standard errors are calculated using the BRR methods.
    '''

    # if vce == 'brr':
    #     if brrweight is not None:
    #         pass
    #     else:
    #         print('Pass BRR weights')
    #         return
    # else:
    #     print('Only have capability for vce=brr')
    #     return

    # # theta_hat is the vector of point estimates computed using sampling
    # # weights for a given stratified survey design
    # # For total variables, we use the '_count' function
    # if theta == 'count':
    #     theta_hat = _count(
    #         df=df, row=row, col=col, weight=weight,
    #         missing=missing)
    #     # theta_hat_b is the vector of point estimates from the ith
    #     # replication
    #     theta_hat_b = _count(
    #         df=df, row=row, col=col, weight=brrweight,
    #         missing=missing)
    # elif theta == 'mean':
    #     pass
    # theta_hat_b.columns.rename('brr', level=0, inplace=True)

    # # broadcasting on columns breaks with np.nan as a name
    # if missing:
    #     theta_hat_b.rename(columns={np.nan: 'NA'}, inplace=True)
    #     theta_hat.rename(columns={np.nan: 'NA'}, inplace=True)
    #     # to keep things easier, also rename index. But code should work
    #     # if you only rename the columns
    #     theta_hat_b.rename(index={np.nan: 'NA'}, inplace=True)
    #     theta_hat.rename(index={np.nan: 'NA'}, inplace=True)
    #     # Need to change fill_value in pd.DataFrame.sub from default None
    #     # to 0 for variance to be correctly calculated.
    #     fill_value = 0
    # else:
    #     fill_value = None

    # if mse:
    #     brr_var = (
    #         (theta_hat_b.sub(theta_hat, axis=0, fill_value=fill_value) ** 2)
    #         .sum(axis='columns', level=1) / len(brrweight)
    #     ).squeeze()
    #     brr_var.name = 'Var'
    # else:
    #     print('Only have MSE capability')
    #     return
    # brr_se = np.sqrt(brr_var)
    # brr_se.name = 'SE'

    # if missing:
    #     theta_hat_b.rename(columns={'Missing': np.nan}, inplace=True)
    #     theta_hat.rename(columns={'Missing': np.nan}, inplace=True)

    if printtab:
        theta_hat = _fmt_var(theta_hat, floatfmt)
        brr_se = _fmt_var(brr_se, '(' + floatfmt + ')')

    newtab = pd.concat(
        [
            theta_hat,
            brr_se
        ], axis=0,
        keys=['Total', 'SE'], names=['stat', row])

    newtab = newtab.swaplevel().sort_index(axis=0, ascending=[True, False])

    if printtab:
        print_idx = pd.Index(
            [i[0] if i[1] == 'Total' else '' for i in list(newtab.index)]
        )
        newtab.index = print_idx

    return newtab

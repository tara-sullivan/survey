import pandas as pd
import numpy as np

# from make_bps_df import bps_derived
import tabulate as tab

import pdb

import os
import inspect
try:
    currpath = os.path.dirname(os.path.abspath(__file__))
except NameError:
    currpath = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))
    rootpath = os.path.dirname(os.path.dirname(currpath))
    datapath = rootpath + '\\tests\\data\\'


def tabulate(
    data, row, col=None, weight=None, *,
    count=False, missing=False, aggfunc=np.sum,
    se=False,
    vce='brr', brrweight=None, mse=True,
    printtab=False, floatfmt=',.3f'
):
    '''
    Two-way tables for survey data. Replicates Stata svy: tabulate twoway.

    * data: dataframe
    * row: row of tabulate

    Optional:
    * col: col of tabulate (for twoway tabulate)
    * weight: pweight (default = 1)
    * count: include count
    * missing: treat missing values like other values
    '''

    df = data.copy()
    # print('program started!')
    n_hat_rc = _agg(df=df, row=row, col=col,
                    weight=weight, missing=missing)

    if count:
        returntab = n_hat_rc
        # if printtab:
        #     print(tab.tabulate(
        #         n_hat_rc,
        #         headers="keys", tablefmt='psql',
        #         floatfmt=floatfmt, numalign='right'
        #     ))
        # try:
        #     return n_hat_rc.squeeze()
        # except:
        #     return n_hat_rc
        # return n_hat_rc.squeeze()
    else:
        tot_idx = 'All'
        n_hat = n_hat_rc.loc[tot_idx]
        p_hat_rc = n_hat_rc / n_hat
        returntab = p_hat_rc

    if se:
        returntab = _se(
            df=df, row=row, col=col, weight=weight, missing=missing,
            aggfunc=aggfunc,
            vce='brr', brrweight=brrweight, mse=mse,
            floatfmt='{:' + floatfmt + '}', printtab=printtab
        )

    if printtab:
        if type(returntab) is pd.Series:
            returntab_toprint = returntab.to_frame(name='Total')
        else:
            returntab_toprint = returntab
        print(tab.tabulate(
            returntab_toprint,
            headers="keys", tablefmt='psql',
            floatfmt=floatfmt, numalign='right'
        ))

    return returntab


def _agg(df, row, col, weight, missing, aggfunc=np.sum):
    '''
    Count total number of observations by variable (or by two variables),
    given policy on missing data and survey weights.
    '''
    # if only tabulating one column, format appropriately to use the same
    # pivot_table command you would use when making a crosstab
    if col is None:
        df['my_col'] = 1
        col_drop = True
        col = 'my_col'
    else:
        col_drop = False

    # If no weight is provided, we want to count all observations
    # To do this for one or two variables using pivot table, I can create a
    # variable = 1 to sum over.
    if weight is None:
        # If including missing values, this weight variable should be =1
        # for all observations
        if missing:
            df['my_weight'] = 1
        # If I'm not calculating missing values, this value should only equal
        # 1 if if the row (index) is not missing
        else:
            notna_idx = (df[row].notna() & df[col].notna())
            df.loc[notna_idx, 'my_weight'] = 1
        weight = 'my_weight'

    # to preserve missing, need to use workaround from pandas #3729
    # TODO: check if this can be fixed using observed pivot_table option
    if missing:
        # current bug; need to add missing dummy to category first
        if df[row].dtype.name == 'category':
            df[row] = df[row].cat.add_categories(['missing_dummy'])
        df[row] = df[row].fillna('missing_dummy')
        if not col_drop:
            if df[col].dtype.name == 'category':
                df[col] = df[col].cat.add_categories(['missing_dummy'])
            df[col] = df[col].fillna('missing_dummy')

    n_hat_rc = (pd.pivot_table(
        data=df, values=weight, index=row, columns=col, dropna=False,
        aggfunc=aggfunc, margins=True)
    )

    # if keeping missing values, replace dummies here
    if missing:
        n_hat_rc = (n_hat_rc.reset_index().replace('missing_dummy', np.nan)
                    .set_index(row))
        if not col_drop:
            n_hat_rc.rename(columns={'missing_dummy': np.nan}, inplace=True)

    # if only tabulating one variable, using pivot_table with margins on
    # creates a column named 1 and a column total. We want to drop the col
    # named one
    if col_drop:
        # if there are multiple values passed (i.e. multiple weights), there
        # are two levels to the column index. drop the outermost one
        if n_hat_rc.columns.nlevels > 1:
            n_hat_rc.drop(columns=1, level=-1, inplace=True)
        else:
            n_hat_rc.drop(columns=1, inplace=True)

    return n_hat_rc.squeeze()


# TODO: make this better
# If there's no column, and the types are series:
def _fmt_var(var, fmt):
    newvar = var.copy()

    if type(newvar) is pd.DataFrame:
        return newvar.applymap(fmt.format)
    if type(newvar) is pd.Series:
        return newvar.map(fmt.format)


def _se(df, row, col, weight, missing,
        aggfunc=np.sum,
        vce='brr', brrweight=None, mse=True,
        floatfmt='{:,.0f}', printtab=False):

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
    theta_hat = _agg(
        df=df, row=row, col=col, weight=weight,
        missing=missing, aggfunc=aggfunc)
    # theta_hat_b is the vector of point estimates from the ith
    # replication
    theta_hat_b = _agg(
        df=df, row=row, col=col, weight=brrweight,
        missing=missing, aggfunc=aggfunc)
    theta_hat_b.columns.rename('brr', level=0, inplace=True)

    # broadcasting on columns breaks with np.nan as a name
    if missing:
        theta_hat_b.rename(columns={np.nan: 'NA'}, inplace=True)
        theta_hat.rename(columns={np.nan: 'NA'}, inplace=True)
        # to keep things easier, also rename index. But code should work
        # if you only rename the columns
        theta_hat_b.rename(index={np.nan: 'NA'}, inplace=True)
        theta_hat.rename(index={np.nan: 'NA'}, inplace=True)
        # Need to change fill_value in pd.DataFrame.sub from default None
        # to 0 for variance to be correctly calculated.
        fill_value = 0
    else:
        fill_value = None

    if mse:
        brr_var = (
            (theta_hat_b.sub(theta_hat, axis=0, fill_value=fill_value) ** 2)
            .sum(axis='columns', level=1) / len(brrweight)
        ).squeeze()
        brr_var.name = 'Var'
    else:
        print('Only have MSE capability')
        return
    brr_se = np.sqrt(brr_var)
    brr_se.name = 'SE'

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


if __name__ == '__main__':
    df = pd.read_stata(datapath + 'nhanes2b.dta')

    # add one row to dataframe
    new_idx = len(df.index)
    df.loc[new_idx, 'diabetes'] = 0
    # df.loc[new_idx, 'finalwgt'] = 1

    # Testing numeric diabetes
    tabulate(data=df, row='diabetes', count=True, floatfmt=',.0f')
    tabulate(data=df, row='diabetes', count=False, floatfmt='.4f')
    tabulate(data=df, row='diabetes', weight='finalwgt', count=True,
             floatfmt=',.0f')
    tabulate(data=df, row='diabetes', weight='finalwgt', count=False,
             floatfmt='.4f')

    # testing categorical race
    tabulate(data=df, row='race', count=True, floatfmt=',.0f')
    tabulate(data=df, row='race', count=False, floatfmt='.4f')
    tabulate(data=df, row='race', weight='finalwgt', count=True,
             floatfmt=',.0f')
    tabulate(data=df, row='race', weight='finalwgt', count=False,
             floatfmt='.4f')

    # tabulate(data=df, row='race', col='diabetes', count=True, floatfmt=',.0f')
    # tabulate(data=df, row='race', col='diabetes', count=False, floatfmt=',.4f')
    # tabulate(data=df, row='race', col='diabetes', weight='finalwgt',
    #          count=True, floatfmt=',.0f')
    # tabulate(data=df, row='race', col='diabetes', weight='finalwgt',
    #          count=False, floatfmt=',.4f')

    # tabulate(data=df, col='race', row='diabetes', count=True, floatfmt=',.0f')
    # tabulate(data=df, col='race', row='diabetes', count=False, floatfmt=',.4f')
    # tabulate(data=df, col='race', row='diabetes', weight='finalwgt',
    #          count=True, floatfmt=',.0f')
    # tabulate(data=df, col='race', row='diabetes', weight='finalwgt',
    #          count=False, floatfmt=',.4f')

    # Testing missing values
    tabulate(data=df, row='diabetes')
    tabulate(data=df, row='diabetes', count=True)
    tabulate(data=df, row='diabetes', count=True, missing=True)

    tabulate(data=df, row='race', col='diabetes', count=True)
    tabulate(data=df, row='race', col='diabetes', count=True, missing=True)

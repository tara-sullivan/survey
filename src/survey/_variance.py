import pandas as pd
import numpy as np

import os
import inspect
try:
    currpath = os.path.dirname(os.path.abspath(__file__))
except NameError:
    currpath = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))
    rootpath = os.path.dirname(os.path.dirname(currpath))
    datapath = rootpath + '\\tests\\data\\'

from _agg import _count


def _se(df, row, col, weight, missing,
        aggfunc=np.sum,
        vce='brr', brrweight=None, mse=True,
        floatfmt='{:,.0f}', printtab=False):
    '''
    Produces a dataframe with a point estimate and standard errors,
    where standard errors are calculated using the BRR methods.
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
    theta_hat = _count(
        df=df, row=row, col=col, weight=weight,
        missing=missing, aggfunc=aggfunc)
    # theta_hat_b is the vector of point estimates from the ith
    # replication
    theta_hat_b = _count(
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

import pandas as pd
import numpy as np

# from make_bps_df import bps_derived
import tabulate as tab

from _agg import _count

print('Done')


def tabulate(
    data, row, col=None, weight=None, *,
    count=False, missing=False,
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
    n_hat_rc = _count(df=df, row=row, col=col,
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


# TODO: make this better
# If there's no column, and the types are series:
def _fmt_var(var, fmt):
    newvar = var.copy()

    if type(newvar) is pd.DataFrame:
        return newvar.applymap(fmt.format)
    if type(newvar) is pd.Series:
        return newvar.map(fmt.format)


if __name__ == '__main__':
    df = pd.read_stata(datapath + 'nhanes2b.dta')

    # add one row to dataframe
    new_idx = len(df.index)
    df.loc[new_idx, 'diabetes'] = 0
    # df.loc[new_idx, 'finalwgt'] = 1

    # Testing numeric diabetes
    tabulate(data=df, row='diabetes', count=True, floatfmt=',.0f')
    # tabulate(data=df, row='diabetes', count=False, floatfmt='.4f')
    # tabulate(data=df, row='diabetes', weight='finalwgt', count=True,
    #          floatfmt=',.0f')
    # tabulate(data=df, row='diabetes', weight='finalwgt', count=False,
    #          floatfmt='.4f')

    # # testing categorical race
    # tabulate(data=df, row='race', count=True, floatfmt=',.0f')
    # tabulate(data=df, row='race', count=False, floatfmt='.4f')
    # tabulate(data=df, row='race', weight='finalwgt', count=True,
    #          floatfmt=',.0f')
    # tabulate(data=df, row='race', weight='finalwgt', count=False,
    #          floatfmt='.4f')

    # # tabulate(data=df, row='race', col='diabetes', count=True, floatfmt=',.0f')
    # # tabulate(data=df, row='race', col='diabetes', count=False, floatfmt=',.4f')
    # # tabulate(data=df, row='race', col='diabetes', weight='finalwgt',
    # #          count=True, floatfmt=',.0f')
    # # tabulate(data=df, row='race', col='diabetes', weight='finalwgt',
    # #          count=False, floatfmt=',.4f')

    # # tabulate(data=df, col='race', row='diabetes', count=True, floatfmt=',.0f')
    # # tabulate(data=df, col='race', row='diabetes', count=False, floatfmt=',.4f')
    # # tabulate(data=df, col='race', row='diabetes', weight='finalwgt',
    # #          count=True, floatfmt=',.0f')
    # # tabulate(data=df, col='race', row='diabetes', weight='finalwgt',
    # #          count=False, floatfmt=',.4f')

    # # Testing missing values
    # tabulate(data=df, row='diabetes')
    # tabulate(data=df, row='diabetes', count=True)
    # tabulate(data=df, row='diabetes', count=True, missing=True)

    # tabulate(data=df, row='race', col='diabetes', count=True)
    # tabulate(data=df, row='race', col='diabetes', count=True, missing=True)

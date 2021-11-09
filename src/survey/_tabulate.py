import pandas as pd
import numpy as np
import pdb

# For printing
from tabulate import tabulate as prettytab
import textwrap

from _agg import _count
import _variance


def tabulate(
    df, row, col=None, weight=None,
    count=False, missing=False, margins=True,
    se=False, brrweight=None,
    show=False, **pkwargs
):
    '''
    Two-way tables for survey data. Replicates Stata svy: tabulate twoway.

    Required:

        * df:     sample dataframe
        * row:    variable to tabulate

    Optional:

        * col:     optional column variable for twoway tabualte
        * weight:  weight: pweight (default = 1)
        * count:   include count
        * margins: include margins
        * missing: treat missing values like other values
        * se:      whether to return standard errors (should only be used
                   for weighted results)
        * brrweight: if calculating standard errors using BRR weights, supply
                     weights here.
        * pkwargs: if show=True, pass along arguments for printing

    Optional pkwargs:

        * floatfmt: float format. Default = ',.2f'
        * col_labels: dictionary to map values of column labels
        * row_labels: dictionary to map values of row labels
    '''
    # if printing, collect printing kwargs:
    if show:
        # formating pkwarg
        if 'floatfmt' not in pkwargs:
            floatfmt = ',.2f'
        else:
            floatfmt = pkwargs['floatfmt']
        # column labels pkwarg
        if 'col_labels' not in pkwargs:
            col_labels = {}
        else:
            col_labels = pkwargs['col_labels']
            col_labels = {k: textwrap.fill(v, 40)
                          for k, v in col_labels.items()}
        if 'row_labels' not in pkwargs:
            row_labels = {}
        else:
            row_labels = pkwargs['row_labels']
            row_labels = {k: textwrap.fill(v, 40)
                          for k, v in row_labels.items()}

    df = df.copy()

    # if not including SE, count using _count function
    if not se:
        n_hat_rc = _count(df=df, row=row, col=col,
                          weight=weight, missing=missing,
                          margins=True)
        # if returning the count without standard errors, you're done
        if count:
            returntab = n_hat_rc
        # if returning proportion without SE, calculate that here
        else:
            if col is None:
                tot_idx = 'All'
            else:
                tot_idx = ('All', 'All')
            n_hat = n_hat_rc.loc[tot_idx]
            p_hat_rc = (n_hat_rc / n_hat)
            returntab = p_hat_rc
    # if including SE, count using _variance._se function
    else:
        if weight is None:
            print('Please input weight')
            return
        if col is not None:
            se_var = [row, col]
        else:
            se_var = row
        # Count the variance and SE here
        if count:
            n_hat_rc, n_hat_se = _variance._se(
                df=df, var=se_var, weight=weight, theta='count',
                brrweight=brrweight, missing=missing)
            returntab, returntab_se = n_hat_rc, n_hat_se
        # count using proportion here. Note that we don't need to manually
        # calculate the proportion here.
        else:
            n_hat_rc, n_hat_se = _variance._se(
                df=df, var=se_var, weight=weight, theta='proportion',
                brrweight=brrweight, missing=missing)
            returntab, returntab_se = n_hat_rc, n_hat_se

    # if not showing, you're done
    if not show:
        if not se:
            return returntab
        else:
            return returntab, returntab_se

    # If showing
    if show:
        # Edits to make if printing a series:
        if isinstance(returntab, pd.Series):
            # Create dataframe to print
            if not se:
                newtab = (returntab.map(f'{{:{floatfmt}}}'.format))
            else:
                # pdb.set_trace()
                newtab = (
                    returntab.map(f'{{:{floatfmt}}}'.format)
                    + '\n'
                    + returntab_se.map(f'({{:{floatfmt}}})'.format))
            newtab.index = (
                newtab.index.to_series().replace(row_labels))
            newtab = newtab.to_frame(name=row)
        # Edits to make if printing a dataframe:
        if isinstance(returntab, pd.DataFrame):
            if se is False:
                newtab = returntab.applymap(f'{{:{floatfmt}}}'.format)
            else:
                # pdb.set_trace()
                # apply map only works with dataframes; map with series
                if type(returntab) is pd.DataFrame:
                    newtab = (
                        returntab.applymap(f'{{:{floatfmt}}}'.format)
                        + '\n'
                        + returntab_se.applymap(f'({{:{floatfmt}}})'.format))
                # elif type(returntab) is pd.Series:
                #     newtab = (
                #         returntab.map(f'{{:{floatfmt}}}'.format)
                #         + '\n'
                #         + returntab_se.map(f'({{:{floatfmt}}})'.format))                    
            newtab.index = newtab.index.to_series().replace(row_labels)
            newtab.rename(columns=col_labels, inplace=True)
        _prettyprint(newtab, floatfmt=floatfmt)
    else:
        if se:
            return returntab, n_hat_se
        else:
            return returntab

    # if count:
    #     returntab = n_hat_rc
    #     # if printtab:
    #     #     print(tab.tabulate(
    #     #         n_hat_rc,
    #     #         headers="keys", tablefmt='psql',
    #     #         floatfmt=floatfmt, numalign='right'
    #     #     ))
    #     # try:
    #     #     return n_hat_rc.squeeze()
    #     # except:
    #     #     return n_hat_rc
    #     # return n_hat_rc.squeeze()
    # else:
    #     tot_idx = 'All'
    #     n_hat = n_hat_rc.loc[tot_idx]
    #     p_hat_rc = n_hat_rc / n_hat
    #     returntab = p_hat_rc

    # if se:
    #     returntab = _se(
    #         df=df, row=row, col=col, weight=weight, missing=missing,
    #         aggfunc=aggfunc,
    #         vce='brr', brrweight=brrweight, mse=mse,
    #         floatfmt='{:' + floatfmt + '}', printtab=printtab
    #     )

    # if show:
    #     if type(returntab) is pd.Series:
    #         returntab_toprint = returntab.to_frame(name='Total')
    # #     else:
    # #         returntab_toprint = returntab
    # #     print(tab.tabulate(
    # #         returntab_toprint,
    # #         headers="keys", tablefmt='psql',
    # #         floatfmt=floatfmt, numalign='right'
    # #     ))

    # return returntab


def _prettyprint(tab, headers='keys', floatfmt='.2f'):
    # pdb.set_trace()
    print(prettytab(
        tab,
        headers=headers, tablefmt='psql',
        numalign='right', floatfmt=floatfmt))


# TODO: make this better
# If there's no column, and the types are series:
def _fmt_var(var, fmt):
    newvar = var.copy()

    if type(newvar) is pd.DataFrame:
        return newvar.applymap(fmt.format)
    if type(newvar) is pd.Series:
        return newvar.map(fmt.format)

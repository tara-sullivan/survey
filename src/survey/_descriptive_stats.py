import numpy as np
import pandas as pd
import pdb

import textwrap
from tabulate import tabulate as prettytab

from _agg import _apply_group_func
import _variance


def mean(df, var, weight, over=None, missing=False,
         se=False, brrweight=None, show=False, **pkwargs
         ):
    '''
    Calculate the mean of a variable, using sample weights. Returns either (1)
    a point estimate or table of means; (2) means and standard errors; or (3)
    prints the table

    Required:

        * df:     sample dataframe
        * var:    variable to caluclate mean for
        * weight: weight variable

    Optional:

        * show:    print final table or return answers
        * over:    subpopulation to caluclate mean over
        * missing: treat missing like other values; only relevant if
                   calculating the mean of var1 over values of var2, and var2
                   contains missing values.
        * se:      whether to return standard errors.
        * brrweight: if calculating standard errors using BRR weights, supply
                     weights here.
        * pkwargs: if show=True, pass along arguments for printing
    '''
    # if printing, collect printing kwargs:
    if show:
        # formating pkwarg
        if 'floatfmt' not in pkwargs:
            floatfmt = ',.2f'
        else:
            floatfmt = pkwargs['floatfmt']
        # column labels pkwark
        if 'col_labels' not in pkwargs:
            col_labels = None
        else:
            col_labels = pkwargs['col_labels']
            col_labels = {k: textwrap.fill(v, 40) for k, v in col_labels.items()}
        if 'row_labels' not in pkwargs:
            row_labels = {}
        else:
            row_labels = pkwargs['row_labels']
            row_labels = {k: textwrap.fill(v, 40) for k, v in row_labels.items()}
        if 'header' not in pkwargs:
            headers = 'keys'
        else:
            header = pkwargs['header']

    if not se:
        y_bar = _w_mean(df=df, var=var, weight=weight,
                        over=over, missing=missing)
        # print the variable if show
        if show:
            # Create dataframe if answer is a single number
            if isinstance(y_bar, (float, int)):
                y_bar_tab = pd.DataFrame(
                    [y_bar],
                    index=['Mean'],
                    columns=col_labels)
                print(prettytab(y_bar_tab, floatfmt=floatfmt))
            # Create dataframe if answer is a series
            if isinstance(y_bar, pd.Series):
                # Create dataframe to print
                y_bar_tab = (y_bar.map(f'{{:{floatfmt}}}'.format))
                y_bar_tab.index = y_bar.index.to_series().replace(row_labels)
                y_bar_tab = y_bar_tab.to_frame(name=var)
                _prettyprint(y_bar_tab, floatfmt=floatfmt)
        # if not showing table, return the table
        else:
            return y_bar
    else:
        # pass
        y_bar, y_se = _variance._se(
            df=df, var=var, weight=weight, theta='mean',
            brrweight=brrweight, missing=missing, over=over)
        if show:
            # There may be an easier way to use variable f-strings
            # with pandas series, but I really haven't found it yet.
            y_bar_tab = (
                y_bar.map(f'{{:{floatfmt}}}'.format)
                + '\n' + y_se.map(f'({{:{floatfmt}}})'.format))
            y_bar_tab.index = y_bar.index.to_series().replace(row_labels)
            y_bar_tab = y_bar_tab.to_frame(name=var)
            _prettyprint(y_bar_tab, floatfmt=floatfmt)
        else:
            return y_bar, y_se


def _w_mean(df, var, weight, over=None, missing=False):
    if over is not None:
        # Write a function I can apply to the group (i.e. the dataframe
        # produced by groupby).
        def _group_w_mean(group):
            return _calc_w_mean(group[var], group[weight])
        y_bar = _apply_group_func(
            df=df, group_var=over, var=var, weight=weight,
            func=_group_w_mean, missing=missing)
    else:
        y_bar = _calc_w_mean(df[var], df[weight])
    return y_bar


def _calc_w_mean(var, weight):
    '''
    Weighted mean, calculated from two series
    '''
    # eliminate obs where df[var].isna(); otherwise .dot will produce nan
    # also fill the missing weight with nan for the same reaon.
    # Note that this could produce issues when calculating standard errors;
    # see [SVY] - svy estimation - Remarks and Examples
    var_notna = var.dropna()
    weight_notna = weight.loc[var_notna.index]
    y_bar = var_notna.dot(weight_notna) / (weight_notna.sum())
    return y_bar


def _fmt_var(var, fmt):
    # newvar = var.copy()
    if isinstance(var, (float, int)):
        return f'{var:{fmt}}'
    elif type(var) is pd.DataFrame:
        return var.applymap(fmt.format)
    elif type(var) is pd.Series:
        return var.map(fmt.format)


def _prettyprint(tab, headers='keys', floatfmt='.2f'):
    # pdb.set_trace()
    print(prettytab(
        tab,
        headers=headers, tablefmt='psql',
        numalign='right', floatfmt=floatfmt))

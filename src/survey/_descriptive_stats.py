import numpy as np
import pandas as pd
import pdb

from statsmodels.stats.weightstats import DescrStatsW

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
        # pdb.set_trace()
        y_bar, y_se = _variance._se(
            df=df, var=var, weight=weight, theta='mean',
            brrweight=brrweight, missing=missing, over=over)
        if show:
            # There may be an easier way to use variable f-strings
            # with pandas series, but I really haven't found it yet.
            y_bar_tab = (
                y_bar.map(f'{{:{floatfmt}}}'.format)
                + '\n' + y_se.map(f'({{:{floatfmt}}})'.format))
            # pdb.set_trace()
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
            # pdb.set_trace()
            # If there are multiple weights, we want to return a series
            # (which would happen if you manually calculated weighted
            # mean), not an ndarray (whichhappens if you use the
            # DescrStatsW class). Calling the _calc_w_mean function will
            # return either a number (if the weight is 1-d) or an array
            # (if the weight is 2-d). If weights are a group of numbers,
            # we want to return a series.
            w_mean_ans = _calc_w_mean(group[var], group[weight])
            if isinstance(weight, list):
                w_mean_ans = pd.Series(w_mean_ans, index=weight)
            return w_mean_ans
        y_bar = _apply_group_func(
            df=df, group_var=over, var=var, weight=weight,
            func=_group_w_mean, missing=missing)
    else:
        y_bar = _calc_w_mean(df[var], df[weight])
        if isinstance(weight, list):
            y_bar = pd.Series(y_bar, index=weight)
    return y_bar


# def _calc_w_mean(var, weight):
#     '''
#     Weighted mean, calculated from two series
#     '''
#     # eliminate obs where df[var].isna(); otherwise .dot will produce nan
#     # also fill the missing weight with nan for the same reaon.
#     # Note that this could produce issues when calculating standard errors;
#     # see [SVY] - svy estimation - Remarks and Examples
#     var_notna = var.dropna()
#     weight_notna = weight.loc[var_notna.index]
#     y_bar = var_notna.dot(weight_notna.fillna(0)) / (weight_notna.sum())
#     return y_bar


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
    svy_class = DescrStatsW(data=var_notna,
                            weights=weight_notna.fillna(0))
    return svy_class.mean


###############################
# Weighted standard deviation #
###############################

def std(df, var, weight, over=None, missing=False,
        se=False, brrweight=None
        ):
    '''
    Calculate the standard deviation of a variable, using sample weights.
    Returns either (1) a point estimate or table of standard deviation; or
    (2) standard deviation and standard errors for those estimates

    Required:

        * df:     sample dataframe
        * var:    variable to caluclate mean for
        * weight: weight variable

    Optional:

        * over:    subpopulation to caluclate mean over
        * missing: treat missing like other values; only relevant if
                   calculating the mean of var1 over values of var2, and var2
                   contains missing values.
        * se:      whether to return standard errors.
        * brrweight: if calculating standard errors using BRR weights, supply
                     weights here.
    '''
    # if printing, collect printing kwargs:

    if not se:
        y_bar = _w_std(df=df, var=var, weight=weight,
                       over=over, missing=missing)
        return y_bar
    else:
        y_bar, y_se = _variance._se(
            df=df, var=var, weight=weight, theta='std',
            brrweight=brrweight, missing=missing, over=over)
        return y_bar, y_se


def _calc_w_std(var, weight):
    '''
    Weighted standard deviation with default degrees of freedom.

    Note that this may produce different results from Stata, I believe because
    of the ways the degree of freedom adjustment is being calculated; looking
    at the source code, I think ddof are adjusted for sum_weights - ddof, which
    may not be correct. I recommedn revisiting this in the future.
    '''
    # eliminate obs where df[var].isna(); otherwise .dot will produce nan
    # also fill the missing weight with nan for the same reaon.
    # Note that this could produce issues when calculating standard errors;
    # see [SVY] - svy estimation - Remarks and Examples
    var_notna = var.dropna()
    weight_notna = weight.loc[var_notna.index]
    svy_class = DescrStatsW(data=var_notna,
                            weights=weight_notna.fillna(0),
                            ddof=1)
    return svy_class.std


def _w_std(df, var, weight, over=None, missing=False):
    if over is not None:
        # Write a function I can apply to the group (i.e. the dataframe
        # produced by groupby).
        def _group_w_std(group):
            # pdb.set_trace()
            # If there are multiple weights, we want to return a series
            # (which would happen if you manually calculated weighted
            # std), not an ndarray (whichhappens if you use the
            # DescrStatsW class). Calling the _calc_w_std function will
            # return either a number (if the weight is 1-d) or an array
            # (if the weight is 2-d). If weights are a group of numbers,
            # we want to return a series.
            if isinstance(weight, list):
                std_list = []
                for w in weight:
                    w_std = _calc_w_std(group[var], group[w])
                    std_list.append(w_std)

                w_std_ans = pd.Series(std_list, index=weight)
            else:
                w_std_ans = _calc_w_std(group[var], group[weight])
            return w_std_ans
        y_bar = _apply_group_func(
            df=df, group_var=over, var=var, weight=weight,
            func=_group_w_std, missing=missing)
    else:
        y_bar = _calc_w_std(df[var], df[weight])
        if isinstance(weight, list):
            y_bar = pd.Series(y_bar, index=weight)
    return y_bar


#####################
# Weighted quantile #
#####################

def quantile(df, var, weight, probs=0.5, over=None, missing=False,
             se=False, brrweight=None
             ):
    '''
    Calculate the quantile of a variable, using sample weights.
    Returns either (1) a point estimate or table of quantile; or
    (2) quantile and standard errors for those estimates

    Required:

        * df:     sample dataframe
        * var:    variable to caluclate mean for
        * weight: weight variable

    Optional:

        * probs:   probability point at which to calculate quantile
        * over:    subpopulation to caluclate mean over
        * missing: treat missing like other values; only relevant if
                   calculating the mean of var1 over values of var2, and var2
                   contains missing values.
        * se:      whether to return standard errors.
        * brrweight: if calculating standard errors using BRR weights, supply
                     weights here.
    '''
    # if printing, collect printing kwargs:

    if not se:
        y_bar = _w_quantile(
            df=df, var=var, weight=weight, probs=probs,
            over=over, missing=missing)
        return y_bar
    else:
        y_bar, y_se = _variance._se(
            df=df, var=var, weight=weight, theta='quantile', probs=probs,
            brrweight=brrweight, missing=missing, over=over)
        return y_bar, y_se


def _calc_w_quantile(var, weight, probs):
    '''
    Compute quantile for weighted sample.

    Note that for the DescrStatsW method quantile, probs can be a single
    probability (i.e. .5) or a list of probabilities ([.25, .5, .75]).

    This method either returns a numpy array or a pandas series or DF; I
    have return_pandas set to true.
    '''
    # eliminate obs where df[var].isna(); otherwise .dot will produce nan
    # also fill the missing weight with nan for the same reaon.
    # Note that this could produce issues when calculating standard errors;
    # see [SVY] - svy estimation - Remarks and Examples
    var_notna = var.dropna()
    weight_notna = weight.loc[var_notna.index]
    svy_class = DescrStatsW(data=var_notna,
                            weights=weight_notna.fillna(0),
                            ddof=1)
    return svy_class.quantile(probs, return_pandas=True)


def _w_quantile(df, var, weight, probs, over=None, missing=False):
    if over is not None:
        # Write a function I can apply to the group (i.e. the dataframe
        # produced by groupby).
        def _group_w_quantile(group):
            # Cannot pass a group of weights to DescrStatsW and execute
            # quantile methods for each weight; do in a list and append
            if isinstance(weight, list):
                quantile_list = []
                for w in weight:
                    w_quantile = _calc_w_quantile(group[var], group[w], probs)
                    quantile_list.append(w_quantile)

                # returns a dataframe with brrweight as columns, prob as index
                w_quantile_ans = pd.concat(quantile_list, axis=1, keys=weight)
            else:
                w_quantile_ans = _calc_w_quantile(
                    group[var], group[weight], probs)
            return w_quantile_ans
        y_bar = _apply_group_func(
            df=df, group_var=over, var=var, weight=weight,
            func=_group_w_quantile, missing=missing)
        # when calculating variance for this estimate, we want the probs to be
        # one level of the index; since over != None, we want a multiindex, w/
        # level 1 = over variable, and level2 = p (the quantile prob). Then the
        # estimate for each of the BRR weights will be in the columns. If using
        # one weight, muliple p, you end up with a dataframe with p in the
        # columns; hence the stack.
        if y_bar.columns.name == 'p':
            y_bar = y_bar.stack('p')
    else:
        if isinstance(weight, list):
            quantile_list = []
            for w in weight:
                w_quantile = _calc_w_quantile(df[var], df[w], probs)
                quantile_list.append(w_quantile)
            # returns a dataframe, columns = weights, index=probs
            w_quantile_ans = pd.concat(quantile_list, axis=1, keys=weight)
        else:
            w_quantile_ans = _calc_w_quantile(df[var], df[weight], probs)
        y_bar = w_quantile_ans
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

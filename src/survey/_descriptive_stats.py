import numpy as np
# import pandas as pd
import pdb

from _agg import _apply_group_func
import _variance


def mean(df, var, weight, over=None, missing=False,
         se=False, brrweight=None
         ):
    if not se:
        y_bar = _w_mean(df=df, var=var, weight=weight,
                        over=over, missing=missing)
    else:
        # pass
        y_bar, y_se = _variance._se(
            df=df, var=var, weight=weight, theta='mean',
            brrweight=brrweight, missing=missing, over=over)
        return y_bar, y_se
    return y_bar


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

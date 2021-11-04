import pandas as pd
import numpy as np
# import unittest

import decimal
from importlib import reload

import os
import inspect
import sys
try:
    currpath = os.path.dirname(os.path.abspath(__file__))
except NameError:
    currpath = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))
rootdir = os.path.dirname(currpath)
srcdir = os.path.join(rootdir, 'src', 'survey')
if rootdir not in sys.path:
    sys.path.append(rootdir)
if srcdir not in sys.path:
    sys.path.append(srcdir)
datapath = currpath + '\\data\\'

# mymod = sys.modules['_descriptive_stats']
# reload(mymod);
# mymod = sys.modules['_agg']
# reload(mymod)
from _agg import _apply_group_func
from _descriptive_stats import mean


def def_test_mean():
    df = pd.read_stata(datapath + 'nmihs.dta')
    myfunc = mean(df=df, var='birthwgt', weight='finwgt')
    answer = 3355.452
    round_num = abs(decimal.Decimal(str(answer)).as_tuple().exponent)
    assert round(myfunc, round_num) == answer, 'mean result not equal'


def test_mean_over():
    '''
    Test mean over another variable.

    Equivalent to: svy: mean tgresult, over(diabetes)
    '''
    df = pd.read_stata(datapath + 'nhanes2brr.dta')
    df['diabetes_cat'] = df['diabetes'].astype('category')
    # brrweight = list(df.columns[df.columns.str.contains('brr')])

    myfunc = mean(
        df=df, var='tgresult', weight='finalwgt',
        se=False, over='diabetes_cat', missing=True
    )
    myfunc = myfunc.loc[myfunc.index.notna()]
    answer = pd.Series(
        data=[136.6997, 191.9708], index=[0, 1])

    pd.testing.assert_series_equal(
        myfunc, answer,
        check_index_type=False, check_names=False
    )


def var_mean_nm_over():
    df = pd.read_stata(datapath + 'nhanes2brr.dta')
    df['diabetes_cat'] = df['diabetes'].astype('category')
    brrweight = list(df.columns[df.columns.str.contains('brr')])

    my_mean, my_se = mean(
        df=df, var='tgresult', weight='finalwgt', se=True,
        brrweight=brrweight, missing=False, over='diabetes_cat'
    )
    ans_mean = pd.Series([136.6997, 191.9708], index=[0, 1])
    ans_se = pd.Series([2.095444, 6.337179], index=[0, 1])

    pd.testing.assert_series_equal(
        round(my_mean, 4), ans_mean,
        check_index_type=False, check_names=False, check_categorical=False
    )
    pd.testing.assert_series_equal(
        round(my_se, 3), round(ans_se, 3),
        check_categorical=False, check_index_type=False, check_names=False
    )



if __name__ == '__main__':
    def_test_mean()
    test_mean_over()
    var_mean_nm_over()

import pandas as pd
import numpy as np
# import unittest

from importlib import reload
import decimal

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
# reload(mymod)
# mymod = sys.modules['_agg']
# reload(mymod)
# mymod = sys.modules['_variance']
# reload(mymod)
from _variance import _se, _var
from _descriptive_stats import _w_mean

df = pd.read_stata(datapath + 'nhanes2brr.dta')
df['diabetes_cat'] = df['diabetes'].astype('category')
brrweight = list(df.columns[df.columns.str.contains('brr')])


#########
# Count #
#########

def var_count_ser_nm():
    '''
    Check the variance for the count of:
        * A series (one variable; col=None)
        * Dropping missing values

    Equivalent to: svy: tab diabetes, se count format (%12.0fc)
    '''
    answer = pd.Series([5478997928256, 54425552665], index=[0, 1])

    _, brr_var = _var(
        df, var='diabetes_cat', col=None, weight='finalwgt', missing=False,
        brrweight=brrweight, mse=True)
    myfunc = brr_var.drop(labels='All', axis=0)
    pd.testing.assert_series_equal(
        myfunc, answer,
        check_index_type=False, check_dtype=False, check_names=False)


def var_count_ser_m():
    '''
    Check the variance for the count of:
        * A series (one variable; col=None)
        * Treating missing values like other values

    Equivalent to: svy: tab diabetes, se count format (%12.0fc)
    '''
    answer = pd.Series([2340726, 233293, 18693], index=[0, 1, 'NA'])
    _, brr_se = _se(
        df, var='diabetes_cat', col=None, weight='finalwgt', missing=True,
        brrweight=brrweight, mse=True)
    # brr_se.index = brr_se.index.fillna('Missing')
    myfunc = round(brr_se, 0).drop(labels='All', axis=0)
    pd.testing.assert_series_equal(
        myfunc, answer,
        check_index_type=False, check_dtype=False, check_names=False
    )


def var_count_df_nm():
    '''
    Check the variance for the count of:
        * A dataframe (two variables; col is not None)
        * Dropping missing values

    Equivalent to: svy: tab diabetes race, se count format (%12.0f)
    '''
    answer = pd.DataFrame(
        [
            [2778220, 1388592, 1208984, 2340726],
            [242168, 98415, 45493, 233293],
            [2912056, 1458814, 1252160, np.nan]],
        index=[0, 1, 'All'], columns=['White', 'Black', 'Other', 'All'])
    _, brr_se = _se(
        df, var=['diabetes_cat', 'race'], weight='finalwgt', missing=False,
        brrweight=brrweight, mse=True)
    brr_se.loc['All', 'All'] = np.nan
    myfunc = round(brr_se, 0)
    pd.testing.assert_frame_equal(
        myfunc, answer,
        check_index_type=False, check_names=False, check_dtype=False
    )


def var_count_df_m():
    '''
    Check the variance for the count of:
        * A dataframe (two variables; col is not None)
        * Dropping missing values

    Equivalent to: svy: tab diabetes race, se count missing
    '''
    answer = pd.DataFrame(
        [
            [2778220, 1388592, 1208984, 2340726],
            [242168, 98415, 45493, 233293],
            [18693, 0, 0, 18693],
            [2912042, 1458814, 1252160, np.nan]],
        index=[0, 1, 'NA', 'All'],
        columns=['White', 'Black', 'Other', 'All']
    )

    _, brr_se = _se(
        df, var=['diabetes_cat', 'race'], weight='finalwgt', missing=True,
        brrweight=brrweight, mse=True)
    brr_se.loc['All', 'All'] = np.nan
    # brr_se.index = brr_se.index.fillna('Missing')
    myfunc = round(brr_se, 0).drop(columns='NA')
    pd.testing.assert_frame_equal(
        myfunc, answer,
        check_index_type=False, check_dtype=False, check_names=False
    )


##############
# Proportion #
##############

def var_prop_ser_nm():
    '''
    Check the variance for the proportion of:
        * A series (one variable; col=None)
        * Dropping missing values

    Equivalent to: svy: tab diabetes, se
    '''
    answer = pd.Series([0.0018, 0.0018], index=[0, 1])
    _, brr_se = _se(
        df, var='diabetes_cat', col=None, weight='finalwgt', missing=False,
        brrweight=brrweight, mse=True, theta='proportion')
    myfunc = brr_se.drop(labels='All', axis=0)
    pd.testing.assert_series_equal(
        round(myfunc, 4), answer,
        check_index_type=False, check_names=False
    )


def var_prop_ser_m():
    '''
    Check the variance for the proportion of:
        * A series (one variable; col=None)
        * Treating missing values like other values

    Equivalent to: svy: tab diabetes, se missing
    '''
    answer = pd.Series([0.001845, 0.001813, 0.000160], index=[0, 1, 'NA'])
    _, brr_se = _se(
        df, var='diabetes_cat', col=None, weight='finalwgt', missing=True,
        brrweight=brrweight, mse=True, theta='proportion')
    # brr_se.index = brr_se.index.fillna('Missing')
    myfunc = round(brr_se, 6).drop(labels='All', axis=0)
    pd.testing.assert_series_equal(
        myfunc, answer,
        check_index_type=False, check_names=False
    )


def var_prop_df_nm():
    '''
    Check the variance for the proportion of:
        * A dataframe (two variables; col is not None)
        * Dropping missing values

    Equivalent to: svy: tab diabetes race, se format(%12.6fc)
    '''
    answer = pd.DataFrame(
        [[0.015863, 0.012175, 0.010191, 0.001814],
         [0.001932, 0.000847, 0.000387, 0.001814],
         [0.016727, 0.012789, 0.010560, np.nan]],
        index=[0, 1, 'All'],
        columns=['White', 'Black', 'Other', 'All']
    )
    _, brr_se = _se(
        df, var=['diabetes_cat', 'race'], weight='finalwgt', missing=False,
        brrweight=brrweight, mse=True, count=False, theta='proportion')
    brr_se.loc['All', 'All'] = np.nan
    myfunc = round(brr_se, 6)
    pd.testing.assert_frame_equal(
        myfunc, answer,
        check_index_type=False, check_names=False, check_dtype=False
    )


def var_proportion_df_m():
    '''
    Check the variance for the count of:
        * A dataframe (two variables; col is not None)
        * Dropping missing values

    Equivalent to: svy: tab diabetes race, se missing format(%12.6fc)
    '''
    answer = pd.DataFrame(
        [[0.015859, 0.012172, 0.010189, 0.001845],
         [0.001931, 0.000846, 0.000387, 0.001813],
         [0.000160, 0.000000, 0.000000, 0.000160],
         [0.016724, 0.012786, 0.010558, np.nan]],
        index=[0, 1, 'NA', 'All'],
        columns=['White', 'Black', 'Other', 'All']
    )
    _, brr_se = _se(
        df, var=['diabetes_cat', 'race'], weight='finalwgt', missing=True,
        brrweight=brrweight, mse=True, count=False, theta='proportion')
    brr_se.loc['All', 'All'] = np.nan
    # brr_se.index = brr_se.index.fillna('Missing')
    brr_se.drop('NA', axis=1, inplace=True)
    myfunc = round(brr_se, 6)
    pd.testing.assert_frame_equal(
        myfunc, answer,
        check_index_type=False, check_names=False, check_dtype=False
    )


########
# Mean #
########

def var_mean_nm():
    '''
    Check the variance for the count of:
        * A series (one variable)
    Equivalent to: svy: mean weight
    '''
    my_mean, my_se = _se(
        df=df, var='weight', weight='finalwgt', brrweight=brrweight,
        theta='mean')

    ans_mean, ans_se = 71.90064, .1656454

    round_num = abs(decimal.Decimal(str(ans_mean)).as_tuple().exponent)
    assert round(my_mean, round_num) == ans_mean, '_mean not equal'
    assert round(my_se, 5) == round(ans_se, 5), '_se not equal'


def var_mean_m():
    '''
    Check the variance for the count of:
        * A series (one variable)
    Equivalent to: svy: mean weight
    '''
    my_mean, my_se = _se(
        df=df, var='tgresult', weight='finalwgt', brrweight=brrweight,
        theta='mean', missing=True)
    ans_mean, ans_se = 138.576, 2.072962
    assert round(my_mean, 3) == ans_mean, '_mean not equal'
    assert round(my_se, 5) == round(ans_se, 5), '_se not equal'


def var_mean_nm_over():
    my_mean, my_se = _se(
        df=df, var='tgresult', weight='finalwgt', theta='mean',
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
    var_count_ser_nm()
    var_count_ser_m()
    var_count_df_nm()
    var_count_df_m()

    var_prop_ser_nm()
    var_prop_ser_m()
    var_prop_df_nm()
    var_proportion_df_m()

    var_mean_nm()
    var_mean_m()
    var_mean_nm_over()

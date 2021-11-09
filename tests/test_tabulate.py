import pandas as pd
import numpy as np

# import survey as svy

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

# import survey as svy
from importlib import reload

from _tabulate import tabulate
# reload(sys.modules['_tabulate'])
# from _tabulate import tabulate
# reload(sys.modules['_variance']);
# from _descriptive_stats import mean

df = pd.read_stata(datapath + 'nhanes2brr.dta')
df['diabetes_cat'] = df['diabetes'].astype('category')
brrweight = list(df.columns[df.columns.str.contains('brr')])


def ser_count_nw_nm():
    '''
    Check a one-way tabulate (no column) that produces a series
        * Count
        * No weight
        * Dropping missing values

    Equivalent to: tab diabetes, missing
    '''
    tabulate(
        df, row='diabetes_cat', count=True,
        col=None, weight=None, missing=False,
        show=True, floatfmt=',.0f',
        row_labels={0: 'No Diab.', 1: 'Diab.', 'All': 'Tot'})
    myfunc = tabulate(
        df, row='diabetes_cat', count=True,
        col=None, weight=None, missing=False)
    answer = pd.Series([9850, 499, 10349], index=[0, 1, 'All'])
    pd.testing.assert_series_equal(
        myfunc, answer,
        check_dtype=False, check_index_type=False
    )


def ser_perc_nw_nm():
    '''
    Check a one-way tabulate (no column) that produces a series
        * Perc, not count
        * No weight
        * Dropping missing values

    Equivalent to: tab diabetes, missing
    '''
    tabulate(
        df, row='diabetes_cat', count=False,
        col=None, weight=None, missing=False,
        show=True, floatfmt=',.2f',
        row_labels={0: 'No Diab.', 1: 'Diab.', 'All': 'Tot'})
    myfunc = tabulate(
        df, row='diabetes_cat', count=False,
        col=None, weight=None, missing=False)
    myfunc = myfunc * 100
    answer = pd.Series([95.18, 4.82, 100.00], index=[0, 1, 'All'])
    pd.testing.assert_series_equal(
        round(myfunc, 2), answer,
        check_dtype=False, check_index_type=False
    )


def ser_count_w_nm_se():
    '''
    Check a one-way tabulate (no column) that produces a series
        * Count
        * Weighted
        * Dropping missing values

    Equivalent to: svy: tab diabetes, count format(%12.0f) se
    '''
    # calculate with se = True. Compare to answer where se=False
    # Compare to answers copied from Stata

    # Save correct answers from Stata
    answer = pd.Series(
        [113119830, 4011281, 117131111], index=[0, 1, 'All'])
    answer_se = pd.Series(
        [2340726, 233293], index=[0, 1]
    )

    # calculate with se = False. Check that it prints, and save the answer
    tabulate(
        df, row='diabetes_cat', count=True,
        col=None, weight='finalwgt', missing=False, se=False,
        show=True, floatfmt=',.0f',
        row_labels={0: 'No Diab.', 1: 'Diab.', 'All': 'Tot'})
    myfunc_no_se = tabulate(
        df, row='diabetes_cat', count=True,
        col=None, weight='finalwgt', missing=False, se=False)
    # Calculate with se = True. Check that it prints
    tabulate(
        df, row='diabetes_cat', count=True,
        col=None, weight='finalwgt', missing=False,
        se=True, brrweight=brrweight,
        show=True, floatfmt=',.0f',
        row_labels={0: 'No Diab.', 1: 'Diab.', 'All': 'Tot'})
    myfunc, myse = tabulate(
        df, row='diabetes_cat', count=True,
        col=None, weight='finalwgt', missing=False,
        se=True, brrweight=brrweight)

    # Check that you get the same answer with or without SE
    pd.testing.assert_series_equal(myfunc, myfunc_no_se)

    myfunc = round(myfunc, 0)
    myse.drop('All', inplace=True)
    myse = round(myse, 0)

    pd.testing.assert_series_equal(
        myfunc, answer,
        check_index_type=False
    )

    pd.testing.assert_series_equal(
        myfunc, answer,
        check_index_type=False
    )


def ser_prop_w_nm_se():
    '''
    Check a one-way tabulate (no column) that produces a series
        * Proportion
        * Weighted
        * Dropping missing values

    Equivalent to: svy: tab diabetes, format(%12.6f) se
    '''
    # calculate with se = True. Compare to answer where se=False
    # Compare to answers copied from Stata
    # calculate with se = False. Check that it prints
    tabulate(
        df, row='diabetes_cat', count=False,
        col=None, weight='finalwgt', missing=False, se=False,
        show=True, floatfmt=',.6f',
        row_labels={0: 'No Diab.', 1: 'Diab.', 'All': 'Tot'})
    # Save the answer for se = False
    myfunc_se_false = tabulate(
        df, row='diabetes_cat', count=False,
        col=None, weight='finalwgt', missing=False, se=False)
    # calculate the answer with se = True. First check that it prints
    tabulate(
        df, row='diabetes_cat', count=False,
        col=None, weight='finalwgt', missing=False,
        se=True, brrweight=brrweight,
        show=True, floatfmt=',.6f',
        row_labels={0: 'No Diab.', 1: 'Diab.', 'All': 'Tot'},
    )
    # Save the answer with se=True. compare the to answer where se=False
    myfunc, myse = tabulate(
        df, row='diabetes_cat', count=False,
        col=None, weight='finalwgt', missing=False,
        se=True, brrweight=brrweight,
    )
    # compare answer where se=True to se=False
    pd.testing.assert_series_equal(myfunc, myfunc_se_false)

    # compare answer from my func to saved stata results
    answer = pd.Series(
        [0.965754, 0.034246, 1.000000], index=[0, 1, 'All'])
    answer_se = pd.Series(
        [0.001814, 0.001814], index=[0, 1])

    myfunc = round(myfunc, 6)
    myse.drop('All', inplace=True)
    myse = round(myse, 6)

    pd.testing.assert_series_equal(
        myfunc, answer,
        check_index_type=False
    )

    pd.testing.assert_series_equal(
        myse, answer_se,
        check_index_type=False, check_dtype=False, check_names=False
    )
##############
# Dataframes #
##############

def df_count_nw_nm():
    '''
    Check a two-way tabulate (col is not None)
        * Count
        * No weight
        * Dropping missing values

    Equivalent to: tab diabetes race
    '''
    tabulate(
        df, row='diabetes_cat', col='race',
        count=True, weight=None, missing=False,
        show=True, floatfmt=',.0f',
        row_labels={0: 'No Diab.', 1: 'Diab.', 'All': 'Tot'},
        col_labels={'White': 'W', 'Black': 'B', 'Other': 'O'})
    myfunc = tabulate(
        df, row='diabetes_cat', col='race',
        count=True, weight=None, missing=False)
    answer = pd.DataFrame(
        [[8659, 1000, 191, 9850], [404, 86, 9, 499], [9063, 1086, 200, 10349]],
        index=[0, 1, 'All'],
        columns=['White', 'Black', 'Other', 'All']
    )
    pd.testing.assert_frame_equal(
        myfunc, answer,
        check_index_type=False, check_categorical=False,
        check_column_type=False, check_names=False
    )


def df_perc_nw_nm():
    '''
    Check a two-way tabulate (col is not None)
        * Count
        * No weight
        * Dropping missing values

    Equivalent to: tab diabetes race
    '''
    pass


def df_count_w_nm_se():
    '''
    Check a two-way tabulate (col is not None) with SE
        * Count
        * Weighted
        * Dropping missing values

    Equivalent to: svy: tabulate diabetes race, count format(%12.0f) se
    '''
    tabulate(
        df, row='diabetes_cat', col='race',
        count=True, weight='finalwgt', missing=False,
        se=True, brrweight=brrweight, show=True, floatfmt=',.0f')

    myfunc, myse = tabulate(
        df, row='diabetes_cat', col='race',
        count=True, weight='finalwgt', missing=False,
        se=True, brrweight=brrweight)
    myse = round(myse, 0)
    myse.loc['All', 'All'] = np.nan

    answer = pd.DataFrame(
        [[99682793, 10528681, 2908356, 113119830],
         [3290354, 660555, 60372, 4011281],
         [102973147, 11189236, 2968728, 117131111]],
        index=[0, 1, 'All'],
        columns=['White', 'Black', 'Other', 'All']
    )
    pd.testing.assert_frame_equal(
        myfunc, answer,
        check_index_type=False, check_categorical=False,
        check_column_type=False, check_names=False)

    answer_se = pd.DataFrame(
        [[2778220, 1388592, 1208984, 2340726],
         [242168, 98415, 45493, 233293],
         [2912056, 1458814, 1252160, np.nan]],
        index=[0, 1, 'All'],
        columns=['White', 'Black', 'Other', 'All']
    )
    pd.testing.assert_frame_equal(
        myse, answer_se,
        check_index_type=False, check_names=False, check_dtype=False
    )


def df_prop_w_nm_se():
    '''
    Check a two-way tabulate (col is not None) of proportions with SE
        * Weighted
        * Dropping missing values

    Equivalent to: svy: tabulate diabetes race, format(%12.6f) se
    '''
    answer = pd.DataFrame(
        [[0.851036, 0.089888, 0.024830, 0.965754],
         [0.028091, 0.005639, 0.000515, 0.034246],
         [0.879127, 0.095527, 0.025345, 1.000000]],
        index=[0, 1, 'All'],
        columns=['White', 'Black', 'Other', 'All']
    )
    answer_se = pd.DataFrame(
        [[0.015863, 0.012175, 0.010191, 0.001814],
         [0.001932, 0.000847, 0.000387, 0.001814],
         [0.016727, 0.012789, 0.010560, np.nan]],
        index=[0, 1, 'All'],
        columns=['White', 'Black', 'Other', 'All']
    )

    tabulate(
        df, row='diabetes_cat', col='race',
        count=False, weight='finalwgt', missing=False,
        se=True, brrweight=brrweight, show=True, floatfmt=',.6f')

    myfunc, myse = tabulate(
        df, row='diabetes_cat', col='race',
        count=False, weight='finalwgt', missing=False,
        se=True, brrweight=brrweight,)
    myfunc = round(myfunc, 6)
    myse = round(myse, 6)
    myse.loc['All', 'All'] = np.nan

    pd.testing.assert_frame_equal(
        myfunc, answer,
        check_index_type=False, check_column_type=False, check_names=False,
        check_categorical=False
    )
    pd.testing.assert_frame_equal(
        myse, answer_se,
        check_index_type=False, check_column_type=False,
        check_names=False, check_categorical=False
    )


if __name__ == '__main__':
    ser_count_nw_nm()
    ser_perc_nw_nm()

    ser_count_w_nm_se()
    ser_prop_w_nm_se()

    df_count_nw_nm()
    # df_perc_nw_nm()

    df_count_w_nm_se()
    df_prop_w_nm_se()

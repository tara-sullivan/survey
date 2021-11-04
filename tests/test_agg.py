import pandas as pd
# import unittest

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

from _agg import _count as agg
mymod = sys.modules['_agg']
reload(mymod)
# reload(_agg.agg)

df = pd.read_stata(datapath + 'nhanes2b.dta')
df['diabetes_cat'] = df['diabetes'].astype('category')


def stata_results_to_df(rows, row_names, col_names):
    '''
    Copy results from stat output into dataframe.
    '''
    dfdata = []
    for row in rows:
        dfdata.append([int(i) for i in row.split()])
    col_names = [
        i if (i == 'Missing' or i == 'All')
        else float(i) for i in col_names.split()]
    row_names = [
        i if (i == 'Missing' or i == 'All')
        else float(i) for i in row_names.split()]
    answer = pd.DataFrame(
        data=dfdata,
        columns=col_names,
        index=row_names
    )
    # Add row total
    if 'All' not in answer.index:
        answer.loc['All'] = answer.sum(numeric_only=True, axis=0)
    # Add column total
    if 'All' not in answer.columns:
        answer.loc[:, 'All'] = answer.sum(numeric_only=True, axis=1)
    return answer


#############
# pd.Series #
#############

def ser_nw_nm_sum():
    '''
    Test aggregating a series (col=None)
        * unweighted (weight=None)
        * nm: ignore missing (missing=False)
        * aggfunc = np.sum()

    Equivalent to: tabulate race
    '''
    myfunc = agg(
        df=df, row='race', col=None, weight=None, missing=False)
    answer = pd.Series(
        [9065, 1086, 200, 10351],
        index=['White', 'Black', 'Other', 'All'])
    pd.testing.assert_series_equal(
        myfunc, answer,
        check_names=False, check_dtype=False)


def ser_w_nm_sum():
    '''
    Test aggregating a series (col=None)
        * weighted (weight=var)
        * nm: ignore missing (missing=False)
        * aggfunc = np.sum()

    Equivalent to: svy: tabulate race, count format(%12.0f)
    '''
    myfunc = agg(
        df=df, row='race', col=None, weight='finalwgt', missing=False)
    answer = pd.Series(
        [102999549, 11189236, 2968728, 117157513],
        index=['White', 'Black', 'Other', 'All'])
    pd.testing.assert_series_equal(
        myfunc, answer,
        check_names=False, check_dtype=False)


def ser_nw_m_sum():
    '''
    Test aggregating a series (col=None)
        * nw: unweighted (weight=None)
        * m: treat missing like other values (missing=True)
        * aggfunc = np.sum()

    Equivalent to: tabulate diabetes, missing
    '''
    myfunc = agg(
        df=df, row='diabetes_cat', col=None, weight=None, missing=True)
    myfunc.index = myfunc.index.fillna('Missing')
    answer = pd.Series(
        [9850, 499, 2, 10351],
        index=[0, 1, 'Missing', 'All'])
    pd.testing.assert_series_equal(
        myfunc, answer,
        check_names=False, check_dtype=False, check_index_type=False)


def ser_w_m_sum():
    '''
    Test aggregating a series (col=None)
        * w: weighted (weight=None)
        * m: treat missing like other values (missing=True)
        * aggfunc = np.sum()

    Equivalent to: svy: tabulate diabetes, count format(%12.0f) missing
    '''
    myfunc = agg(
        df=df, row='diabetes_cat', col=None,
        weight='finalwgt', missing=True)
    myfunc.index = myfunc.index.fillna('Missing')
    answer = pd.Series(
        [113119830, 4011281, 26402, 117157513],
        index=[0, 1, 'Missing', 'All'])
    pd.testing.assert_series_equal(
        myfunc, answer,
        check_names=False, check_dtype=False, check_index_type=False)


################
# pd.DataFrame #
################

def df_nw_nm_sum():
    '''
    Test aggregating a dataframe (col is not None)
        * unweighted (weight=None)
        * nm: ignore missing (missing=False)
        * aggfunc = np.sum()

    Equivalent to: tabulate diabetes lead
    '''
    myfunc = agg(
        df=df, row='diabetes_cat', col='lead', weight=None, missing=False)
    r1_str = (
        ' 1   10   23   67  114  173  240  325  334  382  407  342'
        ' 364  304  268  221  170  177  149  130   98   74   63   46'
        ' 39   39   36   16   16   14   11   11   11    4    4    5'
        '  4    2    1    3    1    1    1    2    1    1    2    2'
        '  1    1    1    1    1'
    )
    r2_str = (
        ' 0    0    2    2    7   10   17   21   16   26   17   16'
        ' 21   12   12    8    4    5    7    3    8    2    1    3'
        '  3    1    3    0    1    2    0    0    0    1    1    0'
        '  0    0    0    0    0    0    0    0    0    0    0    0'
        '  0    0    0    0    0'
    )

    col_str = (
        '  2    3    4    5    6    7    8    9   10   11  '
        '  12   13   14   15    16   17   18'
        ' 19   20   21   22   23   24   25   26   27   28  '
        '  29   30   31   32    33   34   35'
        ' 36   37   38   39   40   41   42   43   45   47  '
        '  49   50   51   52    54   61   64'
        ' 66   80 '
    )
    row_str = ('0 1')

    answer = stata_results_to_df(
        rows=[r1_str, r2_str], row_names=row_str, col_names=col_str)
    pd.testing.assert_frame_equal(
        myfunc.fillna(0), answer, check_names=False,
        check_dtype=False, check_index_type=False, check_column_type=False)


def df_w_nm_sum():
    '''
    Test aggregating a dataframe (col is not None)
        * weighted (weight=var)
        * nm: ignore missing (missing=False)
        * aggfunc = np.sum()

    Equivalent to: svy: tabulate diabetes_cat lead
    '''
    myfunc = agg(
        df=df, row='diabetes_cat', col='lead', weight='finalwgt', missing=False
    )
    r1_str = (
        '  10331   109103   292763   895395  1390472  1963485  2745975  '
        ' 3633234  3872460  4435333  4630550  3861963  4219309  3547662  '
        ' 3113204  2603025  1816100  2033191  1753710  1584311  1087469 '
        ' 886278   702902   539743   490290   485447   418081   195468  '
        ' 205839   186382    98117   166842   104700    20178    56334   '
        ' 50946    70774    28264     2580    58170     3235    10912 '
        ' 8668    38181    16542     3005    13152    30382    10186    '
        ' 7729     7721    10514    17333 54543940 '
    )
    r2_str = (
        '      0        0    45980     9865    31736    60001   143998   '
        ' 148919   143681   247343   147826   124703   152100   121412    '
        ' 95998    42159    15314    19519    49754    22268    69435 '
        '   15195     3566    29256    14855     2894    26611        0    '
        ' 7746    24514        0        0        0     9805     8619       '
        ' 0        0        0        0        0        0        0 '
        '    0        0        0        0        0        0        0       '
        ' 0        0        0        0  1835072 '
    )
    r3_str = (
        '  10331   109103   338743   905260  1422208  2023486  2889973  '
        ' 3782153  4016141  4682676  4778376  3986666  4371409  3669074  '
        ' 3209202  2645184  1831414  2052710  1803464  1606579  1156904 '
        ' 901473   706468   568999   505145   488341   444692   195468  '
        ' 213585   210896    98117   166842   104700    29983    64953   '
        ' 50946    70774    28264     2580    58170     3235    10912 '
        ' 8668    38181    16542     3005    13152    30382    10186    '
        ' 7729     7721    10514    17333 56379012 '
    )
    row_str = ('0  1 All')
    col_str = (
        '    2        3        4        5        6        7        8      '
        '  9       10       11       12       13       14       15       '
        ' 16       17       18       19       20       21       22 '
        '     23       24       25       26       27       28       29    '
        '   30       31       32       33       34       35       36      '
        ' 37       38       39       40       41       42       43 '
        '   45       47       49       50       51       52       54     '
        '  61       64       66       80    All '
    )
    answer = stata_results_to_df(
        rows=[r1_str, r2_str, r3_str], row_names=row_str, col_names=col_str)
    # return myfunc, answer
    pd.testing.assert_frame_equal(
        myfunc.fillna(0), answer, check_names=False,
        check_dtype=False, check_index_type=False, check_column_type=False)


def df_nw_m_sum():
    '''
    Test aggregating a dataframe (col is not None)
        * unweighted (weight=None)
        * m: treat missing like other values (missing=True)
        * aggfunc = np.sum()

    Equivalent to: tabulate diabetes lead, missing
    '''
    r1_str = (
        '  1    10    23    67   114   173   240   325   334   382 '
        ' 407   342   364   304   268   221   170   177   149   130 '
        '  98    74    63    46    39    39    36    16    16    14 '
        '  11    11    11     4     4     5     4     2     1     3 '
        '   1     1     1     2     1     1     2     2     1     1 '
        '   1     1     1  5136 '
    )
    r2_str = (
        ' 0     0     2     2     7    10    17    21    16    26 '
        '  17    16    21    12    12     8     4     5     7     3 '
        '   8     2     1     3     3     1     3     0     1     2 '
        '   0     0     0     1     1     0     0     0     0     0 '
        '   0     0     0     0     0     0     0     0     0     0 '
        '  0     0     0   267 '
    )
    r3_str = (
        '  0     0     0     0     0     1     0     1     0     0 '
        '   0     0     0     0     0     0     0     0     0     0 '
        '    0     0     0     0     0     0     0     0     0     0 '
        '   0     0     0     0     0     0     0     0     0     0 '
        '    0     0     0     0     0     0     0     0     0     0 '
        '   0     0     0     0 '
    )
    row_str = ('0 1 Missing')
    col_str = (
        '  2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17 '
        '  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33'
        '  34  35  36  37  38  39  40  41  42  43  45  47  49  50  51  52'
        '  54  61  64  66  80   Missing '
    )
    answer = stata_results_to_df(
        rows=[r1_str, r2_str, r3_str], row_names=row_str, col_names=col_str)
    myfunc = agg(
        df=df, row='diabetes_cat', col='lead', weight=None, missing=True
    )
    myfunc.index = myfunc.index.fillna('Missing')
    myfunc.columns = myfunc.columns.fillna('Missing')
    pd.testing.assert_frame_equal(
        myfunc.fillna(0), answer, check_names=False, check_like=True,
        check_dtype=False, check_index_type=False, check_column_type=False)


def df_w_m_sum():
    '''
    Test aggregating a dataframe (col is not None)
        * weighted (weight=var)
        * m: treat missing like other values (missing=True)
        * aggfunc = np.sum()

    Equivalent to: svy: tabulate diabetes lead, count
    '''
    r1_str = (
        ' 10331    109103    292763    895395   1390472   1963485   2745975  '
        ' 3633234   3872460   4435333   4630550   3861963   4219309   3547662'
        ' 3113204   2603025   1816100   2033191   1753710 '
        ' 1584311   1087469    886278    702902    539743    490290   '
        ' 485447    418081    195468    205839    186382     98117    '
        ' 166842    104700     20178     56334     50946     70774    '
        ' 28264 '
        '  2580     58170      3235     10912      8668     38181    '
        ' 16542      3005     13152     30382     10186      7729    '
        '  7721     10514     17333  58575890 113119830 '
    )
    r2_str = (
        '     0         0     45980      9865     31736     60001    '
        ' 143998    148919    143681    247343    147826    124703   '
        ' 152100    121412     95998     42159     15314     19519   '
        '  49754 '
        '   22268     69435     15195      3566     29256     14855    '
        '  2894     26611         0      7746     24514         0       '
        '  0         0      9805      8619         0         0         0 '
        '    0         0         0         0         0         0        '
        ' 0         0         0         0         0         0       '
        '  0         0         0   2176209   4011281 '
    )
    r3_str = (
        '     0         0         0         0         0     12535       '
        '  0     13867         0         0         0         0         0 '
        '        0         0         0         0         0         0 '
        '        0         0         0         0         0         0     '
        '    0         0         0         0         0         0       '
        '  0         0         0         0         0         0         0 '
        '    0         0         0         0         0         0       '
        '  0         0         0         0         0         0       '
        '  0         0         0         0     26402 '
    )
    r4_str = (
        '   10331    109103    338743    905260   1422208   2036021  '
        ' 2889973   3796020   4016141   4682676   4778376   3986666  '
        ' 4371409   3669074   3209202   2645184   1831414   2052710   '
        ' 1803464'
        ' 1606579   1156904    901473    706468    568999    505145   '
        ' 488341    444692    195468    213585    210896     98117   '
        ' 166842    104700     29983     64953     50946     70774    '
        ' 28264 '
        ' 2580     58170      3235     10912      8668     38181    '
        ' 16542      3005     13152     30382     10186      7729    '
        '  7721     10514     17333  60752099 117157513 '
    )
    row_str = (' 0 1 Missing All ')
    col_str = (
        '      2         3         4         5         6         7      '
        '   8         9        10        11        12        13        14 '
        '       15        16        17        18        19        20 '
        '        21        22        23        24        25        26      '
        '  27        28        29        30        31        32        33  '
        '      34        35        36        37        38        39 '
        '        40        41        42        43        45        47      '
        '  49        50        51        52        54        61       '
        ' 64        66        80       Missing     All'
    )
    answer = stata_results_to_df(
        rows=[r1_str, r2_str, r3_str, r4_str],
        row_names=row_str, col_names=col_str)

    myfunc = agg(
        df=df, row='diabetes_cat', col='lead', weight='finalwgt', missing=True
    )
    myfunc.index = myfunc.index.fillna('Missing')
    myfunc.columns = myfunc.columns.fillna('Missing')
    pd.testing.assert_frame_equal(
        myfunc.fillna(0), answer, check_names=False, check_like=True,
        check_dtype=False, check_index_type=False, check_column_type=False)


if __name__ == '__main__':
    ser_nw_nm_sum()
    ser_w_nm_sum()
    ser_nw_m_sum()
    ser_w_m_sum()
    df_nw_nm_sum()
    df_w_nm_sum()
    df_nw_m_sum()
    df_w_m_sum()

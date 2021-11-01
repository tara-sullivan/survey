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

from _variance import _se
mymod = sys.modules['_variance']
reload(mymod)


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
        else int(i) for i in col_names.split()]
    row_names = [
        i if (i == 'Missing' or i == 'All')
        else int(i) for i in row_names.split()]
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


if __name__ == '__main__':
    pass

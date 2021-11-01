import pandas as pd

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
from _tabulate import tabulate
# from importlib import reload
# reload(svy)

df = pd.read_stata(datapath + 'nhanes2b.dta')

print(
    '----------------------------------------'
    + '----------------------------------------'
)
# print('One variable')
# print('Unweighted (count):')
# svy.tabulate(data=df, row='race', count=True)
# print('Weighted (count):')
# svy.tabulate(data=df, row='race', weight='finalwgt', count=True)
# print('Unweighted (perc):')
# svy.tabulate(data=df, row='race')
# print('Weighted (perc):')
# svy.tabulate(data=df, row='race', weight='finalwgt')

# print(
#     '----------------------------------------'
#     + '----------------------------------------'
# )
# print('Two variables')
# print('Unweighted (count):')
# svy.tabulate(data=df, row='race', col='diabetes', count=True)
# print('Weighted (count):')
# svy.tabulate(
#     data=df, row='race', col='diabetes', count=True, weight='finalwgt')
# print('Unweighted (perc):')
# svy.tabulate(data=df, row='race', col='diabetes')
# print('Weighted (perc):')
# svy.tabulate(data=df, row='race', col='diabetes', weight='finalwgt')

# print(
#     '----------------------------------------'
#     + '----------------------------------------'
# )

# # add one row to dataframe
# new_idx = len(df.index)
# # df.loc[new_idx, 'diabetes'] = 0
# df.loc[new_idx, 'finalwgt'] = 1

# # # Testing numeric diabetes
# svy.tabulate(data=df, row='diabetes', count=True, floatfmt=',.0f')
# svy.tabulate(data=df, row='diabetes', count=False, floatfmt='.4f')
# svy.tabulate(data=df, row='diabetes', weight='finalwgt', count=True,
#          floatfmt=',.0f')
# svy.tabulate(data=df, row='diabetes', weight='finalwgt', count=False,
#          floatfmt='.4f')

# # testing categorical race
# svy.tabulate(data=df, row='race', count=True, floatfmt=',.0f')
# svy.tabulate(data=df, row='race', count=False, floatfmt='.4f')
# svy.tabulate(data=df, row='race', weight='finalwgt', count=True,
#          floatfmt=',.0f')
# svy.tabulate(data=df, row='race', weight='finalwgt', count=False,
#          floatfmt='.4f')

# svy.tabulate(data=df, row='race', col='diabetes', count=True, floatfmt=',.0f')
# svy.tabulate(data=df, row='race', col='diabetes', count=False, floatfmt=',.4f')
# svy.tabulate(data=df, row='race', col='diabetes', weight='finalwgt',
#          count=True, floatfmt=',.0f')
# svy.tabulate(data=df, row='race', col='diabetes', weight='finalwgt',
#          count=False, floatfmt=',.4f')

# svy.tabulate(data=df, col='race', row='diabetes', count=True, floatfmt=',.0f')
# svy.tabulate(data=df, col='race', row='diabetes', count=False, floatfmt=',.4f')
# svy.tabulate(data=df, col='race', row='diabetes', weight='finalwgt',
#          count=True, floatfmt=',.0f')
# svy.tabulate(data=df, col='race', row='diabetes', weight='finalwgt',
#          count=False, floatfmt=',.4f')

# print('Done')

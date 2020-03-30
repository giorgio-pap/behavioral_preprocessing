#!/usr/bin/env python
# coding: utf-8

import random
import pandas as pd
import decimal

from pingouin import linear_regression

dfObj = pd.DataFrame(columns=['Subj_tr', 'OT', 'condition'])

for iterations in range(0,64):
    x = decimal.Decimal(random.randrange(10, 100))/100
    z = float(x)
    dfObj = dfObj.append({'OT': z}, ignore_index=True)

for i, row in dfObj.iterrows():
    if i <= 15:
        dfObj["condition"][i] = "spec"
    elif i <= 31:
        dfObj["condition"][i] = "sub"
    elif i <= 47:
        dfObj["condition"][i] = "rule"
    elif i <= 63:
        dfObj["condition"][i]  = "gen"

df_spec = dfObj[dfObj.values == "spec"]
df_sub = dfObj[dfObj.values  == "sub"]
df_sub = df_sub.reset_index()
df_rule = dfObj[dfObj.values  == "rule"]
df_rule = df_rule.reset_index()
df_gen = dfObj[dfObj.values  == "gen"]
df_gen = df_gen.reset_index()

y = 16 
a = 0

for i in range(0,y):
    a = a +1
    if i < y:
        df_spec.at[i, 'Subj_tr'] = a
        df_sub.at[i, 'Subj_tr'] = a
        df_rule.at[i, 'Subj_tr'] = a
        df_gen.at[i, 'Subj_tr'] = a

frames = [df_spec, df_sub, df_rule, df_gen]
result_df = pd.concat(frames, sort = False)
result_df = result_df.reset_index()
result_df = result_df.drop(['index', "level_0"], axis=1)  # type: object

# use to check complete dataframe (only with Jupyter Notebook)
# if using spyder replace display with print
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(result_df)


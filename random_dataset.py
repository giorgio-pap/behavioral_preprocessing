#!/usr/bin/env python
# coding: utf-8

import random
import pandas as pd
import decimal

dfObj = pd.DataFrame(columns=['Sub_tr', 'OT', 'cond'])

for iterations in range(0,64):
    x = decimal.Decimal(random.randrange(10, 100))/100
    z = float(x)
    dfObj = dfObj.append({'OT': z}, ignore_index=True)

for i, row in dfObj.iterrows():
    if i <= 15:
        dfObj["cond"][i] = "spec"
    elif i <= 31:
        dfObj["cond"][i] = "sub"
    elif i <= 47:
        dfObj["cond"][i] = "rule"
    elif i <= 63:
        dfObj["cond"][i]  = "gen"

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
        df_spec.at[i, 'Sub_tr'] = a
        df_sub.at[i, 'Sub_tr'] = a
        df_rule.at[i, 'Sub_tr'] = a
        df_gen.at[i, 'Sub_tr'] = a

frames = [df_spec, df_sub, df_rule, df_gen]
result = pd.concat(frames, sort = False)
result = result.reset_index()
result = result.drop(['index', "level_0"], axis=1)

# use to check complete dataframe (only with Jupyter Notebook)
# if using spyder replace display with print
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
    display(result)



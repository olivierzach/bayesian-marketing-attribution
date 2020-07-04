import pandas as pd 
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import patsy as pt
import pymc3 as pm
from scipy.stats import beta
from project_functions import *

"""
Course Project: Bayesian Media Mix Modeling
- use bayesian analysis to measure performance of marketing channels

Data:
- Sales leads for a startup SaaS company by channel and subchannel by day
- Sales wins (customer app purchase) of the leads by channel cohorted by day
- Multiple Channels: Google, Facebook, YouTube, Pinterest, Bing, Amazon, Instragram, LinkenIn, Salesforce
- Multiple SubChannels: duckduckgo, g2crowd, capterra, webinar, yandex, blogs
- Channel attribution only available from June 2019 onwards
- Usage of data has been cleared with the company for this project

Goal:
- estimate credible sets for conversion rate by channel and subchannel
- bayesian regression to estimate weight of channel to influence marketing budget
"""

# initialize plotting
plot = False

# read data 
df = pd.read_csv('bayesian_marketing.csv')
print(f"data dimensions: {df.shape}")

# analyze conversion rate by channel
df_channel = df[['closed_won', 'ft_channel_c']]
df_channel['count'] = 1.0
analyze_channel(df_channel, plot=False)

# analyze conversion rate by sub channel
df_channel = df[['closed_won', 'ft_subchannel_c']]
df_channel['count'] = 1.0
analyze_subchannel(df_channel, plot=False)

# clean data for modeling
drop_cols = [
    'oppty_id',
    'converted_opportunity_id'
]
df.drop(drop_cols, axis=1, inplace=True)

# set index to reference date, id later
df.set_index(['created_date', 'lead_id'], inplace=True)

# convert categorical channels to numeric dummy variables
df = pd.get_dummies(df).reset_index()
print(f"data dimensions after casting: {df.shape}")
print(f"global win rate: {np.mean(df.closed_won)}")

# filter out features with low representation
for i in df.columns[2:]:
    if np.mean(df[i]) < .01:
        print(f"dropping variable {i}")
        df.drop(i, axis=1, inplace=True)
print(f"data dimensions after dropping: {df.shape}")

# provide intercept for model
df['intercept'] = 1.0

# tuck in meta columns for modeling
df = df.set_index(['created_date', 'lead_id'])

# convert to float for model
for i in df.columns:
    df[i] = df[i].astype(float)

# fit models and output results
channel_subchannel_model(df=df, plot=False)
channel_model(df=df, plot=False)
subchannel_model(df=df, plot=False)

if __name__ == "bayesian_marketing":
    bayesian_marketing()

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import patsy as pt
import pymc3 as pm
from scipy.stats import beta
from project_functions import *

"""
Course Project: Bayesian Media Mix Modeling
- use bayesian analysis to measure performance of marketing channels

Data:
- Sales leads for a startup SaaS company by channel by day
- Sales wins (customer app purchase) of the leads by channel cohorted by day
- Multiple Channels: Google, Facebook, YouTube, Pinterest, Bing, Amazon, Instragram, LinkenIn, Salesforce
- Up to two years of data available, but dependent on channel
- Usage of data has been cleared with the company for this project

Goal:
- estimate credible sets for conversion rate by channel
- bayesian regression to estimate weight of channel to influence marketing budget
"""

# read data 
df = pd.read_csv('marketing_wins.csv')
print(f"data dimensions: {df.shape}")

# clean data
df = clean_data(df)

df.columns

# filter to wins data for quick analysis
df_wins = df.filter(regex="wins")
print(f"average wins by channel: \n {df_wins.mean()}")

# filter to leads data for analysis
df_leads = df.filter(regex='_leads')
print(f"average leads by channel: \n {df_leads.mean()}")
print(f"missing values by channel: {df_leads.isna().mean()}")

# define channels
channel_list = [
    'google',
    'facebook',
    'bing',
    'youtube',
    'pinterest',
    'inst',
    'linkedin',
    'salesforce'
]

# set up storage
df_sets = pd.DataFrame()

# calculate credible sets for conversion rates
for i in channel_list:

    # filter data to prepare for credible set calculation
    df_cs = df.filter(regex=i).dropna()
    df_cs[i + '_losses'] = df_cs.iloc[:, 0] - df_cs.iloc[:, 1]
    df_cs['grouper'] = i
    df_cs = df_cs.groupby('grouper').sum().reset_index()
    df_cs.columns = ['grouper', 'leads', 'wins', 'losses']

    # using the binomial-beta conjugate pair
    # posterior reduces to beta(alpha = 1 + wins, losses = 1+losses)
    df_cs['jeffrey_alpha'] = 1 + df_cs['wins']
    df_cs['jeffrey_beta'] = 1 + df_cs['losses']

    # credible set calculation
    df_cs['lower'] = beta.ppf(
        q=[.025],
        a=df_cs['jeffrey_alpha'],
        b=df_cs['jeffrey_beta']
    )

    df_cs['upper'] = beta.ppf(
        q=[.975],
        a=df_cs['jeffrey_alpha'],
        b=df_cs['jeffrey_beta']
    )

    df_sets = pd.concat([df_sets, df_cs], axis=0)

# melt results for easier plotting
df_sets_grouped = pd.melt(
    df_sets[['grouper', 'lower', 'upper']],
    id_vars='grouper',
    value_vars=['lower', 'upper'],
    var_name='cs_band',
    value_name='value'
).sort_values('value', ascending=False)

# plot the credible sets
plt.figure(figsize=(15, 10))
p2 = sns.pointplot(
    x='value',
    y='grouper',
    hue='grouper',
    palette='deep',
    data=df_sets_grouped
)
p2.legend_.remove()
p2.set_title('95% Credible Set Conversion Rate by Marketing Channel')
p2.set(ylabel='Marketing Channel', xlabel='Conversion Rate')
plt.show()


df['intercept'] = 1.0
df = df.dropna().drop('total_leads', axis=1)

for i in df.columns[1:-2]:
    df[i] = df[i].astype(float)

# define model
with pm.Model() as mix_marketing:

    # priors
    b0 = pm.Normal('intercept', mu=0, sigma=1)
    b1 = pm.Normal('google_leads', mu=0, sigma=1)
    b2 = pm.Normal('facebook_leads', mu=0, sigma=1)
    b10 = pm.Normal('google:facebook', mu=0, sigma=1)
    b3 = pm.Normal('youtube_leads', mu=0, sigma=1)
    b4 = pm.Normal('pinterest_leads', mu=0, sigma=1)
    b5 = pm.Normal('bing_leads', mu=0, sigma=1)
    b6 = pm.Normal('instagram_leads', mu=0, sigma=1)
    b7 = pm.Normal('linkedin_leads', mu=0, sigma=1)
    b9 = pm.Normal('salesforce_leads', mu=0, sigma=1)

    # define model with exp link
    theta = (
        b1*df['google_leads'] + 
        b2*df['facebook_leads'] +
        b10 * (df['google_leads'] * df['facebook_leads']) +
        b3*df['youtube_leads'] + 
        b4*df['pinterest_leads'] + 
        b5*df['bing_leads'] + 
        b6*df['instagram_leads'] + 
        b7*df['linkedin_leads'] + 
        b9*df['salesforce_leads']
    )

    # poisson likelihood - modeling counts of wins
    y = pm.Poisson(
        'y',
        mu=np.exp(theta),
        observed=df.dropna()['total_wins'].astype(float).values
    )

with mix_marketing:
    trc_mix = pm.sample(1000, tune=1000, init='adapt_diag')


rvs_mix = [rv.name for rv in strip_derived_rvs(mix_marketing.unobserved_RVs)]
plot_traces_pymc(trc_mix, varnames=rvs_mix)

np.exp(pm.summary(trc_mix, varnames=rvs_mix)[['mean','hpd_2.5','hpd_97.5']])



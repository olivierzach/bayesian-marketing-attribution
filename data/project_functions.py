import re
import pandas as pd
import pymc3 as pm
# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.stats import beta
import numpy as np

def clean_data(df):

    # convert problem dtypes
    for i in df.columns[1:-1]:
        if df[i].dtype == 'object':
            df[i] = (
                df[i].astype(str).str.replace(" ", "").str.replace(",", "").astype(float)
            )

    # rename columns
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df.rename(columns={'instragram_wins': "instragam_wins"}, inplace=True)
    df.columns

    return df

def strip_derived_rvs(rvs):
    '''Convenience fn: remove PyMC3-generated RVs from a list'''
    ret_rvs = []
    for rv in rvs:
        if not (re.search('_log',rv.name) or re.search('_interval',rv.name)):
            ret_rvs.append(rv)
    return ret_rvs


def plot_traces_pymc(trcs, varnames=None):
    ''' Convenience fn: plot traces with overlaid means and values '''

    nrows = len(trcs.varnames)
    if varnames is not None:
        nrows = len(varnames)

    ax = pm.traceplot(trcs, var_names=varnames, figsize=(12,nrows*1.4),
                      lines=tuple([(k, {}, v['mean'])
                                   for k, v in pm.summary(trcs, varnames=varnames).iterrows()]))

    for i, mn in enumerate(pm.summary(trcs, varnames=varnames)['mean']):
        ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data',
                         xytext=(5,10), textcoords='offset points', rotation=90,
                         va='bottom', fontsize='large', color='#AA0022')


def clean_multi_index_headers(df):
    """
    Concatenates a multi-index columns headers into one with a clean format
    Parameters
    ----------
    df : DataFrame
    Returns
    -------
    DataFrame
    """

    df.columns = ['_'.join(col).strip() for col in df.columns.values]

    return df

def analyze_subchannel(df_channel, plot=False):

    # group by channel to get rate, wins, losses by channel
    df_channel = df_channel.groupby('ft_subchannel_c').agg(
        {'closed_won': ['mean', 'sum'], 'count': 'sum'}
        )

    # tidy up headers
    df_channel = clean_multi_index_headers(df_channel).reset_index()

    # explictly calculate the losses
    df_channel['closed_lost_sum'] = (
        df_channel['count_sum'] - df_channel['closed_won_sum']
    )

    # using the binomial-beta conjugate pair
    # posterior reduces to beta(alpha = 1 + wins, losses = 1+losses)
    df_channel['jeffrey_alpha'] = 1 + df_channel['closed_won_sum']
    df_channel['jeffrey_beta'] = 1 + df_channel['closed_lost_sum']

    # credible set calculation
    df_channel['lower'] = beta.ppf(
        q=[.025],
        a=df_channel['jeffrey_alpha'],
        b=df_channel['jeffrey_beta']
    )

    df_channel['upper'] = beta.ppf(
        q=[.975],
        a=df_channel['jeffrey_alpha'],
        b=df_channel['jeffrey_beta']
    )

    print(f'Marketing SubChannel Credible Sets \n {df_channel}')

    # melt results for easier plotting
    df_channel_melt = pd.melt(
        df_channel[['ft_subchannel_c', 'lower', 'upper']],
        id_vars='ft_subchannel_c',
        value_vars=['lower', 'upper'],
        var_name='cs_band',
        value_name='value'
    ).sort_values('value', ascending=False)

    if plot:

        # plot the credible sets
        plt.figure(figsize=(15, 10))
        p2 = sns.pointplot(
            x='value',
            y='ft_subchannel_c',
            hue='ft_subchannel_c',
            palette='deep',
            data=df_channel_melt
        )
        p2.legend_.remove()
        p2.set_title('95% Credible Set Conversion Rate by Marketing Sub-Channel')
        p2.set(ylabel='Marketing Channel', xlabel='Conversion Rate')
        plt.show()

        # plot the credible sets
        plt.figure(figsize=(15, 10))
        p4 = sns.pointplot(
            x='value',
            y='ft_subchannel_c',
            hue='ft_subchannel_c',
            palette='deep',
            data=df_channel_melt[df_channel_melt.value <= .2]
        )
        p4.legend_.remove()
        p4.set_title('95% Credible Set Conversion Rate by Marketing Sub-Channel Filtered by Volume')
        p4.set(ylabel='Marketing Channel', xlabel='Conversion Rate')
        plt.show()

def analyze_channel(df_channel, plot=False):

    # group by channel to get rate, wins, losses by channel
    df_channel = df_channel.groupby('ft_channel_c').agg(
        {'closed_won': ['mean', 'sum'], 'count': 'sum'}
        )

    # tidy up headers
    df_channel = clean_multi_index_headers(df_channel).reset_index()

    # explictly calculate the losses
    df_channel['closed_lost_sum'] = (
        df_channel['count_sum'] - df_channel['closed_won_sum']
    )

    # using the binomial-beta conjugate pair
    # posterior reduces to beta(alpha = 1 + wins, losses = 1+losses)
    df_channel['jeffrey_alpha'] = 1 + df_channel['closed_won_sum']
    df_channel['jeffrey_beta'] = 1 + df_channel['closed_lost_sum']

    # credible set calculation
    df_channel['lower'] = beta.ppf(
        q=[.025],
        a=df_channel['jeffrey_alpha'],
        b=df_channel['jeffrey_beta']
    )

    df_channel['upper'] = beta.ppf(
        q=[.975],
        a=df_channel['jeffrey_alpha'],
        b=df_channel['jeffrey_beta']
    )

    # metric to show the "confidence" of each conversion rate
    df_channel['cs_diff'] = df_channel['upper'] - df_channel['lower']

    print(f'Marketing Channel Credible Sets \n {df_channel}')

    # sort by confidence and then conversion rate
    df_channel_sorted = df_channel.sort_values(
        ['cs_diff', 'closed_won_mean'],
        ascending=True
    )

    # melt results for easier plotting
    df_channel_melt = pd.melt(
        df_channel[['ft_channel_c', 'lower', 'upper']],
        id_vars='ft_channel_c',
        value_vars=['lower', 'upper'],
        var_name='cs_band',
        value_name='value'
    ).sort_values('value', ascending=False)

    if plot:

        # plot the best conversion rates by confidence
        plt.figure(figsize=(15, 10))
        sns.barplot(
            x='closed_won_mean',
            y='ft_channel_c',
            data=df_channel_sorted[df_channel_sorted.cs_diff < .1]
        ).set_title('Conversion Credible Sets order by Credible Set Band Length')
        plt.show()

        # plot the credible sets
        plt.figure(figsize=(15, 10))
        p2 = sns.pointplot(
            x='value',
            y='ft_channel_c',
            hue='ft_channel_c',
            palette='deep',
            data=df_channel_melt
        )
        p2.legend_.remove()
        p2.set_title('95% Credible Set Conversion Rate by Marketing Channel')
        p2.set(ylabel='Marketing Channel', xlabel='Conversion Rate')
        plt.show()

        plt.figure(figsize=(15, 10))
        p3 = sns.scatterplot(
            x='lower',
            y='upper',
            hue='ft_channel_c',
            size='count_sum',
            sizes=(20, 3400),
            x_jitter=True,
            y_jitter=True,
            data=df_channel
        )
        p3.legend_.remove()
        p3.set_title('95% Credible Set Conversion Bands by Total Channel Volume')
        p3.set(ylabel='Upper Credible Set Band', xlabel='Lower Credible Set Band')

        for line in range(0,df_channel.shape[0]):
            p3.text(
                df_channel.lower[line]+.0003,
                df_channel.upper[line],
                df_channel.ft_channel_c[line],
                horizontalalignment='left',
                size='small', 
                color='white', 
                weight='semibold'
            )

        plt.show()

        # plot the credible sets
        plt.figure(figsize=(15, 10))
        p4 = sns.pointplot(
            x='value',
            y='ft_channel_c',
            hue='ft_channel_c',
            palette='deep',
            data=df_channel_melt[df_channel_melt.value <= .15]
        )
        p4.legend_.remove()
        p4.set_title('95% Credible Set Conversion Rate by Marketing Channel Filtered by Volume')
        p4.set(ylabel='Marketing Channel', xlabel='Conversion Rate')
        plt.show()


def channel_subchannel_model(df, plot=False):

    # define channel and subchannel model
    with pm.Model() as logit_marketing:

        # priors
        cpc = pm.Normal('ft_channel_c_cpc', mu=0, sigma=1)
        direct = pm.Normal('ft_channel_c_direct', mu=0, sigma=1)
        display = pm.Normal('ft_channel_c_display', mu=0, sigma=1)
        organic = pm.Normal('ft_channel_c_organic', mu=0, sigma=1)
        paidsocial = pm.Normal('ft_channel_c_paidsocial', mu=0, sigma=1)
        paidvideo = pm.Normal('ft_channel_c_paidvideo', mu=0, sigma=1)
        partner = pm.Normal('ft_channel_c_partner', mu=0, sigma=1)
        facebook = pm.Normal('ft_subchannel_c_facebook', mu=0, sigma=1)
        gdn = pm.Normal('ft_subchannel_c_gdn', mu=0, sigma=1)
        google = pm.Normal('ft_subchannel_c_google', mu=0, sigma=1)
        backlink = pm.Normal('ft_subchannel_c_link', mu=0, sigma=1)
        salesforce = pm.Normal('ft_subchannel_c_salesforce', mu=0, sigma=1)
        youtube = pm.Normal('ft_subchannel_c_youtube', mu=0, sigma=1)

        # define model with logit link
        theta = (
            cpc*df['ft_channel_c_cpc'] + 
            direct*df['ft_channel_c_direct'] +
            display * df['ft_channel_c_display'] + 
            organic*df['ft_channel_c_organic'] + 
            paidsocial*df['ft_channel_c_paidsocial'] + 
            partner*df['ft_channel_c_partner'] + 
            facebook*df['ft_subchannel_c_facebook'] + 
            gdn*df['ft_subchannel_c_gdn'] + 
            google*df['ft_subchannel_c_google'] +
            backlink*df['ft_subchannel_c_link'] + 
            salesforce*df['ft_subchannel_c_salesforce'] + 
            youtube*df['ft_subchannel_c_youtube']
        )

        # poisson likelihood - modeling counts of wins
        likelihood = pm.Bernoulli(
            'y',
            pm.math.sigmoid(theta),
            observed=df['closed_won'].astype(float).values
        )

    with logit_marketing:
        trc_mix = pm.sample(1000, tune=1000)

    # show inference trace plots
    if plot:
        pm.traceplot(trc_mix)
        plt.show()

    # print logistic regression results
    full_model_results = pd.DataFrame(pm.summary(trc_mix))
    key_vals = np.exp(full_model_results[['mean', 'sd', 'hpd_2.5', 'hpd_97.5']])
    print(f"Logistic Regression Results: Channel and SubChannel Model: \n {key_vals}")


def channel_model(df, plot=False):

    # define channel model
    with pm.Model() as logit_marketing:

        # priors
        # int_ = pm.Normal('intercept', mu=0, sigma=1)
        cpc = pm.Normal('ft_channel_c_cpc', mu=0, sigma=1)
        direct = pm.Normal('ft_channel_c_direct', mu=0, sigma=1)
        display = pm.Normal('ft_channel_c_display', mu=0, sigma=1)
        organic = pm.Normal('ft_channel_c_organic', mu=0, sigma=1)
        paidsocial = pm.Normal('ft_channel_c_paidsocial', mu=0, sigma=1)
        paidvideo = pm.Normal('ft_channel_c_paidvideo', mu=0, sigma=1)
        partner = pm.Normal('ft_channel_c_partner', mu=0, sigma=1)

        # define model with logit link
        theta = (
            # int_ + 
            cpc*df['ft_channel_c_cpc'] + 
            direct*df['ft_channel_c_direct'] +
            display * df['ft_channel_c_display'] + 
            organic*df['ft_channel_c_organic'] + 
            paidsocial*df['ft_channel_c_paidsocial']
        )

        # poisson likelihood - modeling counts of wins
        likelihood = pm.Bernoulli(
            'y',
            pm.math.sigmoid(theta),
            observed=df['closed_won'].astype(float).values
        )

    with logit_marketing:
        trc_mix = pm.sample(1000, tune=1000)

    # show inference trace plots
    if plot:
        pm.traceplot(trc_mix)
        plt.show()

    # print logistic regression results
    full_model_results = pd.DataFrame(pm.summary(trc_mix))
    key_vals = np.exp(full_model_results[['mean', 'sd', 'hpd_2.5', 'hpd_97.5']])
    print(f"Logistic Regression Results: Channel Model: \n {key_vals}")

def subchannel_model(df, plot=False):
    # define subchannel model
    with pm.Model() as logit_marketing:

        # priors
        # int_ = pm.Normal('intercept', mu=0, sigma=1)
        facebook = pm.Normal('ft_subchannel_c_facebook', mu=0, sigma=1)
        gdn = pm.Normal('ft_subchannel_c_gdn', mu=0, sigma=1)
        google = pm.Normal('ft_subchannel_c_google', mu=0, sigma=1)
        backlink = pm.Normal('ft_subchannel_c_link', mu=0, sigma=1)
        salesforce = pm.Normal('ft_subchannel_c_salesforce', mu=0, sigma=1)
        youtube = pm.Normal('ft_subchannel_c_youtube', mu=0, sigma=1)

        # define model with logit link
        theta = (
            facebook*df['ft_subchannel_c_facebook'] + 
            gdn*df['ft_subchannel_c_gdn'] + 
            google*df['ft_subchannel_c_google'] +
            backlink*df['ft_subchannel_c_link'] + 
            salesforce*df['ft_subchannel_c_salesforce'] + 
            youtube*df['ft_subchannel_c_youtube']
        )

        # poisson likelihood - modeling counts of wins
        likelihood = pm.Bernoulli(
            'y',
            pm.math.sigmoid(theta),
            observed=df['closed_won'].astype(float).values
        )

    with logit_marketing:
        trc_mix = pm.sample(1000, tune=1000)

    # show inference trace plots
    if plot:
        pm.traceplot(trc_mix)
        plt.show()

    # print logistic regression results
    full_model_results = pd.DataFrame(pm.summary(trc_mix))
    key_vals = np.exp(full_model_results[['mean', 'sd', 'hpd_2.5', 'hpd_97.5']])
    print(f"Logistic Regression Results: SubChannel Model: \n {key_vals}")


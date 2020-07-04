# Bayesian Marketing Attribution

Project repository for ISYE 6420 Bayesian Modeleing and Inference course. Goal of this project was to apply Bayesian analysis to a real-world data set. The data for this project focused on Marketing Attribution for a SaaS company. Aim was to determine, through Bayesian analysis, which marketing channels are most valuable (worth investing more in) to sales conversion.

Project write-up:  
  - `project_writeup.pdf`: https://github.com/olivierzach/bayesian_marketing_attribution/blob/master/references/project_writeup.pdf

# Data Summary

Available data is an anonymized set of website leads (website signups) with flags indicating if the lead eventually purchased a product (wins) or did not purchase a product (losses). The data is broken out by the originating lead source including marketing channel and subchannel. 

  - Marketing Channels: CPC, Direct, Display, Organic Search, Paid Social, Paid Video, Referrals
  - Marketing Subchannels: Facebook, Google, Affiliate, SalesForce, YouTube, Linkedin

Data is available in the `data/bayesian_marketing.csv` directory. 

Goal is to determine which mix of channels and subchannels is best to maximize the amount of purchases (wins) to inform marketing investment. 


# Why Bayes?

Why use Bayesian methods to solve this problem? There are a few challenges with the data that make Bayesian analysis a great choice for this analysis. 

  - `Data Availablility`: only 6 months of data is available from this early stage startup, there is not much historical data available to perform classical analysis
  - `Variable Investment and Channels`: channel investment and rate are variable through the dataset, new marketing channels may have a few weeks of data, older channels may have dropped out, and the rate of investment has changed drastically for channels available since the start of the dataset 

Due to these challenges - marketing channels with very little data (weeks or even days of observations) may show 50%+ conversion rates - is this real or just noise? Standard lead sources, such as Facebook, may show a 1% conversion rate. Should marketing divest 100% of the Facebook budget to this new marketing channel? 

Bayesian Analysis helps sort through these challeges to accurately determine the best investment for conversion growth. 

# Analysis

Data was analyzed in two stages: 

First use Bayesian conjugate pairs and credible sets to give probabalistic estimates of conversion rate by channel and subchannel. Then model the probability of a conversion based on channel and subchannel using Bayesian regression methods. 

Bayesian Conversion Rate Estimates:
  - Conversion rates by channel and subchannel will take advantage of the `Binomial-Beta conjugate pair`, which allows us to sample from and build credible sets directory from the Beta posterior distribution
  - Analysis will use the `non-informative Jeffery's Prior` to allow the sample size of each lead source to inform the length of the credible sets
  - **Ranking channels by conversion rate estimate and credible set length will show which we have the highest confidence in and eliminate channels with high conversion points estimates by wide credible set intervals**
  - See `analyze_channel()` and `analyze_subchannel()` methods available in the `data/project_functions.py` script for details


Bayesian Modeling:
  - Multiple models were developed to estimate the channels and subchannels that result in the highest odds of conversion
  - Each model will be a Logistic Regression fit on a combined set of channels and subchannels, channel only, and subchannel only
  - Bayesian Regression fits the model through simulation and conditional estimation of the model's coefficients. Bayesian regression allows us to encode a prior into the coefficient estimates to help with the small data challenge. 
  - Models were fit using the Bayesian framework provided by `pymc3`
  - Details of the methodology can be found in the `channel_subchannel_model()`, `channel_model()`, `subchannel_model()` method within the `data/project_functions.py` script
  - Results from the model inference will recommend the best channels to invest in based on probability of conversion

# Results

Bayesian analysis immediately proves to be useful. Looking at conversion rates only, we would see extraordinary performance from channels like PR, Advocacy, Sales Rep, Employee and Tradeshow - all have greater than 30% conversion rates. Channels like Youtube and Amazon also look to have extremely high conversion rates in isolation. 

Examining the credible sets of these same channels we can see the lack of confidence in the conversion estimates - intervals can sometimes stretch between 1% and 90% conversion rates for channels with small amounts of data. 

Using the same credible sets we can filter to the channels that have the highest conversion estimate and the shortest intervals from lower to upper estimates using the equi-tailed posterior density. This techniques highlights CPC (4%), Chat (12%), and Partner (3%) leads as high conversion channels with high confidence. 

The same analysis was applied to the marketing subchanels. Analysis highlighted subchannels such as DuckDuckGo and YouTube as channels with high conversion estimate but low confidence.

Results from the models provided the following inference: 
  - `Channel / Subchannel Model`: Paid Video, Google, Affiliate Back-Links, SalesForce channels had the largest conditional affect on conversion odds, with "significant" coefficient posterior densities (credible interval does not contain zero)
  - `Channel Model`: Paid Video and Partner had largest conditional affect on conversion odds
  - `Subchannel Model`: Salesforce, Affiliate Back-Links, Google were the most important variables in estimating conversion odds
  

# Conclusions

Based on this analysis we can easily determine which marketing channels and subchannels are noise and which channels can add value to win conversion. Most have wide credible set intervals showing the lack of confidence in the conversion rate estimates. Among the channels we are actually confident in (narrow credible set intervals) we identify Chat, CPC, Partner, Organic leads as having the highest relative conversion rates. Model experiments identify Paid Video, Google, Affiliate Back-Links, Salesforce leads as having the highest overall contribution to conversion odds. 

From this analysis we can recommend investing more in the above channels while divesting in channels such as GDN, YouTube, and Facebook with confidence. 



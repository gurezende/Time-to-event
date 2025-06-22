#%%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from lifelines import GeneralizedGammaFitter, LogNormalFitter, KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines.plotting import plot_lifetimes
#%%

# Get Data
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
online_retail = fetch_ucirepo(id=352) 
  
# Data (as pandas dataframes) 
X = online_retail.data.features 
y = online_retail.data.targets 

#%%
#To pandas df
df = pd.concat([X,y], axis=1)

# %%

# Data cleaning
# Fill NA Description with - 
# Drop Customers == NA
# Invoice to date time
# Drop NAs from customerID
df['Description'] = df['Description'].replace({np.nan:'-'})
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df.dropna(subset='CustomerID', inplace=True)


# Running some basic information about the dataset
print('-- Meta Data UPDATED--\n')
print(f'Data shape: {df.shape}\n')
print('------------------')
print(df.info(), '\n')
print('------------------')
print(f'Null values:\n{df.isna().sum()}')


# %%

class PrepareData:
    def __init__(self, df):
        self.df = df

    def prepare_data(self):
    # Aggregaring by Unique Invoice and customer
        by_invoice = (
            self.df
            .groupby(['CustomerID','InvoiceDate', 'Country'])
            .agg({'Quantity': 'sum','UnitPrice':'sum', 'Description':'nunique'})
            .reset_index()
            .sort_values(by=['CustomerID', 'InvoiceDate'])
        )

        # Add time (in days) between transactions
        by_invoice['time_btw_trx'] = (
            by_invoice
            .sort_values(by=['CustomerID', 'InvoiceDate'])
            .groupby(['CustomerID'])
            .InvoiceDate
            .diff(1)
            .dt.days
            .fillna(0)
            .astype(int)
        )

        # Check number of transactions
        num_trx_by_customer = (
            by_invoice
            .groupby('CustomerID')
            .InvoiceDate
            .count()
            .reset_index()
            .rename(columns={'InvoiceDate':'trx_ct'})
            .sort_values('trx_ct')
        )

        # Last purchase date
        last_purchase_date = (
            by_invoice
            .groupby('CustomerID')
            .InvoiceDate
            .max()
            .reset_index()
        )

        # Last Purchase Date | Days since last purchase | number of trx
        customer_profile = (
            last_purchase_date
            .merge(by_invoice[['CustomerID','InvoiceDate']],
                on= ['CustomerID', 'InvoiceDate'])
            .merge(num_trx_by_customer,
                on= ['CustomerID'])
            .sort_values(by= ['trx_ct'])
            .reset_index(drop=True)
            .assign(days_since_last_trx = lambda x: (datetime(2011,12,31) - x['InvoiceDate']).dt.days.astype(int))
        )

        # Return
        return customer_profile

    def churn_rule(self, customer_profile):
        # Create a rule to churn or not churn
        # Rule: Customers that purchased last time before 180 days are churned
        # churn_date = customer_profile.InvoiceDate.max() - timedelta(days=180)
        churn_last_shop = 180
        print(f'Rule: Customers that purchased last time before {churn_last_shop} are churned')

        # Setting up the rule
        cond = [
            (customer_profile.days_since_last_trx > churn_last_shop) 
        ]

        vals = [1]

        # Applying the rule
        customer_profile['churn'] = np.select(cond, vals, default=0)

        # Return
        return customer_profile



#%%

### Instance of Data Wrangler ### 
dw = PrepareData(df)

### CREATE CUSTOMER PROFILE ### 
customer_profile = dw.prepare_data()

### CHURN RULE ### 
customer_profile = dw.churn_rule(customer_profile)

#%%

# Checking the status of the customers against their timelines
time = customer_profile['days_since_last_trx'].sample(50, replace=False)
status = customer_profile['churn'][time.index]
plt.figure(figsize=(15, 12));
plot_lifetimes(time, status)
plt.xlabel('Days Since Last Transactions');
plt.ylabel('Customer ID');
plt.title('Customer Churn with lifelines');

#%%

### FINDING THE BEST DISTRIBUTION FOR THE TIME TO EVENT ####

from distfit import distfit

# Instance of distfit
dfit = distfit()

# Fitting the data
results = dfit.fit_transform(customer_profile['days_since_last_trx'],
                             verbose=0)

# Display Top 3 Best fits
results['summary'].head(3)



#%%

### RUNNING SURVIVAL ANALYSIS ESTIMATORS ####

'''
Since we have the distribution of the time to event, we can use 
the lifelines package to estimate the Survival Curve for these distributions
* Kaplan-Meier: because it is non-parametric and serves well most cases
* Lognormal: because it is the best fit for time to event
* Gamma: because it is the second best fit for time to event
'''

# KaplanMeier Estimator (non-parametric)
kmf = KaplanMeierFitter()
kmf.fit(customer_profile['days_since_last_trx'], 
        customer_profile['churn'])
kmf.plot_survival_function(figsize=(15, 7));

ggf = GeneralizedGammaFitter(alpha=0.05)
ggf.fit(customer_profile['days_since_last_trx'],
        customer_profile['churn'].astype(bool))
ggf.plot_survival_function(figsize=(15, 7));

lnf = LogNormalFitter(alpha=0.05)
lnf.fit(customer_profile['days_since_last_trx'],
        customer_profile['churn'].astype(bool))
lnf.plot_survival_function(figsize=(15, 7));

print(f'Kaplan-Meier Median Survival Time: {kmf.median_survival_time_}')
print(f'Gamma Median Survival Time: {ggf.median_survival_time_}')
print(f'LogNormal Median Survival Time: {lnf.median_survival_time_}')

#%%

# Customers with an estimated Survival Time <= 0.25 need action to avoid Churn
risk_of_churn = lnf.survival_function_.query('LogNormal_estimate <= 0.5').head(1).index.astype(int).values[0]
print(risk_of_churn)

preds = 1- lnf.survival_function_at_times(customer_profile['days_since_last_trx'])
customer_profile['churn_pred'] = preds.values


# %%
customer_profile.query('churn_pred >= 0.5')
# %%
customers_to_churn = customer_profile.query('churn_pred >= 0.5')['CustomerID'].unique().tolist()
# %%
# df.query('CustomerID.isin(@customers_to_churn)').to_json('churned_customers.json', orient='records')
df.query('CustomerID.isin(@customers_to_churn)').to_csv('churned_customers.csv', index=False)
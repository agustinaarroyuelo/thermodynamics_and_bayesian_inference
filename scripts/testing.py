import pymc3 as pm
import pandas as pd
import numpy as np

data = pd.read_csv('Howell1.csv', sep=';')
data = data[data.age >= 18]
data.head()
hpds_alfa = []
hpds_beta = []
hpds_sigma = []

for i in range(20):
    with pm.Model() as howell:
        #priors
        α = pm.Normal('α', mu=172, sd=100)
        β = pm.Normal('β', mu=0, sd=10)
        σ = pm.HalfNormal('σ', sd=100)

        #linear model
        μ = pm.Deterministic('μ', α + β * data.weight[:i])    

        # likelihood
        height = pm.Normal('height', mu=μ, sd=σ, observed=data.height[:i])
        
        # sample
        trace = pm.sample(1000)
        #hpds_alfa.append(pm.hpd(trace['α']))
        #hpds_beta.append(pm.hpd(trace['β']))
        #hpds_sigma.append(pm.hpd(trace['σ']))

        hpds_alfa.append(np.abs(trace['α'].mean()-114.071771))
        hpds_beta.append(np.abs(trace['β'].mean()-0.903689))
        hpds_sigma.append(np.abs(trace['σ'].mean()-5.100019))

hpds_a = np.array(hpds_alfa)
hpds_b = np.array(hpds_beta)
hpds_s = np.array(hpds_sigma)
np.save('mean_testing_a_20.npy', hpds_a)
np.save('mean_testing_b_20.npy', hpds_b)
np.save('mean_testing_s_20.npy', hpds_s)
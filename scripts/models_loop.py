import pymc3 as pm
import pandas as pd
import numpy as np

data = pd.read_csv('Howell1.csv', sep=';')
data = data[data.age >= 18]
data.head()
hpds = []

for i in range(len(data.height)):
    with pm.Model() as howell:
        #priors
        α = pm.Normal('α', mu=172, sd=100)
        β = pm.Normal('β', mu=0, sd=10)
        σ = pm.Uniform('σ', lower=0, upper=100)

        #linear model
        μ = pm.Deterministic('μ', α + β * data.weight[:i])    

        # likelihood
        height = pm.Normal('height', mu=μ, sd=σ, observed=data.height[:i])
        
        # sample
        trace = pm.sample(100)
        hpds.append(pm.hpd(trace['β']))

hpds = np.array(hpds)
np.save(hpds)
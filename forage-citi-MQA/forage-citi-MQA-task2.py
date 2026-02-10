# this script is intended to create a fair price of coffee future contracts using the cost of carry
# model, Black Scholes model, and Monte Carlo simulations

import numpy as np
from scipy.stats import norm

## cost of carry model

# given
s_t = 1.20
r = .02 # per annum
d = .01 # per annum
T = .5 # in years
X = 1.25 # strike price
sigma = .25

# write out cost of carry model equation andprint result
F_t = s_t * np.exp((r + d) * T)
print(f'The proper price for the coffee futures contract is ${F_t:.2f} / lb')

## Black Scholes Model

# calculate d1
d1 = (np.log(s_t / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
# calculate d2
d2 = d1 - sigma * np.sqrt(T)

# calculate call option and print result
C = s_t * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
print(f'The call option price is ${C:.2f}.')

## Monte Carlo Simulation

# given
num_sims = 10000
num_steps = 252

# determine time increment
dt = T / num_steps

# simulate price paths
np.random.seed(42)  
price_paths = np.zeros((num_steps, num_sims))
price_paths[0] = s_t
for t in range(1, num_steps):
    z = np.random.standard_normal(num_sims)
    price_paths[t] = price_paths[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

# calculate average simulated price at maturity
avg_price = np.mean(price_paths[-1])
print(f'The average price based on the simulations for coffee futures is ${avg_price:.2f}.')







import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm

def option_payoff(S, K, option_type):
    payoff = 0.0
    if option_type == "call":
        payoff = max(S - K, 0)
    elif option_type == "put":
        payoff = max(K - S, 0)
    return payoff

S0 = 100
K = 100
T = 1.0
r = 0.06
q = 0.06
sigma = 0.35
option_type = "call"
n_steps = 100 # number of steps
np.random.seed(102623) # make result reproducable
n_simulation = 4000 # number of simulation
dt = T/n_steps
sqrt_dt = np.sqrt(dt)

payoff = np.zeros((n_simulation), dtype = float)
step = range(0, int(n_steps), 1)

for i in range(0, n_simulation):
    ST = S0
    for j in step:
        epsilon = np.random.normal()
        ST *= np.exp((r - q - 0.5*sigma*sigma)*dt +
        sigma*epsilon*sqrt_dt)

    payoff[i] = option_payoff(ST, K, option_type)

option_variance = np.var(payoff*np.exp(-r*T))
option_price = np.mean(payoff)*np.exp(-r*T)
# print(option_type + ' price =', round(option_price, 8))
# print(option_type + ' variance =', round(option_variance, 8))



payoff = np.zeros((n_simulation), dtype = float)
step = range(0, int(n_steps), 1)

for i in range(0, n_simulation):
    ST1 = S0
    ST2 = S0
    for j in step:
        epsilon = np.random.normal()
        ST1 *= np.exp((r - q - 0.5*sigma*sigma)*dt +
        sigma*epsilon*sqrt_dt)
        ST2 *= np.exp((r - q - 0.5*sigma*sigma)*dt -
        sigma*epsilon*sqrt_dt)

    payoff[i] = (option_payoff(ST1, K, option_type)+option_payoff(ST2, K, option_type))/2

option_variance = np.var(payoff*np.exp(-r*T))
option_price = np.mean(payoff)*np.exp(-r*T)

# print(option_type + ' price =', round(option_price, 8))
# print(option_type + ' variance =', round(option_variance, 8))


payoff = np.zeros((n_simulation), dtype = float)
step = range(0, int(n_steps), 1)
risk_neutral_ST = S0*np.exp((r-q)*T)

for i in range(0, n_simulation):
    ST = S0

    for j in step:
        epsilon = np.random.normal()
        ST *= np.exp((r - q - 0.5*sigma*sigma)*dt +
        sigma*epsilon*sqrt_dt)

    payoff[i] = option_payoff(ST, K, option_type) - ST + risk_neutral_ST


option_variance = np.var(payoff*np.exp(-r*T))
option_price = np.mean(payoff)*np.exp(-r*T)
print(option_type + ' price =', round(option_price, 8))
print(option_type + ' variance =', round(option_variance, 8))


# Calculate d1 and d2
d1 = (1 / (sigma * sqrt(T))) * (log(S0 / K) + (r - q + 0.5 * sigma**2) * T)
d2 = d1 - sigma * sqrt(T)

# Calculate N(d1) and N(d2) using cumulative distribution function
N_d1 = norm.cdf(d1)
N_d2 = norm.cdf(d2)

# Calculate the call option price using the Black-Scholes-Merton formula
C = S0 * exp(-q * T) * N_d1 - K * exp(-r * T) * N_d2

# Print the calculated call option price
print(f"The price of the European call option is approximately: {C:.2f}")
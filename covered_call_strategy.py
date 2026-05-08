import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#adapted from Octave's financial toolkit
def blsprice(Price, Strike, Rate, Time, Volatility):
    sigma_sqrtT = Volatility * np.sqrt(Time)

    d1 = 1 / sigma_sqrtT * (np.log(Price / Strike) + (Rate + Volatility**2 / 2) * Time)
    d2 = d1 - sigma_sqrtT
    phi1 = norm.cdf(d1)
    phi2 = norm.cdf(d2)

    disc = np.exp (-Rate * Time)
    F    = Price * np.exp ((Rate) * Time)

    Call = disc * (F * phi1 - Strike * phi2)
    Put  = disc * (Strike * (1 - phi2) + F * (phi1 - 1))

    return Call, Put

# Parameters
Volatility = 0.35
r = 0.07 
T = 1.5
strikes = [101, 115, 130]
S0 = 100  
Sinit = 100  
drift = 0.09 
num_simulations = 80000


# Initialize lists for results
mean_value = []
sd_value  = []
var_95_value = []
cvar_95_value = []

# Simulate for each strike price
for K in strikes:
    # Calculate the price of the call option
    call_price = blsprice(Price = S0, Strike = K, Rate = r, Time = T, Volatility = Volatility)[0]

    # Simulate final asset prices using GBM
    z = np.random.normal(0, 1, num_simulations)
    ST = S0 * np.exp((drift - Volatility**2 / 2) * T + Volatility * z * np.sqrt(T))

    # Calculate final portfolio values
    final_values = ST - np.maximum(ST - K, 0) + call_price * np.exp(r * T)
    log_returns = np.log(final_values / Sinit)

    # Calculate statistics
    mean_value.append(np.mean(log_returns))
    sd_value.append(np.std(log_returns))
    VaR_95 = np.percentile(log_returns, 5)
    var_95_value.append(VaR_95)
    cvar_95_value.append(np.mean(log_returns[log_returns <= VaR_95]))
    
    if K == 130:
        plt.hist(log_returns, bins=80, density=True, color='blue')
        plt.title(f"Probability Density Function for K = 130")
        plt.xlabel("Log Return")
        plt.ylabel("Density")
        plt.show()


# Print table
print(f"{'Strike(K)':<10}{'Mean Return':<15}{'Std Dev':<15}{'VaR (95%)':<15}{'CVaR (95%)':<15}")

for k, mean, sd, var, cvar in zip(strikes, mean_value, sd_value, var_95_value, cvar_95_value):
    print(f"{k:<10}{mean:<15.4f}{sd:<15.4f}{var:<15.4f}{cvar:<15.4f}")


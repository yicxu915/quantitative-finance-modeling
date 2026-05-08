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

#adapted from Octave's financial toolkit
def blsdelta(Price, Strike, Rate, Time, Volatility):
    d1 = 1 / (Volatility * np.sqrt(Time)) * (np.log (Price / Strike) + (Rate + Volatility**2 / 2) * Time)

    phi = norm.cdf(d1)
    CallDelta = phi
    PutDelta = phi - 1
    return CallDelta, PutDelta


# Parameters
sigma = 0.14
r = 0.08
T = 2.0
K = 90
S0 = 90
mu = 0.10
rebalance_intervals = [10, 100, 1000, 10000]


# Simulate Delta Hedging
results = []
for intervals in rebalance_intervals:
    
    # define initial parameters
    dt = T / intervals  
    S = [S0]  
    initial_V = blsprice(S0, K, r, T, sigma)[1]  
    initial_delta = blsdelta(S0, K, r, T, sigma)[1]  
    alpha = [initial_delta]
    B = [initial_V - alpha[0] * S0]  
    Pi = [-initial_V + alpha[0] * S0 + B[0]] 
    
    # start the path
    for i in range(1, intervals + 1):
        # update stock price using GBM
        Z = np.random.normal(0, 1)
        S_new = S[-1] * np.exp((mu - (sigma**2)/2) * dt + sigma * Z * np.sqrt(dt))
        S.append(S_new)

        time_to_maturity = T - i * dt
        
        # update alpha
        if time_to_maturity > 0:
            delta = blsdelta(S_new, K, r, time_to_maturity, sigma)[1]
            

        V = blsprice(S_new, K, r, time_to_maturity, sigma)[1]
        B_new = B[-1] * np.exp(r * dt) - (delta - alpha[-1]) * S_new
        Pi.append(-V + delta * S_new + B_new)
        alpha.append(delta)
        B.append(B_new)
        
    # calculate relative error and record result
    relative_error = abs(np.exp(-r * T) * Pi[-1] / abs(initial_V))
    results.append((intervals, relative_error))

    time = np.arange(0, T+dt,dt)
    
    # Plot 
    plt.figure()
    plt.plot(time, S, label="Stock Price (S)")
    plt.plot(time,[a * s for a, s in zip(alpha, S)], label="Stock Holding (α*S)")
    plt.plot(time,B, label="Risk-Free Account (B)")
    plt.plot(time,Pi, label="Portfolio Value (Π)")
    plt.title(f"Rebalances = {intervals})")
    plt.xlabel("Time")
    plt.legend()
    plt.grid()
    plt.show()

# Plot relative hedge error vs. rebalance intervals
intervals, errors = zip(*results)
plt.figure()
plt.plot(intervals, np.abs(errors), marker="o")
plt.xscale("log")
plt.title("Relative Hedge Error vs. Number of Rebalances")
plt.xlabel("Number of Rebalances")
plt.ylabel("Absolute Relative Hedge Error")
plt.grid()
plt.show()

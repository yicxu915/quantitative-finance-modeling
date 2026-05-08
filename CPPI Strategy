import numpy as np
import matplotlib.pyplot as plt

# define parameters
T = 2.0 
volatility = 0.32
drift = 0.12  
P0 = 100  
r = 0.06  
dt = 1 / 250  
initial_cash = 100
N = int(T / dt)  
S0 = 100  
initial_risky = 0  
num_simulations = 80000
FM = [(0, 1), (0, 0.5), (0, 2), (85, 2), (85, 4)]

# create vectors to store values
F_val =[]
M_val =[]
mean_val =[]
sd_val = []
var_val =[]
cvar_val =[]

# loop over each pair of F, M
for mult in FM:
    F=mult[0]
    M=mult[1]
    
    # Initialize first portfolio
    S = S0 * np.ones(num_simulations)
    alpha = M * ((np.maximum(0, initial_cash * np.exp(r * dt) + initial_risky * S - F)) / S)
    B0 = initial_cash * np.exp(r * dt) - (alpha - initial_risky) * S0
    B = np.full(num_simulations, B0)
    P = S * alpha + B 
    
    # Simulate paths
    for t in range(N):
        # Update risky asset prices (Geometric Brownian Motion)
        z = np.random.normal(0, 1, num_simulations)
        S *= np.exp((drift - (volatility**2)/2) * dt + volatility * z * np.sqrt(dt))  
        alpha_new = M * ((np.maximum(0, B * np.exp(r * dt) + alpha * S - F)) / S)
        B = B * np.exp(r * dt) - (alpha_new - alpha) * S
        P = alpha_new * S + B
        alpha = alpha_new
    
    # Final portfolio values and returns
    final = P
    R = np.log(final / P0)  
    
    # Calculate VaR and CVaR
    VaR = np.percentile(R, 5)  
    CVaR = np.mean(R[R <= VaR])  
    
    F_val.append(F)
    M_val.append(M)
    mean_val.append(np.mean(R))
    sd_val.append(np.std(R))
    var_val.append(VaR)
    cvar_val.append(CVaR)
    
    # Plot histogram of returns
    plt.hist(R, bins=200, density=True, color="blue")
    plt.title(f"F={F}, M={M}")
    plt.xlabel("Log Returns (R)")
    plt.ylabel("Density")
    plt.grid()
    plt.show()
    
# Print table
print(f"{'Floor (F)':<10}{'Multiplier (M)':<15}{'Mean Return':<15}{'Std Dev':<15}{'VaR (95%)':<15}{'CVaR (95%)':<15}")

for F, M, mean, sd, var, cvar in zip(F_val, M_val, mean_val, sd_val, var_val, cvar_val):
    print(f"{F:<10}{M:<15}{mean:<15.4f}{sd:<15.4f}{var:<15.4f}{cvar:<15.4f}")


import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 108
K = 110
r = 0.09
T = 1.0
gamma = 20
epsilon = 1e-8
time_steps = [10/250, 2.5/250, 1/250]
M = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 96000, 128000, 192000, 256000]

def generate_monte_carlo_simulation(dt, M, epsilon, gamma, r, S0, T, K):
    """
    Prices a European put option using Monte Carlo simulation
    under state-dependent volatility dynamics.

    Parameters
    ----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    ...
    """
    S = np.full(M, S0)
    N = int(T/dt)
    for i in range(N):
        rv = np.random.normal(0, 1, M)
        volatility = gamma / S  # Compute volatility
        S = S+ S * (r * dt + volatility * rv * np.sqrt(dt))  
        S = np.maximum(S, epsilon)  # Prevent negative prices
    payoffs = np.maximum(K - S, 0)
    V = np.exp(-r * T) * np.mean(payoffs)
    return(V)


for dt in time_steps:
    option_prices = []
    for sim_num in M:
        est_V = generate_monte_carlo_simulation(dt=dt, M=sim_num, epsilon=epsilon, gamma=gamma, r=r, S0=S0, T=T, K=K)
        option_prices.append(est_V)
        
    plt.plot(M, option_prices, marker='o')
    plt.title(f"Option Value vs Number of Simulations (Δt = {dt:.4f})")
    plt.xlabel("Number of Simulations (M)")
    plt.ylabel("Option Price")
    plt.grid()
    plt.show()


for sim_num in M:      
    S = np.full(sim_num, S0)  
    dt = 1 / 250  
    N = int(T / dt)  

    for i in range(N):
        rv = np.random.normal(0, 1, sim_num)  
        volatility = gamma / S  
        S = S + S * (r * dt + volatility * rv * np.sqrt(dt))  
        S = np.maximum(S, epsilon)  

    payoffs = np.maximum(K - S, 0)
    V = np.exp(-r * T) * payoffs  
    mean = np.mean(V)  
    std = np.std(V)  
    lower = mean - 1.96 * std / np.sqrt(sim_num)  
    upper = mean + 1.96 * std / np.sqrt(sim_num)  

    print(f"M: {sim_num:<8} Option Value: {mean:.4f} 95% CI: [{lower:.4f}, {upper:.4f}]")

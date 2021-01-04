# -*- coding: utf-8 -*-

'''
The following code implements a Sequential Monte Carlo for a "local level" Dynamic Linear Model
The algorithm is from Petris et al. - Dynamic Linear Models with R
'''

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import random

'''
The Dynamic Linear Model is specified by the observation and state equation as follows:
y[t] = theta[t] + v, where v is distributed as a Normal(0,V)
theta[t] = theta[t-1] + w, where w is distributed as a Normal(0,W)

In addition, the prior on theta is a distributed as a Normal(m0,c0)

In the following implementation, the parameters of the model are considered known.
We then generate a process of dimension t with the specified parameters.
'''
m0, C0, V, W = 5, 3, 2, 1
t = 100
theta = np.zeros(t)
theta[0] = stats.norm.rvs(loc = m0, scale = C0)
y = np.zeros(t)
for t in range(1, t):
    theta[t] = stats.norm.rvs(loc = theta[t-1], scale = W)
    mt = theta[t]
    y[t] = stats.norm.rvs(loc = mt, scale = V, size = 1)


fig, ax = plt.subplots(figsize=(16,9)) # Plotting the generated process - latent state theta and observation y
ax.plot(y[1:])
ax.plot(theta[1:])

N = 1000 # N is the number of "particles", i.e. the dimension of the sample generated.
tol = N/2 # Tolerance level for the Effective Sample Size. 

sd_importance = np.sqrt(W - W**2/(V + W)) # Definition of the importance distribution standard deviation
sd_theta_y = np.sqrt(V + W) 


'''
In the following, the algorithm is implemented.
Firstly, the arrays used in the algorithm are initialized.
'''

w_t = np.zeros(shape = (t + 1, N))
thetas_sim = np.zeros(shape = (t + 1, N))
pi_hat_sample = np.zeros(shape = (t+1,N))
ESS = np.zeros(t+1)
theta_res = np.zeros(shape = (t+1, N)) # auxiliary array used for the resampling step
thetas_est = np.zeros(t) # Monte Carlo approximations of filtering mean of theta_t|y_1:t
filt_est = np.zeros(shape = (t + 1, N)) # approximate sample from theta_t|y_1:t at each t


thetas_sim[1] = stats.norm.rvs(loc = m0, scale = C0) # initialization from the prior

w_t[1] = np.repeat(1/N,N) # initialization with equal weights

filt_est[1] = np.random.choice(thetas_sim[1], N, p=w_t[1]) 


for i in range(2,t+1):
    
    # Drawing theta_i's from the importance distribution
    y_theta = (y[i-1] - thetas_sim[i-1])
    var_sum = W + V
    mean_importance = thetas_sim[i-1] + W * y_theta/var_sum
    thetas_sim[i] = stats.norm.rvs(loc = mean_importance, scale = sd_importance**2)

    # Updating the weights w_t
    pi_g = w_t[i-1] * stats.norm.pdf(y[i-1], loc = thetas_sim[i-1], scale = sd_theta_y**2)
    w_t[i] = pi_g / np.sum(pi_g)
    
    # Evaluating ESS
    ESS[i] = (np.sum(w_t[i]**2))**(-1)

    # Multinomial resampling
    if ESS[i] < tol:
        index = np.random.choice(range(N), N , p= w_t[i])

        for c in range(N):
            theta_res[:,c] = thetas_sim[:,index[c]]
        thetas_sim = theta_res

        w_t[i] = np.repeat(1/N, N)

    # Drawing a sample from the approximate filtering distribution in t:
    filt_est[i] = np.random.choice(thetas_sim[i], N, p=w_t[i])

    # Monte Carlo approximations of filtering mean at t
    thetas_est[i-1] = np.dot(thetas_sim[i],w_t[i]) / np.sum(w_t[i])


# Graph of ESS, which indicates the points at which the multinomial resampling has been implemented:
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(ESS[1:], label = "Effective Sample Size")
ax.legend();


'''
In the following code, some plots are drawn in order to qualitatively assess the performance of the algorithm compared to the (exact) Kalman filter.
The first plot shows the observation y and the filtering estimates of the algorithm.
Then, some draws from the approximate filtering distribution at various t are plotted.
After computing the Kalman filter estimates, the second plot shows a comparison between Kalman filter and the algorithm implemented.
'''

# Observation and filtering
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(y[1:], label = "Observations")
ax.plot(thetas_est[1:], label = "Estimated thetas")
ax.legend();


# Graph of approximate filtering distributions
fig, ax = plt.subplots(figsize=(12,8), nrows = 3, ncols = 2)
c = [10,40,90]
for i,j in enumerate([2,30,80]):
    k = 0
    ax[i][k].hist(filt_est[j], alpha=0.5, bins=100, density=True, stacked=True, label = f"Filtering at t={j}")
    ax[i][k].legend();
    k += 1
    ax[i][k].hist(filt_est[c[i]], alpha=0.5, bins=100, density=True, stacked=True,label = f"Filtering at t={c[i]}")
    ax[i][k].legend();


# Closed-form solutions for Kalman filter
r = np.zeros(t)
q = np.zeros(t)
m = np.zeros(t)
f = np.zeros(t)
c = np.zeros(t)
a = np.zeros(t)
m[0] = m0
c[0] = C0
r[0] = c[0] + W
for t in range(1,t):
    a[t] = m[t-1]
    r[t] = c[t-1] + W

    f[t] = a[t]
    q[t] = r[t] + V

    m[t] = a[t] + r[t]*(y[t]-f[t])/q[t]
    c[t] = r[t] - (r[t]**2) / q[t]
theta_kalman = m

# Comparison between Kalman filter and Sequential MC
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(theta_kalman[1:], label = "Kalman filter")
ax.plot(thetas_est[1:], label = "Sequential MC")
ax.legend();

# %% 
import pandas as pd
import numpy as np
import random
import scipy.stats
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# %%
k = 3 # clusters
# d = 2 # dimensions
prob = 1/k # uniform prior on cluster assignments
n = 150 # data points
iterations = 1000
lambdas = 1
sigma = 0.5 # note: sigma^2


# %% Generative process (synthetic data)
d = 2
z_true = np.zeros(shape = (n))
z_true = np.random.choice(range(k), p = np.repeat(prob, k), size = n)

mu_true = np.zeros(shape = (k,d))

mu_true[0] = np.array((0,0))
mu_true[1] = np.array((1,2))
mu_true[2] = np.array((3,-4))

x = np.zeros(shape = (n,d))

for i in range(n):
    cluster_assignment = int(z_true[i])
    x[i] = scipy.stats.multivariate_normal.rvs(mean = mu_true[cluster_assignment], cov = sigma * np.identity(d))



# %% Matrix declaration
z = np.zeros(shape = (n, iterations))
mu = np.zeros(shape = (k, d, iterations))

n_k = np.zeros(shape = (k, iterations))
k_bar = np.zeros(shape = (k,d, iterations))


# %% Graphical representation of synthetic data:
from pylab import *
import matplotlib.pyplot as plt

x1 = x[:,0]
x2 = x[:,1]

fig = plt.figure(figsize = (16,9))
ax = fig.add_subplot(111)

scatter(x1, x2, s=100 ,marker='o')

[ plot( [dot_x,dot_x] ,[0,dot_y], '.', linewidth = 3 ) for dot_x,dot_y in zip(x1,x2) ] 
[ plot( [0,dot_x] ,[dot_y,dot_y], '.', linewidth = 3 ) for dot_x,dot_y in zip(x1,x2) ]

left,right = ax.get_xlim()
low,high = ax.get_ylim()
arrow( left, 0, right -left, 0, length_includes_head = True, head_width = 0.08 )
arrow( 0, low, 0, high-low, length_includes_head = True, head_width = 0.08 ) 

show()

# %% Initialization:
mu[:,:,0] = np.zeros(shape = (k,d)) # mixture locations
z[:,0] = np.random.choice(a = np.array((range(k))), p = np.repeat(prob, k), size = n)

# %% Gibbs sampler:
for it in range(iterations-1):
    
    for i in range(n):
        #cluster_assignment = int(z[i, it])
        prob_temp = np.zeros(shape = (3,1))
        for kk in range(k):
            prob_temp[kk] = prob * scipy.stats.multivariate_normal.pdf(x = x[i], mean = mu[kk,:,it], cov = sigma * np.identity(d))
        # print(f"Probabilities: {prob_temp}")
        prob_temp = prob_temp / prob_temp.sum()
        # print(f"Probabilities, normalized: {prob_temp}")
        z[i,it+1] = np.random.choice(a = np.array((0,1,2)), p = prob_temp[:,0])
    
    for kk in range(k):

        n_k[kk,it] = np.unique(z[:,it], return_counts=True)[1][list(np.unique(z[:,0], return_counts=True)[0]).index(kk)]
        for i in range(n):
            if z[i,it] == kk:
                k_bar[kk,:,it] += x[i]
        k_bar[kk,:,it] = k_bar[kk,:,it] / n_k[kk,it]

        mean_k = (n_k[kk,it] / sigma) / ((n_k[kk,it]/sigma) + 1/lambdas) * k_bar[kk,:,it]
        cov_k = 1 / (n_k[kk,it]/sigma + 1/lambdas)

        mu[kk,:,it+1] = scipy.stats.multivariate_normal.rvs(mean = mean_k, cov = cov_k)

    print(f"Iteration {it} complete.")

# %% Burn-in and trimming of the relevant chains
burn_in = int(iterations/2)

z_trimmed = z[:,burn_in:]
mu_trimmed = mu[:,:,burn_in:]

# %% Bayesian estimates: mean of the approximate posteriors
z_est = z_trimmed.mean(axis = 1)
mu_est = mu_trimmed.mean(axis = 2)

# Sampling from the approximate posterior
x_sampl_1 = scipy.stats.multivariate_normal.rvs(mean = mu_est[0,:], cov = sigma * np.identity(2), size = 50)
x_sampl_2 = scipy.stats.multivariate_normal.rvs(mean = mu_est[1,:], cov = sigma * np.identity(2), size = 50)
x_sampl_3 = scipy.stats.multivariate_normal.rvs(mean = mu_est[2,:], cov = sigma * np.identity(2), size = 50)

# %% Graphical representation:
from pylab import *
import matplotlib.pyplot as plt

x11 = x_sampl_1[:,0]
x12 = x_sampl_2[:,0]
x13 = x_sampl_3[:,0]

x21 = x_sampl_1[:,1]
x22 = x_sampl_2[:,1]
x23 = x_sampl_3[:,1]

fig = plt.figure(figsize = (16,9))
ax = fig.add_subplot(111)

scatter(x11, x21, s=100 ,marker='o')
scatter(x12, x22, s=100 ,marker='o')
scatter(x13, x23, s=100 ,marker='o')

[ plot( [dot_x,dot_x] ,[0,dot_y], '.', linewidth = 3 ) for dot_x,dot_y in zip(x11,x21) ] 
[ plot( [0,dot_x] ,[dot_y,dot_y], '.', linewidth = 3 ) for dot_x,dot_y in zip(x11,x21) ]

[ plot( [dot_x,dot_x] ,[0,dot_y], '.', linewidth = 3 ) for dot_x,dot_y in zip(x12,x22) ] 
[ plot( [0,dot_x] ,[dot_y,dot_y], '.', linewidth = 3 ) for dot_x,dot_y in zip(x12,x22) ]

[ plot( [dot_x,dot_x] ,[0,dot_y], '.', linewidth = 3 ) for dot_x,dot_y in zip(x13,x23) ] 
[ plot( [0,dot_x] ,[dot_y,dot_y], '.', linewidth = 3 ) for dot_x,dot_y in zip(x13,x23) ]

left,right = ax.get_xlim()
low,high = ax.get_ylim()
arrow( left, 0, right -left, 0, length_includes_head = True, head_width = 0.08 )
arrow( 0, low, 0, high-low, length_includes_head = True, head_width = 0.08 ) 

show()

# %% Trace plots of the means of three clusters
fig, ax = plt.subplots(figsize = (16,9), nrows = k, ncols = d)
for ax2 in range(ax.shape[1]):
    ax[ax1,ax2].plot(mu[0,ax2,:])

# %% Effective Sample Size
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
def eff_sample_size(chain):
    '''
    Compute the Effective Sample Size for the MCMC provided
    '''
    dimen = len(chain)
    somma = np.sum(acf(chain))
    den = 1 + 2*somma
    ess = dimen/den

    return ess

# %% Showing ESS results for cluster means
for i in range(k):
    print(f"ESS (means cluster{i}): ",eff_sample_size(mu[i,0,:]), eff_sample_size(mu[i,0,:]))

# %% Kolmogorov-Smirnov nonparametric test
from scipy import stats
test_results = np.zeros(50)
effective_ss = []

mu_thinned = mu[:,:,::2] # thinning


for i in range(k):
    for j in range(d):    
        burnin = mu_thinned[i,j,:int(len(mu_thinned[i,j,:])/3)]
        sample1 = mu_thinned[i,j,int(len(mu_thinned[i,j,:])/3):2*int(len(mu_thinned[i,j,:])/3)]
        sample2 = mu_thinned[i,j,2*int(len(mu_thinned[i,j,:])/3):]
    print(f"Test completed for cluster {i}")
    print(stats.ks_2samp(sample1, sample2))


z_thinned = z[:,::2]
for i in range(n):
    burnin = z_thinned[i,:int(len(z_thinned)/3)]
    sample1 = z_thinned[i,int(len(z_thinned)/3):2*int(len(z_thinned)/3)]
    sample2 = z_thinned[i,2*int(len(z_thinned)/3):]   
print(f"Test completed for data point {i}")
print(stats.ks_2samp(sample1, sample2))

# %% --------- Application 2: image segmentation ----------------------------
#%% 
import pandas as pd
import numpy as np
import random
import scipy.stats
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import math


# %%
it = 0
k = 8 # clusters
# d = 2 # dimensions
prob = 1/k # uniform prior on cluster assignments
# n = 150 # data points
iterations = 200
lambdas = 0.003
sigma = 0.003 # note: sigma^2


# %% Image data
from matplotlib.image import imread
# img = imread('/Users/leonardo/Downloads/xp.jpg')
img = imread('home.jpg')


# img = imread('home.png')
img_size = img.shape
x = img.reshape(img_size[0] * img_size[1], img_size[2])
x = x.astype(float)

d = x.shape[1]
n = x.shape[0]

# Normalizing color intensities between 0 and 1
for dim in range(x.shape[1]):
    for i in range(x.shape[0]):
        x[i, dim] /= 255


#%% Matrix declaration
z = np.zeros(shape = (n, iterations))
mu = np.zeros(shape = (k, d, iterations))

n_k = np.zeros(shape = (k, iterations))
k_bar = np.zeros(shape = (k,d, iterations))


# %% Gibbs sampler:
# mu[:,:,0] = x.mean(axis = 0) # mixture locations
mu[:,:,0] = np.zeros(shape = (k,d))  # mixture locations
z[:,0] = np.random.choice(a = np.array((range(k))), p = np.repeat(prob, k), size = n)

for it in range(iterations-1):
    
    for i in range(n):
        # cluster_assignment = int(z[i, it])
        prob_temp = np.zeros(shape = (k,1))
        for kk in range(k):
            prob_temp[kk] = prob * scipy.stats.multivariate_normal.pdf(x = x[i], mean = mu[kk,:,it], cov = sigma * np.identity(d))
        # print(f"Probabilities: {prob_temp}")
        prob_temp /= prob_temp.sum()
        # print(f"Probabilities, normalized: {prob_temp}")
        z[i,it+1] = np.random.choice(a = np.array(range(k)), p = prob_temp[:,0])
    
    for kk in range(k):
        if kk in list(np.unique(z[:,it], return_counts=True)[0]):
            n_k[kk,it] = np.unique(z[:,it], return_counts=True)[1][list(np.unique(z[:,it], return_counts=True)[0]).index(kk)]
        for i in range(n):
            if z[i,it] == kk:
                k_bar[kk,:,it] += x[i]
        k_bar[kk,:,it] = k_bar[kk,:,it] / n_k[kk,it]

        mean_k = (n_k[kk,it] / sigma) / ((n_k[kk,it]/sigma) + 1/lambdas) * k_bar[kk,:,it]
        cov_k = 1 / (n_k[kk,it]/sigma + 1/lambdas)

        mu[kk,:,it+1] = scipy.stats.multivariate_normal.rvs(mean = mean_k, cov = cov_k)

    print(f"Iteration {it} complete.")

# %% Burn-in and trimming
burn_in = int(iterations/3)
mu_trimmed = mu[:,:,burn_in:]
mu_est = mu_trimmed[:,:,:].mean(axis=2)

x_compr = np.zeros(shape = (1024,d))
for i in range(x.shape[0]):
    x_compr[i] = mu[int(z[i,it-1]),:,it] * 255
x_compr = x_compr.astype(int)

x_reshaped = x.reshape(img_size[0], img_size[1], img_size[2])
x_compr = x_compr.reshape(img_size[0], img_size[1], img_size[2])

# %% Drawing the final results
fig, ax = plt.subplots(1, 2, figsize = (12, 8))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[1].imshow(x_compr)
ax[1].set_title(f'Segmented Image with {k} colors')
for ax in fig.axes:
    ax.axis('off')
plt.tight_layout();
    
# %% Plot of the means of three clusters
fig, ax = plt.subplots(figsize = (16,9), nrows = k, ncols = d)
for ax1 in range(ax.shape[0]):
    for ax2 in range(ax.shape[1]):
        ax[ax1,ax2].plot(mu[ax1,ax2,:])
        ax[ax1,ax2].set_ylim((-0.2,1.2))

# %% Original pixels, cluster assignment and assigned cluster means
show_its = it-1
for i in range(int(x.shape[0]/20)):
    print(x[i], z[i,show_its], mu_est[int(z[i,show_its]),:])
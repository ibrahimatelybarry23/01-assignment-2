import numpy as np
import matplotlib.pyplot as plt
import scipy
from utils import p, A, X, Y, quasi_newton_nd
plt.rcParams.update({'font.size': 18})


#%% Define f, df0
def f(x, y, A, beta):
    val = -2*(A.T @(y-p(x,A,beta)))/(1+beta*x)**2
    return val

df0 = 2*A.T@A


#%% Set a value beta
beta = 1e-1


#%% Generate the vectors of emissions and measures
bar_x = np.array([0, 0, 1, 0, 0, 0, 0])
bar_y = p(bar_x, A, beta)


#%% Plot the sensors/sources
fig = plt.figure(0)
fig.clf()
ax = fig.gca()
for idx, _ in enumerate(X):
    markersize = np.abs(bar_x[idx]) / np.linalg.norm(bar_x) *  50
    ax.plot(X[idx, 0], X[idx, 1], 'o', 
            alpha=0.2, markersize=markersize, color='#1f77b4')

for idx, _ in enumerate(Y):
    markersize = np.abs(bar_y[idx]) / np.linalg.norm(bar_y) *  50
    ax.plot(Y[idx, 0], Y[idx, 1], 'o', 
            alpha=0.2, markersize=markersize, color='#ff7f0e')
    
a=ax.plot(X[:, 0], X[:, 1], '*', label='Sources')
b=ax.plot(Y[:, 0], Y[:, 1], 'o', label='Sensors')
ax.legend()
ax.grid(True)

    
#%% Run the quasi Newton method to obtain the estimation
k_max = 100 
eps_res = 1e-10
eps_inc = 1e-10
x0 = np.zeros(len(bar_x))
def f_handle(x):
    return f(x, bar_y, A, beta)
x_iter, fx_iter = quasi_newton_nd(f_handle, df0, x0, k_max, eps_res, eps_inc)

hat_x = x_iter[-1]


#%% Visualize the exact and estimated values
xx = np.arange(len(bar_x))
width = 0.3
offset = 0.3

fig = plt.figure(1)
fig.clf()
ax = fig.gca()
ax.bar(xx, bar_x, width=width, label='Exact')
ax.bar(xx + offset, hat_x, width=width, label='Estimated')
ax.grid(True)
ax.set_xticks(xx + width, ['Source %d' %(i+1) for i in xx], rotation='vertical')
ax.legend()
fig.tight_layout()




# Task 3 (c) error

betas = np.arange(0, 0.11, 0.01)
errors = np.zeros(len(betas))

for i, b in enumerate(betas):
    bar_y_b = p(bar_x, A, b)                  
    f_b = lambda x: f(x, bar_y_b, A, b)       
    x_iter_b, _ = quasi_newton_nd(f_b, df0, x0, k_max, eps_res, eps_inc)
    hat_x_b = x_iter_b[-1]
    errors[i] = np.linalg.norm(bar_x - hat_x_b)

fig = plt.figure(2)
fig.clf()
ax = fig.gca()
ax.semilogy(betas, errors, 'o-')                  
ax.grid(True)

fig.tight_layout()


plt.show()



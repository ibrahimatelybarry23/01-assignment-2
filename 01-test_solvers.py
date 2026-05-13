#%% Imports
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12,
                    'lines.linewidth': 2})
from utils import bisection, newton, simplified_newton
    

#%% Define the function and its derivative
f = lambda x: x ** 3 - x - 1


def df(x):
    return 3*x**2-1


#%% Define the stopping values
k_max = 100
eps_res = 1e-10
eps_inc = 1e-10



#%% Bisection method
# Define the initial interval
a = 1
b = 2

# Compute the solution
x_bis, fx_bis = bisection(f, a, b, k_max=k_max, eps_res=eps_res, eps_inc=eps_inc)


#%% Newton method
# Define the initial guess
x0 = 0.58

# Compute the solution
x_new, fx_new = newton(f, df, x0, k_max, eps_res, eps_inc)


#%% Simplified Newton method
# Define the initial guess
x0 = 1.06

# Compute the solution
df_x0 = df(x0)
x_sim_new, fx_sim_new = simplified_newton(f, df_x0, x0, k_max, eps_res, eps_inc)








#%% Plot f on [-3, 3] to locate the real roots
x_ = np.linspace(-3, 3, 1000)
fig = plt.figure(0)
fig.clf()
ax = fig.gca()
ax.plot(x_, f(x_))
ax.plot(x_, 0 * x_, 'k')
ax.set_title(r'$f(x) = x^3 - x - 1$ on $[-3, 3]$')
ax.grid(True)








#%% Visualize the results
# Evaluate the function f to visualize it
x = np.linspace(0, 2, 1000)
fx = f(x)

# Plot the iterations
fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot(131)
ax.plot(x, fx)
ax.plot(x_bis, f(x_bis), '.-')   
ax.plot(x, 0 * x, 'k') 
ax.set_title('Bisection')
ax.grid(True)

ax = fig.add_subplot(132)
ax.plot(x, fx)
ax.plot(x_new, f(x_new), '.-')    
ax.plot(x, 0 * x, 'k') 
ax.set_title('Newton')
ax.grid(True)

ax = fig.add_subplot(133)
ax.plot(x, fx)
ax.plot(x_sim_new, f(x_sim_new), '.-')    
ax.plot(x, 0 * x, 'k') 
ax.set_title('Simplified Newton')

ax.grid(True)

# Plot the residuals
fig = plt.figure(2)
fig.clf()
ax = fig.gca()
ax.semilogy(np.abs(fx_bis), '.-')
ax.semilogy(np.abs(fx_new), '.-')
ax.semilogy(np.abs(fx_sim_new), '.-')
ax.legend(['Bisection', 'Newton', 'Simplified Newton'])
ax.grid(True)

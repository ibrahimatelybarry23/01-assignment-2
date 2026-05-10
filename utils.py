#%% Imports
import numpy as np


#%% Task 1 / Task 2
def bisection(f, a, b, k_max, eps_res, eps_inc):
    k = 0
    delta_x = np.inf
    F_k = np.inf

    x_l = []
    fx_l = []
    while k<k_max and abs(F_k)> eps_res and delta_x > eps_inc:
      x_k = (a+b)/2
      f_k = f(x_k)
      if f_k * f(a)<0:
        b= x_k
      else:
        a = x_k
      F_k = f_k
      delta_x = abs(b-a)
      k+=1
      x_l.append(x_k)
      fx_l.append(abs(F_k))
    x=np.array(x_l)
    f_x=np.array(fx_l)
    return x,f_x


def newton(f, df, x0, k_max, eps_res, eps_inc):
   
    x_k = x0
    k=0
    delta_x = np.inf
    F_k = f(x_k)

    x_l = [x_k]
    fx_l = [abs(F_k)]

    while k<k_max and abs(F_k)>eps_res and delta_x > eps_inc:
      x_new = x_k-(f(x_k)/df(x_k))
      delta_x = abs(x_new-x_k)
      x_k =x_new
      F_k = f(x_k)
      k+=1
      x_l.append(x_k)
      fx_l.append(abs(F_k))
    x= np.array(x_l)
    f_x= np.array(fx_l)
    return x,f_x




def simplified_newton(f, df_x0, x0, k_max, eps_res, eps_inc):
    x_k = x0
    k=0
    delta_x = np.inf
    F_k = f(x_k)

    x_l = [x_k]
    fx_l = [abs(F_k)]

    while k<k_max and abs(F_k)>eps_res and delta_x > eps_inc:
      x_new = x_k-f(x_k)/df_x0
      delta_x = abs(x_new-x_k)
      x_k =x_new
      F_k = f(x_k)
      k+=1
      x_l.append(x_k)
      fx_l.append(abs(F_k))
    x= np.array(x_l)
    f_x= np.array(fx_l)
    return x,f_x


#%% Task 3
X = np.array([[0.0884925 , 0.19598286],
       [0.04522729, 0.32533033],
       [0.38867729, 0.27134903],
       [0.82873751, 0.35675333],
       [0.28093451, 0.54269608],
       [0.14092422, 0.80219698],
       [0.07455064, 0.98688694]])


Y = np.array([[0.30461377, 0.09767211],
       [0.68423303, 0.44015249],
       [0.12203823, 0.49517691],
       [0.03438852, 0.9093204 ],
       [0.25877998, 0.66252228],
       [0.31171108, 0.52006802],
       [0.54671028, 0.18485446],
       [0.96958463, 0.77513282],
       [0.93949894, 0.89482735],
       [0.59789998, 0.92187424]])


A = np.array([[ 4.21175272,  2.89752105,  5.18264494,  1.71039215,  2.24389577,
         1.38256981,  1.08873821],
       [ 1.55318909,  1.54026267,  2.93802915,  5.99361781,  2.4030903 ,
         1.53166   ,  1.22111941],
       [ 3.32150045,  5.36458902,  2.87248059,  1.38864119,  6.02955842,
         3.25097109,  2.02430053],
       [ 1.397846  ,  1.71206303,  1.37034139,  1.03344577,  2.26340342,
         6.6189928 , 11.44854137],
       [ 2.0135082 ,  2.50546186,  2.42614256,  1.54608037,  8.20633658,
         5.47183845,  2.68073513],
       [ 2.54116911,  3.02979704,  3.84090364,  1.84431535, 26.178134  ,
         3.03218448,  1.90982695],
       [ 2.18172508,  1.92017237,  5.55078312,  3.02768267,  2.24344201,
         1.35360993,  1.07446732],
       [ 0.94841578,  0.97277426,  1.30051059,  2.26525516,  1.37585841,
         1.20612386,  1.08726124],
       [ 0.90811743,  0.94320794,  1.20200495,  1.82031404,  1.33905464,
         1.24389082,  1.14964515],
       [ 1.12764996,  1.22969353,  1.46339469,  1.63813931,  2.02343276,
         2.11690835,  1.89619488]])


def p(x, A, beta):
    y = A @ (x / (1 + beta * x))
    return y


def quasi_newton_nd(f, df0, x0, k_max, eps_res, eps_inc):
    x_k = x0.copy()
    k = 0 
    delta_x = np.inf 
    F_k = f(x_k)
    x_l = [x_k.copy()]
    fx_l = [np.linalg.norm(F_k)]
    while k<k_max and np.linalg.norm(F_k)>eps_res and delta_x > eps_inc:
        delta = np.linalg.solve(df0,f(x_k))
        x_new = x_k-delta
        delta_x = np.linalg.norm(x_new-x_k)
        x_k = x_new
        F_k = f(x_k)
        k+=1
        x_l.append(x_k.copy())
        fx_l.append(np.linalg.norm(F_k))
    x=np.array(x_l)
    f_x=np.array(fx_l) 
    return x, f_x
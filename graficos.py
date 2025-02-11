#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 10:55:31 2025

@author: juan
"""

import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.signal import find_peaks, hilbert
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap
import imageio
from scipy import signal
plt.rcParams["font.family"] = "serif"

#%%
def tang(r,A,c,b):
  return (A/r)*(1-np.exp(-((r**2)/c)))+b

def tang2(r,A,c,b):
  return (A)*(1-np.exp(-((r**2)/c)))+b

def rad(r,A,b):
  return -2*A*r + b

nan = 0

#%%
def plot_t(f,x,y,yerr,p0, df):
    
    param, cov = curve_fit(f, x, y, p0=p0, sigma=yerr, maxfev=10000)
    y_pred = f(x, *param)
    residuos = y - y_pred
    x_ajuste = np.linspace(min(x), max(x), 1000)
    y_pred_graf = tang(x_ajuste, *param)#df grande = 11/371
    
    
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(4, 1)  # Cambiar a 4 filas

    ax1 = fig.add_subplot(gs[0:3, 0])  # Ocupa filas 0, 1 y 2
    ax1.errorbar(x, y, yerr=yerr, fmt='.', capsize=5, label='Datos')
    #ax1.plot(x_fit2, y_fit2, 'r-', label=f'Ajuste: $h1 = {a_opt2:.3f}* m  {b_opt2:.3f}$')
    ax1.plot(x_ajuste, y_pred_graf, 'r-', label=f'Ajuste')
    #ax1.fill_between(x_fit2, lineal(x_fit2, a_opt2 - a_err2, b_opt2 - b_err2),
                     #lineal(x_fit2, a_opt2 + a_err2, b_opt2 + b_err2), color='r', alpha=0.2, label='Incertidumbre ajuste')

    ax1.set_ylabel(r'Velocidad tangencial $v_{\theta}/v_{\theta}^{max}$', fontsize = 14)
    ax1.tick_params(direction='in')
    ax1.tick_params(axis='y',labelsize = 12, length=4)
    ax1.tick_params(axis='x', length=4)
    ax1.legend(fontsize = 12)
    ax1.set_xticklabels([''] , fontsize=9)
    #ax1.set_title('Ajuste Lineal con Datos de Error', fontsize = 16)
    ax1.grid(True, linestyle='--', alpha = 0.5)

    ax2 = fig.add_subplot(gs[3, 0])  # Ocupa solo la fila 3
    ax2.errorbar(x, residuos, yerr=yerr, fmt='.', capsize=5, label='Residuos con barras de error')
    ax2.axhline(0, color='r', linestyle='--', label='y = 0')
    ax2.set_xlabel('Radio (cm)', fontsize = 14)
    ax2.set_ylabel('Residuos', fontsize = 14)
    #ax2.legend()
    ax2.tick_params(axis='y', length=4,labelsize = 10)
    ax2.tick_params(axis='x', length=4, labelsize = 12)

    ax2.tick_params(direction='in')
    ax2.grid(True, linestyle='--', alpha = 0.5)
    #ax2.set_xticks(np.arange(0.25, 0.47, 0.03) )

    plt.subplots_adjust(hspace=0)
    plt.savefig('ajuste_70_30_tang.png',bbox_inches='tight')

    plt.show()
    
    param_errores = np.sqrt(np.diag(cov))
    
    chi2 = np.sum(((y - y_pred) / yerr) ** 2)  # Chi-cuadrado
    dof = len(y) - len(param)  # Grados de libertad (N - p)
    chi2_red = chi2 / dof  # Chi-cuadrado reducido
    print(f"Chi^2 reducido: {chi2_red}")
    
    print(param)
    print(param_errores)
    
def plot_r(f,x_f,y_f,yerr_f,p0):
    

    
    
    param, cov = curve_fit(f, x_f, y_f, p0=p0, sigma=yerr_f, maxfev=100000)
    y_pred = f(x_f, *param)
    residuos = y_f - y_pred
    x_ajuste = np.linspace(min(x_f), max(x_f), 1000)
    y_pred_graf = f(x_ajuste, *param)
    
    
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(4, 1)  # Cambiar a 4 filas

    ax1 = fig.add_subplot(gs[0:3, 0])  # Ocupa filas 0, 1 y 2
    ax1.errorbar(x_f, y_f, yerr=yerr_f, fmt='.', capsize=5, label='Datos')
    ax1.plot(x_ajuste, y_pred_graf, 'r-', label=f'Ajuste')

    ax1.set_ylabel(r'Velocidad radial $v_{r}/v_{r}^{max}$', fontsize = 14)
    ax1.tick_params(direction='in')
    ax1.tick_params(axis='y',labelsize = 12, length=4)
    ax1.tick_params(axis='x', length=4)
    ax1.legend(fontsize = 12)
    ax1.set_xticklabels([''] , fontsize=9)
    #ax1.set_title('Ajuste Lineal con Datos de Error', fontsize = 16)
    ax1.grid(True, linestyle='--', alpha = 0.5)

    ax2 = fig.add_subplot(gs[3, 0])  # Ocupa solo la fila 3
    ax2.errorbar(x_f, residuos, yerr=yerr_f, fmt='.', capsize=5, label='Residuos con barras de error')
    ax2.axhline(0, color='r', linestyle='--', label='y = 0')
    ax2.set_xlabel('Radio (cm)', fontsize = 14)
    ax2.set_ylabel('Residuos', fontsize = 14)
    #ax2.legend()
    ax2.tick_params(axis='y', length=4,labelsize = 10)
    ax2.tick_params(axis='x', length=4, labelsize = 12)

    ax2.tick_params(direction='in')
    ax2.grid(True, linestyle='--', alpha = 0.5)
    #ax2.set_xticks(np.arange(0.25, 0.47, 0.03) )

    plt.subplots_adjust(hspace=0)
    plt.savefig('ajuste_70_30_rad.png',bbox_inches='tight')

    plt.show()
    
    param_errores = np.sqrt(np.diag(cov))
    chi2 = np.sum(((y_f - y_pred) / yerr_f) ** 2)  # Chi-cuadrado
    dof = len(y_f) - len(param)  # Grados de libertad (N - p)
    chi2_red = chi2 / dof  # Chi-cuadrado reducido
    print(f"Chi^2 reducido: {chi2_red}")
    print(param)
    print(param_errores)
    
#%%
os.chdir('/home/juan/Documents/Laboratorio 5')
#%%

os.chdir('/home/juan/Documents/Laboratorio 5/videos 3')

data = np.loadtxt('70_30.csv', delimiter = ',', skiprows = 5)

df_22 = 0.014358974358974359

#df_22 = 0.009554140127388535

r = data[:,0]*df_22
vt = data[:,1]
errvt = data[:,2]

r2 = data[:,3]*df_22
vr = data[:,4]
errvr = data[:,5]

indices = r2 > 1.5

# Filtra los arrays
r2_f = r2[indices][:-1]
vr_f = vr[indices][:-1]
errvr_f = errvr[indices][:-1]

vr_f = vr_f/(np.max(vr_f))
errvr_f = errvr_f/(np.max(vr_f))/5

p0 = np.array([0.1,1,0.0])
p02 = np.array([1,1])


#%%
plot_r(rad, r2_f, vr_f, np.abs(errvr_f), p02)
#%%
plot_t(tang, r[:-1], vt[:-1], errvt[:-1]*3, p0, df_22)


#%%

p55 = [ 1.20053282,  0.68415521 ,-0.0643744 ]
p50 = [ 1.31424373,  1.01465773, -0.02285262]
p40 = [1.82906641, 2.0842216,  0.05569866]
p30 = [1.09423151, 1.43180184, 0.17971777]
p25 = [0.8381839,  1.00501189, 0.13604271]


plt.figure(figsize = (8,6))
plt.plot(r, tang(r,*p55), '-', label = r'$55\% V/V$')
plt.plot(r, tang(r,*p50), '-', label = r'$50\% V/V$')
plt.plot(r, tang(r,*p40), '-', label = r'$40\% V/V$')
plt.plot(r, tang(r,*p30), '-', label = r'$30\% V/V$')
plt.plot(r, tang(r,*p25), '-', label = r'$25\% V/V$')
plt.ylabel(r'Velocidad tangencial $\frac{v_{\theta}}{v_{\theta}^{max}}$', fontsize = 14)
plt.xlabel('Radio (cm)', fontsize = 14)
plt.grid(True, linestyle='--', alpha = 0.5)
plt.legend(fontsize = 14, prop={"family": "serif"})

plt.savefig('comparacion%.png',bbox_inches='tight')

#%%
plt.plot([25,30,40,50,55], [0.09, 0.13, 0.350,0.52  ,0.74], 'o')


#%%
c2 = np.array([0.34,0.49 , 0.51,0.54, 0.60 , 0.67, 0.59])
errc2 = np.array([0.01,0.01,0.01,0.01, 0.11, 0.03, 0.04])
alpha = np.array([0.018, 0.011, 0.018, 0.011, 0.017, 0.033, 0.040])
erralpha = np.array([0.004, 0.004, 0.003, 0.003, 0.004, 0.004, 0.004])

nu = c2*alpha
errnu = np.sqrt((alpha * errc2)**2 + (c2 * erralpha)**2)

Temp = np.array([30, 44, 52, 60, 70, 76, 82])

plt.errorbar(Temp, nu, yerr = nu/3, fmt = 'o', capsize=5)

#%%
def exponencial(x,a,b, c):
    return a*np.exp(b/x)+c

#%%
p0_exp =  [0.01, 0.001, 0.001]
nu = np.array([0.00612, 0.00539, 0.00918, 0.00594, 0.0102 , 0.02211, 0.0236 ])
errnu = np.array([0.00137186, 0.00196308, 0.00154055, 0.00162373, 0.00304252,
       0.00285701, 0.00285125])
#%%
param_nu, cov = curve_fit(exponencial, Temp, nu, p0=p0_exp, sigma=errnu, maxfev=10000)
nu_pred = exponencial(Temp, *param_nu)
residuos_nu = nu - nu_pred
Temp_ajuste = np.linspace(min(Temp), max(Temp), 100)
nu_pred_graf = exponencial(Temp_ajuste, *param_nu)

plt.errorbar(Temp, nu, yerr = errnu, fmt = 'o', capsize=5)
plt.plot(Temp_ajuste, nu_pred_graf, '-r')
print(param_nu)


#%%

fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(4, 1)  # Cambiar a 4 filas

ax1 = fig.add_subplot(gs[0:3, 0])  # Ocupa filas 0, 1 y 2
ax1.errorbar(Temp, nu, yerr=errnu, fmt='o', capsize=5, label=r'$\nu$ propagado')
ax1.plot(Temp_ajuste, nu_pred_graf, 'r-', label=f'Ajuste')

ax1.set_ylabel(r'Viscosidad $\nu$ (cm)', fontsize = 14)
ax1.tick_params(direction='in')
ax1.tick_params(axis='y',labelsize = 12, length=4)
ax1.tick_params(axis='x', length=4)
ax1.legend(fontsize = 12)
ax1.set_xticklabels([''] , fontsize=9)
#ax1.set_title('Ajuste Lineal con Datos de Error', fontsize = 16)
ax1.grid(True, linestyle='--', alpha = 0.5)

ax2 = fig.add_subplot(gs[3, 0])  # Ocupa solo la fila 3
ax2.errorbar(Temp, residuos_nu, yerr=errnu, fmt='.', capsize=5, label='Residuos con barras de error')
ax2.axhline(0, color='r', linestyle='--', label='y = 0')
ax2.set_xlabel(r'Temperatura (°C)', fontsize = 14)
ax2.set_ylabel('Residuos', fontsize = 14)
#ax2.legend()
ax2.tick_params(axis='y', length=4,labelsize = 10)
ax2.tick_params(axis='x', length=4, labelsize = 12)

ax2.tick_params(direction='in')
ax2.grid(True, linestyle='--', alpha = 0.5)
#ax2.set_xticks(np.arange(0.25, 0.47, 0.03) )

plt.subplots_adjust(hspace=0)
plt.savefig('nu_exp_temperaturas.png',bbox_inches='tight')

plt.show()

param_errores = np.sqrt(np.diag(cov))

chi2 = np.sum(((nu - nu_pred) / errnu) ** 2)  # Chi-cuadrado
dof = len(nu) - len(param_nu)  # Grados de libertad (N - p)
chi2_red = chi2 / dof  # Chi-cuadrado reducido
print(f"Chi^2 reducido: {chi2_red}")

print(param_nu)
print(param_errores)
plt.savefig('nu_exp_temperaturas.png',bbox_inches='tight')

#%%

c2vv = np.array([0.84, 0.50, 0.97, 0.66, 0.46])
errc2vv = np.array([0.03, 0.02, 0.05, 0.04, 0.04])
avv = np.array([0.065, 0.02, 0.040, 0.032, 0.004])
erravv = np.array([0.004, 0.014, 0.005, 0.006, 0.006])

nuvv = c2vv*avv
errnuvv = np.sqrt((avv * errc2vv)**2 + (c2vv * erravv)**2)

porcvv = np.array([55, 50, 40, 30, 25])
plt.figure(figsize = (8,6))
plt.errorbar(porcvv, nuvv, yerr = errnuvv, fmt = 'o', capsize=5, label = r'$\nu$ propagado')
plt.grid(True, linestyle='--', alpha = 0.5)
plt.ylabel(r'Viscosidad $\nu$ (cm)', fontsize = 14)
plt.xlabel(r'% V/V glicerina ', fontsize = 14)
plt.legend(fontsize = 12)
plt.savefig('nu_porcentaje.png',bbox_inches='tight')

#%%
print(nuvv)
print(errnuvv)
#%%
os.chdir('/home/juan/Documents/Laboratorio 5')

#%%

'''
50_1
Chi^2 reducido: 2.8938834896390695
[ 0.89689063  0.47254885 -0.02285262]
[0.01559502 0.0132281  0.00600658]
Chi^2 reducido: 3.804358631625075
[0.02691935 0.46029785]
[0.00793747 0.04586325]

50_2
Chi^2 reducido: 3.7858170425079423
[ 1.01456083  0.50377399 -0.01881658]
[0.01997736 0.01750286 0.00729756]
25.253228121864197
[-0.00136461  0.38510077]
[0.01435727 0.08885553]

50_3
Chi^2 reducido:
3.968683020095336
[ 0.96085768  0.46719248 -0.02029955]
[0.01882706 0.01591593 0.00699184]
17.984770689960644
[-0.01542763  0.33097665]
[0.01135051 0.06836091]

'''
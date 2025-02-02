from data import *
import numpy as np
import math as m
from scipy import interpolate
from scipy import integrate
from scipy import optimize
from scipy import ndimage
import os, sys
from os import listdir
from os.path import isfile, join
import h5py as h5
import time
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib as mpl
import cupy
import cupyx.scipy.ndimage

# Define a function that returns lambda(z) (for cosm const) at z
# BL01 eq. 23

# critical density in solar masses per Mpc^3 NOT including h^2
CRITDENMSOLMPC = 2.7755e11

OMEGA0 = 0.308  # matter fraction today
LAMBDA0 = 0.692  # dark energy fraction today
HPARAM = 0.678  # hubble constant today/100
OMEGAB = 0.046

KM_MPC = 3.08567758e19 # km in a Mpc 

# This is scaling the relations that assumed initialls that the
# CMB temp was 2.7
TCMB = 2.726 #CMB temp today
thetaCMB = TCMB / 2.7

SEC_YEAR = 3.154e7

om0hh = OMEGA0 * HPARAM * HPARAM
zEquality = 2.50e4 * om0hh * pow(thetaCMB, -4.0) - 1.0

MSUN = 1.989e+33 # g in a solar mass
ALPHAB = 2.0e-13 # case B rec coef in cm^3/s

PROTON_MASS = 1.6726e-24 # g

DENCGS = MSUN /  3.0e21**3 #* OMEGAB / OMEGA0 # multiply density from simulation in MSUN/kpc^3 to get to CGS


def lambdaZ(z):
	z = z + 1.
	temp = LAMBDA0
	temp = temp / (temp + OMEGA0 * z**3 + ((1. - OMEGA0 - LAMBDA0) * z**2))
	return temp

# Define a function that returns the growth function at a redshift z
# From Einstein and Hu


def growthFac(z):

	omZ = omegaZ(z=z)
	lamZ = lambdaZ(z=z)

	D = ((1.0 + zEquality) / (1.0 + z) * 5.0 * omZ / 2.0 * pow(pow(omZ, 4.0 / 7.0) - lamZ + (1.0 + omZ / 2.0) * (1.0 + lamZ / 70.0), -1.0))
	D = D / ((1.0 + zEquality) * 5.0 / 2.0 * OMEGA0 * pow(pow(OMEGA0, 4.0 / 7.0) - LAMBDA0 + (1.0 + OMEGA0 / 2.0) * (1.0 + LAMBDA0 / 70.0), -1.0))
	return D

# Define a function that returns omega dark matter at z
# for arbitrary cosmology
# BL01 eq. 23


def omegaZ(z):
	z = z + 1.
	temp = OMEGA0 * z**3.0
	temp = temp / (temp + LAMBDA0 + (1. - OMEGA0 - LAMBDA0) * z**2)

	return temp

###Fit to the *linear* overdensity at which virialization takes place, 
### it is 1.69 with slight cosmology dependence.

def delCrit0(z):

	t1 = 0.15*(12*m.pi)**(2./3.)
	omZ = 0+omegaZ(z=z)

	if type(omZ) == float or type(omZ) == int or type(omZ) == np.float64:
		if abs(omZ-1) < 1.0e-5:
			return t1
		elif abs(LAMBDA0) < 1.0e-5:
			t1 = t1*omZ**0.0185
			return t1
		else:
			t1 = t1*omZ**0.0055
			return t1

	elif type(omZ) == np.ndarray:
		ans = np.zeros(len(omZ))
		for i in range(len(omZ)):
			if abs(omZ[i]-1) < 1.0e-5:
				ans[i]=t1
			elif abs(LAMBDA0) < 1.0e-5:
				temp_t1 = t1*omZ[i]**0.0185
				ans[i]=temp_t1
			else:
				temp_t1 = t1*omZ[i]**0.0055
				ans[i]=temp_t1
		return ans

	else:
		print('HEY THE INPUT OF z IS WEIRD')

def delta_c(z):
    a = 18.0 * np.pi * np.pi
    b = 82.0 * (omegaZ(z) - 1.0)
    c = 39.0 * (omegaZ(z) - 1.0)**2.0
    return a + b - c;

def M_vir(T, z):
    num = 1.0e8 * T**(3.0/2.0)
    a = 1.98e4
    b = 1.22/0.6
    
    c = (OMEGA0 /omegaZ(z) * delta_c(z) / (18.0 * np.pi * np.pi))**(1.0/3.0)
    e = (1.0 + z) / 10.0 * HPARAM**(2.0/3.0)
    
    return num / ((a*b*c*e)**(3.0/2.0))

# eq 7 in tramonte et al 2017
def linDen_from_realDen(dm, z):
	dc = delCrit0(z=z) / growthFac(z=z)
	a = dc / 1.68647
	b = 1.68647 - 1.35 / (1.0 + dm)**(2./3.) - 1.12431 / (1.0 + dm)**(1./2.) + 0.78785 / (1.0 + dm)**(0.58661)
	return a*b

def cosmicTime(z):
    if(abs(OMEGA0 - 1.0) <= 1.0e-3):
        ht = 2.0/3.0/(1.0 + z)**1.5
        return ht
    if(abs(OMEGA0 + LAMBDA0 - 1.0) <= 1.0e-3):
        temp = np.sqrt((1.0-OMEGA0)/OMEGA0)/ (1.0 + z)**1.5
        ht = np.log(temp+np.sqrt(1.0+temp*temp))
        ht *= 2.0/3.0/np.sqrt(1.0-OMEGA0)
        return ht;
    return 0.0

# time in seconds between z1 and z2
def time_sep(z1, z2):
    diff = abs(cosmicTime(z1) - cosmicTime(z2));
    return diff / (HPARAM * 100 / KM_MPC)

def realDen_from_linDen(linDen, name='/data/groups/comp-astro/rmebane/realDen_from_linDen_splrep_interp.npy', **kwargs):

    # read in interp file
    tckD = np.load(name, allow_pickle=True)

    if type(linDen) == np.ndarray:
        ans = np.zeros(len(linDen))
        for i in range(len(linDen)):
            if linDen[i] >= 1.5874:
                ans[i] = 18 * m.pi**2
            else:
                ans[i] = interpolate.splev(linDen[i], tckD)
        return ans

    else:
        if linDen >= 1.5874:
            return 18 * m.pi**2
        else:
            ans = interpolate.splev(linDen, tckD).item()
            return ans


def nm_press_cond(tM, z, regionMass, d):
    #   returns the conditional press schechter halo mass function as dn / dm
    #   sigma(m) table given to me by Adam Trapp
    sigm_table = np.load('/data/groups/comp-astro/rmebane/sigma0fM_interp.npy', allow_pickle=True)
    sig = 10**interpolate.splev(np.log10(regionMass), sigm_table, der=0)
    sig_M = 10**interpolate.splev(np.log10(tM), sigm_table, der=0)
    if(tM > regionMass):
        #print(tM)
        x=1
    sig_Mprime = np.sqrt(sig_M**2 - sig**2)

    dlsdlM = interpolate.splev(np.log10(tM), sigm_table, der=1)
    dlsdlMprime = sig_M**2 / sig_Mprime**2 * dlsdlM

    dCritZprime = delCrit0(z=z) / growthFac(z=z) - d

    tdn = np.log10(np.sqrt(2.0 / m.pi) * dCritZprime * abs(dlsdlMprime))
    tdn += np.log10(np.exp(1)) * (-dCritZprime**2 / (2.0 * sig_Mprime**2))
    tdn -= np.log10(tM * sig_Mprime)
    tdn += np.log10(CRITDENMSOLMPC * om0hh)
    # THIS LINE IS TO CONVERT TO AN EULERIAN DENSITY. I get fancy here. If you don't want to
    # use my realDen_from_linDen function (and associated interp file), you can just multiply by (1 + d1 * growthFac(z=z)) instead
    #if withVolChange:
    realDen = realDen_from_linDen(linDen=d * growthFac(z=z))
    #tdn += np.log10(1 + d * growthFac(z=z))
    tdn += np.log10(1 + realDen)

    return 10**tdn / tM





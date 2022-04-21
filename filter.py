from cosmo import *
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

def maxone(a):
    if(a > 1.0):
        return 1.0
    return a

# density in CGS
# dt in seconds
# returns in total number of recombinations per baryon
# This assumes a mean density, you should probably instead integrate over the subgrid density distribution below
def get_nrec(den, dt, ion_frac = 1.0):
    ion_density = den * OMEGAB / OMEGA0 * 0.76 / PROTON_MASS * ion_frac # number density of ions
    return ion_density * ALPHAB * dt

# Code for integrating over the subgrid density distribution to find nrec
# See Sobacchi and Mesinger 2014
def density_pdf_Cnorm(D, z, Dcell, A, C):
    Beta = 2.5
    zeff = (1.0 + z) * Dcell**(1./3.) - 1.
    d0 = 7.61 / (1.0 + zeff)
    expon = -1. * (D**(-2./3.) - C)**2.0 / (2.0 * (2. * d0 / 3.)**2.0)
    return A * np.exp(expon) * D**(-1. * Beta)

def expectation_int_Cnorm(D, z, Dcell, A, C):
    return D * density_pdf_Cnorm(D, z, Dcell, A, C)

def density_expectation_Cnorm(C, z, Dcell):
    A = 1. / integrate.quad(density_pdf_Cnorm, 0, np.inf, args=(z, Dcell, 1.0, C))[0]
    return Dcell - integrate.quad(expectation_int_Cnorm, 0, np.inf, args=(z, Dcell, A, C))[0]

def find_C(z, Dcell):
    return optimize.fsolve(density_expectation_Cnorm, Dcell**(-2./3.), args=(z, Dcell))

def density_pdf(D, z, Dcell):
    C = find_C(z, Dcell)
    A = 1. / integrate.quad(density_pdf_Cnorm, 0, np.inf, args=(z, Dcell, 1.0, C))[0]
    Beta = 2.5
    zeff = (1.0 + z) * Dcell**(1./3.) - 1.
    d0 = 7.61 / (1.0 + zeff)
    expon = -1. * (D**(-2./3.) - C)**2.0 / (2.0 * (2. * d0 / 3.)**2.0)
    return A * np.exp(expon) * D**(-1. * Beta)

# Equation 13 in Sobacchi and Mesinger 2014 for dnrec/dt
def dnrec_dt_integrand(D, z, Dcell, rho_mean = 72.55 * DENCGS):
    a = 1.0 / (1.0 + z)
    nH_mean = rho_mean * OMEGAB / OMEGA0 * 0.76 / PROTON_MASS / a**3
    return density_pdf(D, z, Dcell) * D * nH_mean * ALPHAB

# Dcell is big Delta overdensity
def dnrec_dt(Dcell, z):
    return integrate.quad(dnrec_dt_integrand, 0, np.inf, args=(z, Dcell))[0]

def fstar_test(z, M):
    return 0.1

def fstar_21cmFAST(z, M):
    return 0.05 * (M / 1.0e10)**0.5

def fesc_21cmFAST(z, M):
    return 0.1 * (M / 1.0e10)**(-0.5)

def fesc_test(z, M):
    return 0.1

def Nion_test(z, M):
    return 5.0e3

def Mmin_test(z):
    return 1.0e8 * MSUN

def Mmin_ACT(z):
    return M_vir(1.0e4, z) * MSUN

def fcoll_integrand(M, z, regionMass, d, fstar, fesc, Nion):
    nm = nm_press_cond(M / MSUN, z, regionMass / MSUN, d)
    return nm / MSUN / 3.0e24**3 * fstar(z, M) * OMEGAB / OMEGA0 * Nion(z, M) * fesc(z, M) * M

# actual fcoll, without SF params
def fcoll_true_integrand(M, z, regionMass, d):
    nm = nm_press_cond(M / MSUN, z, regionMass / MSUN, d)
    return nm / MSUN / 3.0e24**3 * M

def fcoll_ion_s(z, regionMass, d, rho, fstar=fstar_21cmFAST, fesc=fesc_21cmFAST, Nion=Nion_test, Mmin=Mmin_ACT, debug_flag=1):
	# set the max integration mass equal to the region mass (can't have a halo more massive than the region!)
	Mmax = regionMass
	N = 10
	Ms = np.logspace(np.log10(Mmin(z)), np.log10(Mmax), N)
	Ms_den = np.logspace(np.log10(1), np.log10(Mmax), N)
	#res = integrate.quad(fcoll_integrand, Mmin(z), Mmax, args=(z, regionMass, d, fstar, fesc, Nion))
	res = 0
	den = 0
	i = 0
	while(i < N - 1):
		res += integrate.quad(fcoll_integrand, Ms[i], Ms[i+1], args=(z, regionMass, d, fstar, fesc, Nion))[0]
		den += integrate.quad(fcoll_true_integrand, Ms_den[i], Ms_den[i+1], args=(z, regionMass, d))[0]
		i += 1
	if(debug_flag==1):
		return res / den #rho
	if(debug_flag==2):
		return den
	return res/rho

def build_1d_interp_table(z, den, output, N=1000, rMass = 1.0e16*MSUN):
    if(USE_CUPY):
        rhomean = cupy.mean(den)
        rhomax = cupy.max(den)
        rhomin = cupy.min(den[cupy.nonzero(den)])
        dmax = (rhomax - rhomean) / rhomean
        dmin = (rhomin - rhomean) / rhomean
        ds = cupy.linspace(linDen_from_realDen(dmin, z), linDen_from_realDen(dmax, z), N)
        res = cupy.zeros(N)
        rho_mean = cupy.mean(den) * MSUN /  3.0e21**3 * OMEGAB / OMEGA0
    else:
        rhomean = np.mean(den)
        rhomax = np.max(den)
        rhomin = np.min(den[np.nonzero(den)])
        dmax = (rhomax - rhomean) / rhomean
        dmin = (rhomin - rhomean) / rhomean
        ds = np.linspace(linDen_from_realDen(dmin, z), linDen_from_realDen(dmax, z), N)
        res = np.zeros(N)
        rho_mean = np.mean(den) * MSUN /  3.0e21**3 * OMEGAB / OMEGA0
    for i in range(0,N):
        print(i)
        res[i] = fcoll_ion_s(z, rMass, ds[i], rho_mean*ds[i] + rho_mean, debug_flag=1)
    if(USE_CUPY):
        cupy.savetxt(output, cupy.c_[ds, res])
    else:
        np.savetxt(output, np.c_[ds, res])

# returns distance from center in number of cells of pos in a cube of side length s
def distance_from_center(s, pos):
    dx = abs(pos[0] - s/2.)
    dy = abs(pos[1] - s/2.)
    dz = abs(pos[2] - s/2.)
    return np.sqrt(dx**2.0 + dy**2.0 + dz**2.0)

# returns a spherical kernel of cell radius R to convolve with a density field
def spherical_kernel(r):
    R = int(np.ceil(r))
    if(USE_CUPY):
        res = cupy.ones((2*R, 2*R, 2*R))
    else:
        res = np.ones((2*R, 2*R, 2*R))
    for i in range(0,2*R):
        for j in range(0,2*R):
            for k in range(0,2*R):
                if(distance_from_center(R, (i,j,k)) > r):
                    res[i,j,k] = 0
    if(USE_CUPY):
        s = cupy.sum(res)
    else:
        s = np.sum(res)
    return res / s

# no .dat in output!
# box_len in Mpc
def filter_box(z, zprev, den, box_len, output_prefix, Nrec = [], logR = True, max_R = 50, min_R = -1, NR=25, imax=-1, jmax=-1, kmax=-1, interp_file="", avg_rec=0.0, save_steps=False, save_full_box=False, boost=1.0, cf = 1.0):
    #unless min filtering scale set, set it to the cell size
    if(min_R < 0):
        min_R = box_len / len(den)
    # decide the filtering scales
    if(logR):
        Rs = np.logspace(np.log10(max_R), np.log10(min_R), NR)
    else:
        Rs = np.linspace(max_R, min_R, NR)
    cell_size = box_len / len(den)
    # filter radii in units of number of cells
    Rcs = Rs / cell_size
    N = len(den)
    if(USE_CUPY):
        effs = cupy.empty_like(den)
        if(len(Nrec) != 0):
            Nrec_new = cupy.empty_like(den)
    else:
        effs = np.empty_like(den)
        if(len(Nrec) != 0):
            Nrec_new = np.empty_like(den)
    if(USE_CUPY):
        rho_mean = cupy.mean(den)
    else:
        rho_mean = np.mean(den)
    if(interp_file != ""):
        #can probably do this differently, most values are dummies since we are just reading in a file
        #currently just doing a 1d interp since it's much faster for high res sims 
        #Results about the same, but need further testing
        #interp_table = build_nion_table(z, den, 1, 1, 10, 0, 1, in_file=interp_file)
        (dsi, effsi) = read_1d_interp_table(interp_file)
    if(imax < 0):
        imax=N
    if(jmax < 0):
        jmax=N
    if(kmax < 0):
        kmax=N
    # generate the convolved fields
    for r in Rcs:
        kernel = spherical_kernel(r)
        if(USE_CUPY):
            f = cupyx.scipy.ndimage.convolve(den, kernel, mode="wrap")
            # Skip recombinations if a previous Nrec field is not provided
            # this is useful if you're just doing a single snapshot at high z
            if(len(Nrec) != 0):
                Nrec_filtered = cupyx.scipy.ndimage.convolve(Nrec, kernel, mode="wrap")
        else:
            f = ndimage.convolve(den, kernel, mode="wrap")
            if(len(Nrec) != 0):
                Nrec_filtered = ndimage.convolve(Nrec, kernel, mode="wrap")
        print("Generated smoothed density field at R=" + "{:.2f}".format(r*cell_size) + " Mpc...")
        vol = 4.0/3.0 * np.pi * (r * cell_size* 1.0e3)**3.0
        ia = []
        ja = []
        ka = []
        iona = []
        nreca = []
        Ra = []
        dena = []
        for i in range(0, imax):
            for j in range(0, jmax):
                for k in range(0, kmax):
                    if(effs[i,j,k] < 1.0 and f[i,j,k] > 0.0):
                    	# enclosed mass calculation not necessary in the 1d interp case
                        mass = vol * f[i,j,k]
                        dr = (f[i,j,k] - rho_mean) / rho_mean
                        # convert to linear overdensity
                        d = linDen_from_realDen(dr, z)
                        # Compute the ionizing efficiency if a table isn't provided, or interpolate if it is
                        # boost factor is just for testing
                        if(interp_file == ""):
                            eff = fcoll_ion_s(z, mass * MSUN, d, 0) * boost
                        else:
                            if(USE_CUPY):
                                eff = cupy.interp(d, dsi, effsi) * boost
                            else:
                                eff = np.interp(d, dsi, effsi) * boost # 1D
                        effs[i,j,k] = eff
                        # subtract off total recombinations
                        # efficiency field is now equivalent to an ionized field
                        if(len(Nrec) != 0):
                            effs[i,j,k] -= Nrec_filtered[i,j,k]
                        # If there are more recombinations than possible, set efficiency to 0
                        # isnan check probably isn't necessary, just there in case we accidentally
                        # filter a region with 0 density in a low res simulation
                        if(effs[i,j,k] < 0.0 or np.isnan(effs[i,j,k])):
                            effs[i,j,k] = 0.0


                    # save at all filter steps if true
                    #effs = np.clip(effs, 0.0, 1.0)
                    if(save_steps==True):
                        ia.append(i)
                        ja.append(j)
                        ka.append(k)
                        iona.append(maxone(effs[i,j,k]))
                        Ra.append(r * cell_size)
                        dena.append(den[i,j,k])
                    if(len(Nrec) != 0):
                        dt = time_sep(z, zprev)
                        Nrec_new[i,j,k] = Nrec[i,j,k] + get_nrec(den[i,j,k] * DENCGS, dt, ion_frac=maxone(effs[i,j,k])) * cf
        if(save_steps==True):
            np.savetxt(output_prefix + "_r" + str(r*cell_size) + ".dat", np.c_[ia, ja, ka, iona, Ra, dena])


                    #m = np.log10(get_mass((i,j,k), R, den, z, 300))
                    #print(str(i) + " " + str(j) + " " + str(k) + " " + str(ion) + " " + str(R) + " " + str(den[i,j,k]))# + " " + str(m))
    if(USE_CUPY):
        effs = cupy.clip(effs, 0.0, 1.0)
    else:
        effs = np.clip(effs, 0.0, 1.0)
    ion_den = 0
    total_den = 0
    for i in range(0, imax):
        for j in range(0, jmax):
            for k in range(0, kmax):
                ia.append(i)
                ja.append(j)
                ka.append(k)
                iona.append(effs[i,j,k])
                nreca.append(Nrec_new[i,j,k])
                Ra.append(0)
                dena.append(den[i,j,k])
                total_den += den[i,j,k]
                ion_den += den[i,j,k] * effs[i,j,k]
    if(USE_CUPY):
        ia = cupy.array(ia)
        ja = cupy.array(ja)
        ka = cupy.array(ka)
        iona = cupy.array(iona)
        nreca = cupy.array(nreca)
        Ra = cupy.array(Ra)
        dena = cupy.array(dena)
        # if we are filtering the entire box, just save everything
        if(save_full_box):
        	cupy.save(output_prefix + "_ion", effs)
        	cupy.save(output_prefix + "_nrec", Nrec_new)
        # if we are just filtering part of the box for testing, just save important values
       	else:
        	cupy.save(output_prefix, cupy.array([ia, ja, ka, iona, nreca, Ra, dena]))

    else:
        np.savetxt(output_prefix + ".dat", np.c_[ia, ja, ka, iona, Ra, dena])
    # returns ion field, new Nrec field, neutral fraction
    return (effs, Nrec_new, (total_den - ion_den) / total_den)










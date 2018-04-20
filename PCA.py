# -*- coding: utf-8 -*-
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from test_ppxf import fit_ppxf
from meanclip import meanclip
from photutils import aperture_photometry, centroid_com, CircularAperture
import pdb
import os
from astropy import units as u
from astropy.analytic_functions import blackbody_lambda, blackbody_nu
# Declear all PATHs being used
Object = '/Users/antoinedumont/Documents/s_research/flux_calibration/ngc404/ngc404_combine_near9_fluxcal.fits'  # Path of file of Object 
templates = 'Selection' # enter 'All', 'Single' or 'Selection' 
Radii= '8' # Radii in multiples of 0.1"
calculate_flux = 'No' # enter 'Yes' or 'No' if you want to get the flux
minfit = 20150    # Start of the target spectra to be extracted
maxfit = 24250   # End of the target spectra to be extracted
z = -0.000597    # Redshift estimation from NED
###--------------###–---------------####-------------###–---------------###------
hdulist = fits.open(Object)
sci=hdulist[1]
var=hdulist[2]
# Declear all constant values being used

name_object=sci.header['OBJECT']
lambda0=sci.header['CRVAL3']
dlambda=sci.header['CD3_3']
s=range(sci.header['NAXIS3'])
lambda1=np.array(s)*dlambda+lambda0
pixscale = sci.header['PIXSCALE']  # Arcsec/pixel

#Find the center of the galaxy
ifu=sci.data
#Make a S/N map for finding the correct center of the galaxy
signal = np.zeros((np.shape(ifu)[1], np.shape(ifu)[2]))
noise = np.zeros((np.shape(ifu)[1], np.shape(ifu)[2]))
mask1 = (lambda1 >= 22140) & (lambda1 <= 22600)  # Range for S/N
mask2 = (lambda1 >=(min(lambda1))) & (lambda1 <=(max(lambda1)))    # Range for the whole spectrum
for k in range(np.shape(ifu)[1]-1):
    for l in range(np.shape(ifu)[2]-1):
            uspec = ifu[mask1, k, l]
            mean, sigma, subs = meanclip(uspec, clipsig=3, returnSubs=True)
            signal[k, l] = mean
            noise[k, l] = sigma

signoise= signal/noise
# Make integrated spectrum within 0.5-1.0 arcsecs
imsize = np.shape(signal)                   
maxval = np.max(signal)
maxpos = np.where(signal == np.max(signal))[0]
xinit = maxpos % imsize[1]
yinit = maxpos/imsize[1]
print('Initial center: ', xinit, yinit)
xcen, ycen = np.unravel_index(signal.argmax(), signal.shape)
print('Target center: ', xcen, ycen)
rad1arc = (0.1/pixscale)         # 0.1" in pixels
wave = lambda1[mask2]         # in angstrom


c1=range(np.shape(ifu)[1])
c2=range(np.shape(ifu)[2])
######------------------#######–-----------------########–--------------------
# Delete SKYLINE 
mask_badpixels = [] 
for row in open('/Users/antoinedumont/Documents/s_research/flux_calibration/ppxf_fit/rousselot2000.dat'):
     mask_badpixels.append(row)
mask_badpixels=np.array(mask_badpixels)
badpixels = np.zeros((mask_badpixels[30:].shape))
for i,j in enumerate(mask_badpixels[30:]):
    badpixels[i]=j[:9]
mask_badpixels = np.ones((lambda1.shape),bool)
for i,j in enumerate(lambda1):
     for k in badpixels:
         if abs(j-k)<=dlambda:
             mask_badpixels[i]=False
             mask_badpixels[i-1]=False
             mask_badpixels[i+1]=False
    
######------------------#######–-----------------########–--------------------
lambda1 = lambda1[mask_badpixels]
             
ifu_new = []
for i in c1:
    for j in c2:
        rad=np.sqrt((i-xcen)**2+(j-ycen)**2)    
        # If rad is inner part of the circle
        if rad <= float(Radii)*rad1arc: # this loop get flux at each pixel
            mask = (lambda1 >= minfit) & (lambda1 <= maxfit)
            ifu_new.append(ifu[:,i,j][mask_badpixels][mask]/np.median(ifu[mask_badpixels][mask][:,i,j]))
    
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca.fit(ifu_new)
mean = pca.mean_
components = pca.components_


fig = plt.figure(figsize=(5, 5))
fig.subplots_adjust(hspace=0, top=0.95, bottom=0.1, left=0.12, right=0.93)
for i in range(4):
    ax = fig.add_subplot(511+(i+1))
    ax.set_title('component {}'.format(i))
    ax.plot(lambda1[mask]/(1+z),components[i],'k')
ax = fig.add_subplot(511+0)
ax.set_title('mean')
ax.plot(lambda1[mask]/(1+z),mean,'k')
plt.show(block=False)

# comulative eigenvalue
evals = pca.explained_variance_
evals_cs = np.array([i/sum(evals) for i in evals])

#Matrices pixels and slopes
#coefficients = np.zeros((np.shape(ifu)[1], np.shape(ifu)[2]))
coefficients = []
pixels1 = []
pixels2 = []
fluxes = []
for i in c1:
    for j in c2:
        nm=signal[i,j]
        if nm !=np.nan:
            if nm != 0.0:
                # Calculate the Radius:
                rad=np.sqrt((i-xcen)**2+(j-ycen)**2)    
                # If rad is inner part of the circle
                if rad <= float(Radii)*rad1arc: # this loop get flux at each pixel
                    median_flux = np.median(ifu[mask_badpixels][mask][:,i,j])
                    flux = ifu[:,i,j][mask_badpixels][mask]/median_flux
                    coeff = np.dot(components,flux-mean)
                    coefficients.append(coeff[0])
                    if rad <= 2*rad1arc:
                        bb_flux = np.trapz((mean)*median_flux*coeff[0]*evals_cs[0],x=lambda1[mask],dx=dlambda)
                        fluxes.append(bb_flux)
                    pixels1.append(i)
                    pixels2.append(j)
# colormap for continuum slopes
xcen_new,ycen_new= xcen*pixscale,ycen*pixscale
piels1_new=np.array(pixels1)*pixscale; piels2_new=np.array(pixels2)*pixscale
pixels1_new = piels1_new-xcen_new; pixels2_new=piels2_new-ycen_new
fig,ax=plt.subplots()

cax=ax.scatter(pixels2_new,pixels1_new,c=coefficients,s=80,marker='s',cmap='viridis')
cax2=ax.scatter(0,0,color='r',s=30)
cbar=fig.colorbar(cax,label='value Eigenvalue 1th Egeinspectra')
ax.set_xlabel('Arcsec')
ax.set_ylabel('Arcsec')
plt.title(name_object)
plt.show(block=False)

print 'blackbody flux {} erg/s/cm^2'.format(np.sum(fluxes))

# comulative eigenvalue plot

fig = plt.figure(3)
ax2 = plt.axes()
ax2.grid()
ax2.plot(evals_cs, color='k')
ax2.set_xlabel('Eigenvalue Number')
ax2.set_ylabel('Cumulative Eigenvalues')
plt.show(block=False)

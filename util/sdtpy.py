#================================================================
#   File name   : module.py
#   Author      : Gregory S.H. Paek
#   Created date: 2022-12-15
#   Website     : -
#================================================================
import warnings
warnings.filterwarnings('ignore')
import os
import glob
import time
import numpy as np
from astropy.table import Table, vstack, QTable, join
from astropy.io import ascii
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import WMAP9 as cosmo
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.special import erf
from scipy.integrate import trapezoid
from scipy.stats import norm
import speclite.filters
#	Plot presetting
import matplotlib.pyplot as plt
import matplotlib as mpl
#
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')
#
# Constants
c_ums = 3e14                  # c in um/s
c = 3e8                       # m/s
h = 6.626e-34                 # Planck constant   [J/Hz]
k = 1.38e-23                  # Boltzman constant [J/K]
rad2arcsec = (180/np.pi*3600) # 206265 arcsec
arcsec2rad = 1/rad2arcsec
#	Unit
lamunit = u.Angstrom
flamunit = u.erg/u.second/u.cm**2/u.Angstrom
#
def func_linear(x, a, b):
	"""Function for making natural drop in a shorter wavelength of CMOS QE, optics transmission

	Parameters
	----------
	x : float
		_description_
	a : float
		_description_
	b : float
		_description_

	Returns
	-------
	float
		_description_
	"""
	# return a*x**2+b
	return a*np.log(x)+b
#
def tophat_trans(x, center=0, fwhm=1, smoothness=0.1):
	from scipy.special import erf, erfc
	t_left  = erfc(+((2*(x-center)/fwhm)-1)/smoothness)/2
	t_right = erfc(-((2*(x-center)/fwhm)+1)/smoothness)/2
	return (t_left*t_right)
#
def makeSpecColors(n, palette='Spectral'):
	#	Color palette
	import seaborn as sns
	palette = sns.color_palette(palette, as_cmap=True,)
	palette.reversed

	clist_ = [palette(i) for i in range(palette.N)]
	cstep = int(len(clist_)/n)
	clist = [clist_[i*cstep] for i in range(n)]
	return clist
#
def convert_lam2nu(lam):
	nu = (const.c/(lam)).to(u.Hz)
	return nu
#
# def convert_fnu2flam(fnu, lam):
# 	flam = (fnu*const.c/(lam**2)).to((u.erg/((u.cm**2)*u.second*u.Angstrom)))
# 	return flam
#
def convert_fnu2flam(fnu, lam):
    fnu_cgs = fnu.to(u.erg / (u.cm**2 * u.s * u.Hz))  # Convert mJy to erg / (cm^2 * s * Hz)
    lam_cm = lam.to(u.cm)  # Convert Angstrom to cm
    flam = (fnu_cgs * const.c / (lam_cm**2)).to(u.erg / (u.cm**2 * u.s * u.Angstrom), equivalencies=u.spectral_density(lam_cm))
    return flam

# def convert_flam2fnu(flam, lam):
# 	# c = const.c.to(u.cm/u.second)
# 	# fnu = (flam*lam.to(u.cm)**2/c).to((u.erg/(u.cm**2 * u.second * u.Hz)))
# 	c = const.c
# 	cval = c.value
# 	flamval = flam.to(u.erg/(u.cm**2 * u.second * u.Hz)).value
# 	lamval = lam.to(u.m).value
# 	fnu = (1e9*flamval*(lamval**2)/(cval))*(u.erg/(u.cm**2 * u.second * u.Hz))
# 	return fnu
# def convert_flam2fnu(flam, lam):
# 	c = const.c.to(u.cm/u.second)
# 	fnu = (flam*lam.to(u.cm)**2/c).to((u.erg/(u.cm**2 * u.second * u.Hz)))
# 	return fnu
# def convert_flam2fnu(flam, lam):
# 	c = const.c.to('cm/s')
# 	fnu = (flam*lam**2/const.c).to((u.erg/((u.cm**2)*u.second*u.Hz)))
# 	return fnu
# def convert_flam2fnu(flam, lam):
#     c = const.c.to('cm/s')
#     fnu = (flam * lam**2 / c).to((u.erg / ((u.cm**2) * u.second * u.Hz)))
#     return fnu
# def convert_flam2fnu(flam, lam):
#     c = const.c.to('cm/s')
#     lam_cm = lam.to(u.cm)  # Convert lam to centimeters
#     fnu = (flam * lam_cm**2 / c).to((u.erg / ((u.cm**2) * u.second * u.Hz)))
#     return fnu
def convert_flam2fnu(flam, lam):
    c = const.c.to('cm/s')
    lam_cm = lam.to(u.cm)  # Convert lam to centimeters
    fnu = (flam * lam_cm**2 / c).to(u.erg / (u.cm**2 * u.s * u.Hz), equivalencies=u.spectral_density(lam_cm))
    return fnu
#
def convert_app2abs(m, d):
	M = m - (5*np.log10(d)-5)
	return M
#
def convert_abs2app(M, d):
	m = M + (5*np.log10(d)-5)
	return m
#
def synth_phot(wave, flux, wave_lvf, resp_lvf, tol=1e-3, return_photonrate = False):
    """
    Quick synthetic photometry routine.

    Parameters
    ----------
    wave : `numpy.ndarray`
        wavelength of input spectrum.
    flux : `numpy.ndarray`
        flux density of input spectrum in f_nu unit
        if `return_countrate` = True, erg/s/cm2/Hz is assumed
    wave_lvf : `numpy.ndarray`
        wavelength of the response function
    resp_lvf : `numpy.ndarray`
        response function. assume that this is a QE.
    tol : float, optional
        Consider only wavelength range above this tolerence (peak * tol).
        The default is 1e-3.

    Returns
    -------
    synthethic flux density in the input unit
        if return_photonrate = True, photon rates [ph/s/cm2]

    """
    index_filt, = np.where(resp_lvf > resp_lvf.max()*tol)

    index_flux, = np.where(np.logical_and( wave > wave_lvf[index_filt].min(), 
                                           wave < wave_lvf[index_filt].max() ))

    wave_resamp = np.concatenate( (wave[index_flux], wave_lvf[index_filt]) )
    wave_resamp.sort()
    wave_resamp = np.unique(wave_resamp)
    flux_resamp = np.interp(wave_resamp, wave, flux)
    resp_resamp = np.interp(wave_resamp, wave_lvf, resp_lvf)

    if return_photonrate:
        h_planck = 6.626e-27 # erg/Hz
        return trapezoid(resp_resamp / wave_resamp * flux_resamp, wave_resamp) / h_planck
        
    return trapezoid(resp_resamp / wave_resamp * flux_resamp, wave_resamp) \
         / trapezoid(resp_resamp / wave_resamp, wave_resamp)


def calculate_aperture_fraction(seeing, optfactor, figure=True):
	# np.random.seed(0)
	# seeing = 1.5
	# optfactor = 0.6731
	mu = 0.0
	sigma = seeing*2.3548

	x = np.linspace(-8, 8, 1000)
	y = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x-mu)**2 / (2 * sigma**2))
	y_cum = 0.5 * (1 + erf((x - mu)/(np.sqrt(2 * sigma**2))))

	indx_aperture = np.where(
		(x>-sigma*optfactor) &
		(x<+sigma*optfactor)
	)
	xaper = x[indx_aperture]
	yaper = y[indx_aperture]

	frac = np.sum(yaper)/np.sum(y) 

	if figure:
		plt.plot(x, y, alpha=0.7, label=f'PDF of N(0, {sigma:1.3f})')
		plt.plot(xaper, yaper, alpha=0.75, label=f'Aperture ({frac*1e2:.1f}%)', lw=5,)
		plt.xlabel('x', fontsize=20)
		plt.ylabel('f(x)', fontsize=20)
		plt.legend(loc='lower center', fontsize=14)
		plt.show()
	return frac
#
def convert_snr2magerr(snr):
	merr = 2.5*np.log10(1+1/snr)
	return merr

def get_random_point(mu, sigma, n=10):
	"""
	mu, sigma = 17.5, 0.1
	n = 10
	"""
	x = np.arange(mu-sigma*n, mu+sigma*n, sigma*1e-3)
	y = norm(mu, sigma).pdf(x)
	return np.random.choice(x, p=y/np.sum(y))

def apply_redshift_on_spectrum(spflam, splam, z, z0=0, scale=True):
	d = cosmo.luminosity_distance(z)
	#	Shifted wavelength
	zsplam = splam*(1+z)/(1+z0)
	#	z-->distance
	##	distance scaling
	if scale:
		zspfnu = convert_flam2fnu(spflam, zsplam)
		zspabsmag = zspfnu.to(u.ABmag)
		zspappmag = convert_abs2app(zspabsmag.value, d.to(u.pc).value)*u.ABmag
		zspappfnu = zspappmag.to(u.uJy)
		zspappflam = convert_fnu2flam(zspappfnu, zsplam)
		return (zspappflam, zsplam)
	else:
		return (spflam, zsplam)

def extract_param_kn_sim_cube(knsp):
	part = os.path.basename(knsp).split('_')

	if part[1] == 'TP':
		dshape = 'toroidal'
	elif part[1] == 'TS':
		dshape = 'spherical'
	else:
		dshape = ''

	#	Latitude
	if part[5] == 'wind1':
		lat = 'Axial'
	elif part[5] == 'wind2':
		lat = 'Edge'
	else:
		lat = ''

	#	Ejecta mass for low-Ye [solar mass]
	md = float(part[7].replace('md', ''))

	#	Ejecta velocity for low-Ye [N*c]
	vd = float(part[8].replace('vd', ''))

	#	Ejecta mass for high-Ye [solar mass]
	mw = float(part[9].replace('mw', ''))

	#	Ejecta velocity for high-Ye [N*c]
	vw = float(part[10].replace('vw', ''))

	#	Angle
	try:
		if 'angularbin' not in knsp:
			angle = float(part[11].replace('angle', ''))
		else:
			angle = int(part[11].replace('angularbin', ''))
	except:
		angle = None

	return (dshape, lat, md, vd, mw, vw, angle)


def fill_nan_with_interpolation(array):
	n = len(array)
	nan_indices = np.where(np.isnan(array.value))[0]  # .value to get the numerical part

	for i in nan_indices:
		left_idx = i
		right_idx = i

		# Find the closest non-nan value on the left
		while left_idx >= 0 and np.isnan(array[left_idx].value):
			left_idx -= 1

		# Find the closest non-nan value on the right
		while right_idx < n and np.isnan(array[right_idx].value):
			right_idx += 1

		# Calculate interpolated value based on neighbors
		if left_idx >= 0 and right_idx < n:
			array[i] = (array[left_idx] + array[right_idx]) / 2
		elif left_idx >= 0:  # if nan is on the right edge
			array[i] = array[left_idx]
		elif right_idx < n:  # if nan is on the left edge
			array[i] = array[right_idx]


#----------------------------------------------------------------
#	Main class
#----------------------------------------------------------------
class SevenDT:
	def __init__(self):
		#	Optics info.
		D = 50.5               # effetive diameter [cm]
		D_obscuration = 29.8   # Central Obscuration (diameter)
		EFL = 1537.3           # [mm]
		Deff = np.sqrt(D**2 - D_obscuration**2)
		self.d = D
		self.d_obscuration = D_obscuration
		self.d_efl = EFL
		self.d_eff = Deff
		#	Camera
		array = 'CMOS'       # detector array type
		dQ_RN = 3.           # [e], readout noise 
		I_dark = 0.01        # [e/s], dark current
		pixel_size = 3.76    # [um], "pitch"
		theta_pixel = 0.517  # [arcsec], pixel scale 
		nxpix, nypix = 9576, 6388  # [pixels], detector format, approx. 9k x 6k
		self.array = array
		self.dQ_RN = dQ_RN
		self.I_dark = I_dark
		self.pixel_size = pixel_size
		self.theta_pixel = theta_pixel
		self.nxpix = nxpix
		self.nypix = nypix

	def echo_optics(self):
		print(f'D             : {self.d}cm')
		print(f'D_obscuration : {self.d_obscuration}cm')
		print(f'Deff          : {self.d_eff:1.3f}cm')
	
	def generate_filterset(self, bandmin, bandmax, bandwidth, bandstep, bandrsp, lammin=1000, lammax=10000, lamres=1000):
		lam = np.arange(bandmin, bandmax, bandstep)
		wave = np.linspace(lammin, lammax, lamres)

		#	Create filter_set definition
		filter_set = {
			'cwl': lam,
			'wave': wave
			}

		filterNameList = []
		for ii, wl_cen in enumerate(lam):
			rsp = tophat_trans(wave, center=wl_cen, fwhm=bandwidth)*bandrsp
			# filter_set.update({f'{ii}': rsp})
			# filtername = f'm{wl_cen/1e1:g}'
			filtername = f'm{wl_cen:g}'
			filter_set.update({filtername: rsp})
			indx_resp = np.where(rsp>rsp.max()*1e-3)
			filterNameList.append(filtername)

		#	ticks
		step = 500
		ticks = np.arange(round(filter_set['cwl'].min(), -2)-step, round(filter_set['cwl'].max(), -2)+step, step)
		self.filterset = filter_set
		self.filterNameList = filterNameList
		self.lam = wave
		self.lammin = lammin
		self.lammax = lammax
		self.lamres = lamres
		self.bandcenter = lam
		self.bandstep = bandstep
		self.bandwidth = bandwidth
		self.color = makeSpecColors(len(lam))
		self.ticks = ticks
		return filter_set

	def plot_filterset(self):

		filterset = self.filterset

		plt.figure(figsize=(12, 4))
		#	Wavelength
		lam = filterset['wave']
		for ii, filtername in enumerate(self.filterNameList):
			#	Central wavelength
			cwl = filterset['cwl'][ii]
			#	Response [%]
			rsp = filterset[filtername]*1e2
			#	Cut the tails of curve
			indx_rsp = np.where(rsp>rsp.max()*1e-3)
			#	Plot
			plt.plot(lam[indx_rsp], rsp[indx_rsp], c=self.color[-ii-1], lw=3)
			if ii%2 == 0:
				plt.text(cwl, 100, filtername, horizontalalignment='center', size=9)
			elif ii%2 == 1:
				plt.text(cwl, 107.5, filtername, horizontalalignment='center', size=9)

		#	Plot Setting
		# step = 500
		# xticks = np.arange(round(filterset['cwl'].min(), -2)-step, round(filterset['cwl'].max(), -2)+step, step)
		# plt.xticks(xticks)
		plt.xticks(self.ticks)
		plt.grid(axis='y', ls='-', c='silver', lw=1, alpha=0.5)
		plt.ylim(0, 1.15*1e2)
		plt.xlabel(r'Wavelength [$\rm \AA$]', fontsize=20)
		plt.ylabel('Transmission [%]', fontsize=20)
		plt.minorticks_on()
		plt.tight_layout()
	

	def get_CCD_Hamamtsu_QE(self):
		# QE table of Gemini GMOS-N Hamamatsu CCD
		T_comp = Table.read('http://www.gemini.edu/sciops/instruments/gmos/gmos_n_ccd_hamamatsu_sc.txt', format='ascii.no_header', names=('wavelength', 'QE'))
		# T_comp['wavelength'] = T_comp['wavelength'].astype(float) * 1e-3
		T_comp['wavelength'] = T_comp['wavelength'].astype(float) * 1e1
		T_comp['wavelength'].unit = u.Angstrom
		T_comp['wavelength'].format = '8.4f'
		self.qe_hamamtsu = T_comp
		return T_comp


	def get_CMOS_IMX455_QE(self, path_table='../conf/QE.csv'):
		T_qe = ascii.read(path_table)
		T_qe['wavelength'] = T_qe['wave'] * 1e1
		T_qe['wavelength'].unit = u.Angstrom
		T_qe['wavelength'].format = '8.4f'
		# T_qe['QE'] = T_qe['qe']*1e-2

		#	Curve-fit interpolation
		# lam_optics = T_qe['wavelength'].value
		# total_optics = T_qe['QE']
		# popt, pcov = curve_fit(func_linear, lam_optics[3:10], total_optics[3:10])
		# xdata = np.arange(1000, 4000, 50)
		# ydata = func_linear(xdata, *popt)
		# nx = np.append(xdata, lam_optics)
		# ny = np.append(ydata, total_optics)
		# ny[ny<0] = 0

		nT_qe = Table()
		nT_qe['wavelength'] = T_qe['wavelength']
		nT_qe['wavelength'].unit = u.Angstrom
		nT_qe['wavelength'].format = '8.4f'
		# nT_qe['QE'] = ny

		self.qe_table = T_qe
		# self.qe = ny
		self.qe = T_qe['QE']
		return nT_qe

	def get_CMOS_IMX455_QE_forum(self, path_table='../util/sony.imx455.qhy600.dat'):
		T_qe = ascii.read(path_table)
		T_qe['wavelength'] = T_qe['lam'] * 1e1
		T_qe['wavelength'].unit = u.Angstrom
		T_qe['wavelength'].format = '8.4f'
		T_qe['QE'] = T_qe['qe']*1e-2

		#	Curve-fit interpolation
		lam_optics = T_qe['wavelength'].value
		total_optics = T_qe['QE']
		popt, pcov = curve_fit(func_linear, lam_optics[:2], total_optics[:2])
		xdata = np.arange(1000, 4000, 50)
		ydata = func_linear(xdata, *popt)
		nx = np.append(xdata, lam_optics)
		ny = np.append(ydata, total_optics)
		ny[ny<0] = 0

		nT_qe = Table()
		nT_qe['wavelength'] = nx
		nT_qe['wavelength'].unit = u.Angstrom
		nT_qe['wavelength'].format = '8.4f'
		nT_qe['QE'] = ny

		self.qe_table = nT_qe
		self.qe = ny
		return nT_qe

	def plot_QE(self):

		plt.figure(figsize=(12,4))
		plt.plot(self.qe_hamamtsu['wavelength'], self.qe_hamamtsu['QE']*1e2, 'o-', c='grey', lw=3, label='Hamamatsu CCD')
		x, y = self.qe_table['wavelength'], self.qe_table['QE']
		plt.plot(x[y>0], y[y>0]*1e2, 'o-', c='tomato', lw=3, label='Sony IMX455')
		#	7DS range
		xl, xr = self.bandcenter.min()-self.bandwidth, self.bandcenter.max()-self.bandwidth
		plt.axvspan(xl, xr, facecolor='silver', alpha=0.5, label='7DT range')

		#	Plot Setting
		# filterset = self.filterset
		# step = 500
		# xticks = np.arange(filterset['cwl'].min()-step, filterset['cwl'].max()+step, step)
		# plt.xticks(xticks)

		xl, xr = plt.xlim()
		plt.ylim([-10, 100.0])
		plt.xlabel(r'Wavelength [$\rm \AA$]', fontsize=20)
		plt.ylabel('QE', fontsize=20)
		plt.legend(loc='upper right', fontsize=14)
		plt.minorticks_on()
		plt.tight_layout()
	
	def get_optics(self, path_table='../util/optics.efficiency.dr350.dr500.csv'):
		optbl = ascii.read(path_table)
		self.optics_table = optbl

		#	Curve-fit interpolation
		lam_optics = optbl['nm']*10
		total_optics = optbl['total']
		popt, pcov = curve_fit(func_linear, lam_optics[:2], total_optics[:2])
		xdata = np.arange(1000, 4000, 1)
		ydata = func_linear(xdata, *popt)
		nx = np.append(xdata, lam_optics)
		ny = np.append(ydata, total_optics)

		lambda_mid = self.lam
		#	Consider only total optics efficiency
		eff_optics = optbl['total']
		#	efficiency of filter (varialble:LVF) is already considered
		eff_LVF = 1.00           # LVF peak transmission (filter top transmission)
		#	QE in detector (CMOS)
		# eff_fpa = np.interp(lambda_mid, self.qe['wavelength'], self.qe['QE'])
		# eff_opt = np.interp(lambda_mid, optbl['nm']*1e1, optbl['total'])
		eff_fpa = np.interp(lambda_mid, self.qe_table['wavelength'], self.qe_table['QE'])
		eff_opt = np.interp(lambda_mid, nx[ny>0], ny[ny>0])
		eff_total = eff_opt * eff_fpa

		self.optics_fpa = eff_fpa
		self.optics_opt = eff_opt
		self.optics_tot = eff_total
	
	def get_sky(self, path_table='../util/skytable.fits'):
		s = Table.read(path_table)

		lam = s['lam']*u.nm
		#	Photon Rate [ph/s/m2/micron/arcsec2]
		I = s['flux']*u.photon/u.second/(u.m**2)/u.um/(u.arcsec**2)
		flam = (const.h*const.c/lam)*I/(u.photon/u.arcsec**2)
		flam = flam.to(u.erg/u.second/(u.cm**2)/u.Angstrom)
		fnu = convert_flam2fnu(flam, lam)
		abmag = fnu.to(u.ABmag)

		s['flam'] = flam
		s['fnu'] = fnu
		s['abmag'] = abmag
		self.sky_table = s
		return s

	
	def plot_sky(self):
		s = self.sky_table
		filterset = self.filterset

		plt.figure(figsize=(12,4))
		plt.plot(s['lam']*1e1, s['trans']*1e2, c='dodgerblue', label='sky')
		#	Plot setting
		xl, xr = self.bandcenter.min()-self.bandwidth, self.bandcenter.max()-self.bandwidth
		plt.axvspan(xl, xr, facecolor='silver', alpha=0.5, label='7DT range')
		plt.xlabel('wavelength [$\mu m$]')
		plt.ylabel('Transmission [%]')
		plt.legend(loc='upper right', fontsize=14)
		plt.minorticks_on()
		plt.tight_layout()
		#	Plot Setting
		# step = 500
		# xticks = np.arange(filterset['cwl'].min()-step, filterset['cwl'].max()+step, step)
		# plt.grid(axis='y', ls='-', c='silver', lw=1, alpha=0.5)
		# plt.xticks(xticks)
		plt.xticks(self.ticks)
		plt.xlim([self.ticks[0], self.ticks[-1]])

	def smooth_sky(self, smooth_fractor=10):
		s = self.sky_table
		trans_smooth = gaussian_filter(s['trans'], smooth_fractor)
		self.smooth_sky = trans_smooth
		return trans_smooth

	def plot_sky_smooth(self):
		s = self.sky_table
		filterset = self.filterset

		plt.figure(figsize=(12,4))
		plt.plot(s['lam']*1e1, s['trans']*1e2, alpha=0.25, c='dodgerblue', label='Original')
		plt.xlabel('wavelength [$\AA$]')
		plt.ylabel('Transmission [%]')

		plt.legend(loc='upper right', fontsize=14)
		# plt.xlim(wmin-(fwhm*2), wmax+(fwhm*2))

		from scipy.ndimage import gaussian_filter
		trans_smooth = gaussian_filter(s['trans'], 10)
		plt.plot(s['lam']*1e1, trans_smooth*1e2, c='dodgerblue', label='Smooth')

		#	Plot Setting
		# step = 500
		# xticks = np.arange(filterset['cwl'].min()-step, filterset['cwl'].max()+step, step)
		# plt.grid(axis='y', ls='-', c='silver', lw=1, alpha=0.5)
		# plt.xticks(xticks)
		plt.xticks(self.ticks)
		plt.xlim([self.ticks[0], self.ticks[-1]])


		plt.legend(loc='lower center', fontsize=14)
		plt.minorticks_on()
		plt.tight_layout()

	def plot_sky_sb(self):
		s = self.sky_table
		filterset = self.filterset

		#
		wl_nm = s['lam']          # nm
		wl_um = wl_nm / 1e3       # micron
		wl_cm = wl_um / 1e4       # cm
		wl_am = wl_angstrom = wl_nm * 10  # angstrom
		nu = 3e18 / wl_angstrom   # Hz

		I_lambda = s['flux']      # [ph/s/m2/micron/arcsec2] photon reate
		f_lambda = I_lambda * (h*c/wl_cm) / (1e2**2) / (1e4)  # erg/s/cm2/A
		f_nu = f_lambda * wl_angstrom * (wl_cm/c) / (1e-23 * 1e-6)  # micro Jansky

		lam = s['lam']*u.nm
		#	Photon Rate [ph/s/m2/micron/arcsec2]
		I = s['flux']*u.photon/u.second/(u.m**2)/u.um/(u.arcsec**2)
		flam = (const.h*const.c/lam)*I/(u.photon/u.arcsec**2)
		flam = flam.to(u.erg/u.second/(u.cm**2)/u.Angstrom)

		fnu = convert_flam2fnu(flam, lam)
		abmag = fnu.to(u.ABmag)
		# abmag

		plt.figure(figsize=(12,4))
		# plt.plot(wl_angstrom[o], -2.5*np.log10(f_nu[o]*1e-6*1e-23)-48.60, alpha=0.75, c='dodgerblue')
		# plt.plot(wl_angstrom, -2.5*np.log10(f_nu*1e-6*1e-23)-48.60, alpha=0.75, c='dodgerblue')
		plt.plot(lam.to(u.Angstrom), abmag, alpha=0.75, c='dodgerblue')
		plt.axhline(y=21.8, color='tomato', ls='--', label='Dark night')
		#	Setting
		plt.xlabel(r'wavelength [$\AA$]')
		plt.ylabel(r'$SB_\nu$ [$mag/arcsec^2$]')
		plt.title('Sky brightness in AB mag')
		plt.ylim(24,14)
		xl, xr = self.bandcenter.min()-self.bandwidth, self.bandcenter.max()-self.bandwidth
		plt.axvspan(xl, xr, facecolor='silver', alpha=0.5, label='7DT range')
		plt.legend(loc='best', fontsize=14)
		plt.minorticks_on()
		plt.tight_layout()
		#	Plot Setting
		# step = 500
		# xticks = np.arange(filterset['cwl'].min()-step, filterset['cwl'].max()+step, step)
		# plt.grid(axis='y', ls='-', c='silver', lw=1, alpha=0.5)
		plt.xticks(self.ticks)
		plt.xlim([self.ticks[0], self.ticks[-1]])

	
	def calculate_response(self, group='sdt_default', textrotation=90):
		s = self.sky_table
		filterset = self.filterset
		trans_smooth = self.smooth_sky
		filterColors = self.color
		T_qe = self.qe_table
		_ = plt.figure(figsize=(12, 4))

		response = {
			'cwl': self.bandcenter,
			'wave': self.lam
			}

		for ii, (cwl, filtername) in enumerate(zip(filterset['cwl'], self.filterNameList)):
			
			wave_lvf = filterset['wave']
			resp_lvf = filterset[filtername]
			indx_resp = np.where(resp_lvf>resp_lvf.max()*1e-3)
			resp_sys = resp_lvf.copy()
			#	QE
			intp_qe = np.interp(wave_lvf, T_qe['wavelength'], T_qe['QE'])
			# print('T_qe', T_qe['wavelength'][0], T_qe['wavelength'][-1])
			# print('wave_lvf', wave_lvf[0], wave_lvf[-1])
			# print(resp_sys.shape, intp_qe.shape)
			try:
				# print(f"resp_sys: {len(resp_sys)}")
				# print(f"intp_qe: {len(intp_qe)}")
				resp_sys *= intp_qe
				#	Sky
				intp_trans = np.interp(wave_lvf, s['lam']*1e1, trans_smooth)
				resp_sys *= intp_trans 
				#	Optics
				intp_optics = np.interp(wave_lvf, self.lam, self.optics_opt)
				resp_sys *= intp_optics
				
				response.update({filtername: resp_sys})
				# resp_sys *= eff_mirrors * eff_optics
				# intp_optics = np.interp(wave_lvf, optbl['nm']*1e1, optbl['total'])
				
				if ii == len(self.filterNameList)-1:
					plt.plot(wave_lvf, intp_qe*1e2, label='CMOS QE', c='silver', lw=3)
					plt.plot(wave_lvf, intp_trans*1e2, label='Sky transmission', c='dodgerblue', alpha=0.5, lw=3)
					plt.plot(wave_lvf, intp_optics*1e2, label='Optics system', c='purple', alpha=0.5, lw=3)
					# plt.plot(wave_lvf[indx_resp], resp_lvf[indx_resp], label='Filter', c=filterColors[-ii-1], alpha=0.25,)
					plt.plot(wave_lvf[indx_resp], resp_lvf[indx_resp]*1e2, c=filterColors[-ii-1], alpha=0.25,)
					plt.plot(wave_lvf[indx_resp], resp_sys[indx_resp]*1e2, label='Total response', c=filterColors[-ii-1], alpha=1.00)
				else:
					plt.plot(wave_lvf[indx_resp], resp_lvf[indx_resp]*1e2, c=filterColors[-ii-1], alpha=0.25)
					plt.plot(wave_lvf[indx_resp], resp_sys[indx_resp]*1e2, c=filterColors[-ii-1], alpha=1.00)
				# plt.text(cwl, 1.1*resp_sys[indx_resp].max()*1e2, f"m{cwl/1e1:g}", horizontalalignment='center', size=10)
				if ii%2 == 0:
					plt.text(cwl, 1.1*resp_sys[indx_resp].max()*1e2, filtername, horizontalalignment='center', size=10, rotation=textrotation)
				elif ii%2 == 1:
					plt.text(cwl, 1.1*resp_sys[indx_resp].max()*1e2, filtername, horizontalalignment='center', size=10, rotation=textrotation)
			except Exception as e:
				print(filtername, e)
				pass
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.xlabel(r'Wavelength [$\AA$]')
		plt.ylabel('Response (%)')

		#	Plot Setting
		# step = 500
		# xticks = np.arange(filterset['cwl'].min()-step, filterset['cwl'].max()+step, step)
		# plt.xticks(xticks[1:])
		# plt.xlim([xticks[0], xticks[-1]])
		plt.xticks(self.ticks)
		plt.xlim([self.ticks[0], self.ticks[-1]])

		plt.ylim([0, 1*1e2])
		plt.legend(loc='center right', framealpha=1.0, fontsize=10)
		plt.minorticks_on()
		plt.tight_layout()
		self.response = response

		#	Filter response curve table
		tblist = []
		for ii, (cwl, filtername) in enumerate(zip(filterset['cwl'], self.filterNameList)):

			rsptbl = Table()
			rsptbl['index'] = [ii]*len(response['wave'])
			rsptbl['name'] = [filtername]*len(response['wave'])
			rsptbl['lam'] = filterset['wave'] * u.Angstrom # wavelength [AA]
			rsptbl['centerlam'] = cwl * u.Angstrom
			if (type(self.bandwidth) == int) | (type(self.bandwidth) == float):
				rsptbl['bandwidth'] = self.bandwidth * u.Angstrom
			else:
				rsptbl['bandwidth'] = self.bandwidth[-ii-1] * u.Angstrom
			rsptbl['response'] = response[filtername]
			rsptbl['lam'].format = '.1f'
			tblist.append(rsptbl)
			#	Make zero values for the both sides of the wavelength
			rsptbl['response'][0] = 0.0
			rsptbl['response'][-1] = 0.0

		totrsptbl = vstack(tblist)
		totrsptbl['group'] = group
		self.response_table = totrsptbl

		return totrsptbl

	def generate_custom_fiterset(self, combtbl, group='sdt_custom'):
		_ = plt.figure(figsize=(12, 4))

		filterColors = makeSpecColors(len(np.unique(combtbl['name'])))
		self.color = filterColors

		response = {
			'cwl': np.array([combtbl['centerlam'][combtbl['name'] == name][0] for name in np.unique(combtbl['name'])]),
			'wave': np.unique(combtbl['lam'])
			}

		self.filterNameList = np.unique(combtbl['name'])

		for ii, (cwl, filtername) in enumerate(zip(response['cwl'], self.filterNameList)):
			_ = combtbl[combtbl['name'] == filtername]
			wave_lvf = _['lam']
			resp_lvf = _['response']

			resp_sys = resp_lvf.copy()
			
			response.update({filtername: resp_sys})
			
			if ii == len(self.filterNameList)-1:
				plt.plot(wave_lvf, resp_sys*1e2, label='Total response', c=filterColors[-ii-1], alpha=1.00)
			else:
				plt.plot(wave_lvf, resp_sys*1e2, c=filterColors[-ii-1], alpha=1.00)
			plt.text(cwl, 1.1*resp_sys.max()*1e2, filtername, horizontalalignment='center', size=10)

		# plt.axhline(y=0, ls='-', color='k', lw=1)
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.xlabel(r'wavelength [$\AA$]')
		plt.ylabel('Response (%)')

		#	Plot Setting
		# step = 500
		# xticks = np.arange(response['cwl'].min()-step, response['cwl'].max()+step, step)
		# plt.xticks(xticks[1:])
		# plt.xlim([xticks[0], xticks[-1]])
		plt.xticks(self.ticks)
		plt.xlim([self.ticks[0], self.ticks[-1]])


		plt.ylim([0, 1*1e2])
		plt.legend(loc='center right', framealpha=1.0, fontsize=10)
		plt.minorticks_on()
		plt.tight_layout()
		self.response = response

		#	Filter response curve table'
		self.response_table = combtbl

		# return combtbl
		
	def get_phot_aperture(self, exptime, fwhm_seeing, optfactor=0.6731, verbose=True):
		"""_summary_
		Parameters
		----------
		exptime : float
			_description_
		fwhm_seeing : float
			_description_
		optfactor : float
			_description_

		Returns
		-------
		_type_
			_description_
		"""
		# exptime = 233000.	#	7DS Deep survey (IMS)
		# fwhm_seeing = 1.5     # [arcsec]
		fwhm_peeing = fwhm_seeing/self.theta_pixel

		# How many pixels does a point source occupy?
		# Effective number of pixels for a Gaussian PSF with fwhm_seeing
		# optfactor = 0.6731

		r_arcsec = optfactor*fwhm_seeing
		r_pix = optfactor*fwhm_seeing/self.theta_pixel

		aperture_diameter = 2*r_arcsec
		aperture_diameter_pix = 2*r_pix

		Npix_ptsrc = np.pi*(r_pix**2)
		Narcsec_ptsrc = np.pi*(r_arcsec**2)

		if verbose:
			print(f"Aperture radius   : {r_arcsec:1.3f} arcsec")
			print(f"Aperture radius   : {r_pix:1.3f} pix")
			print(f"fwhm_seeing       : {fwhm_seeing:1.3f} arcsec")
			print(f"exptime           : {exptime:g} second")
			print(f"SEEING*N Diameter : x{aperture_diameter/fwhm_seeing:1.3f}")
			print(f"Aperture Diameter : {aperture_diameter:1.3f} arcsec")
			print(f"Aperture Diameter : {aperture_diameter/self.theta_pixel:1.3f} pixel")
			print(f"Aperture Area     : {np.pi*(aperture_diameter/2)**2:1.3f} arcsec2")
			print(f"Aperture Area     : {Npix_ptsrc:1.3f} pixel2")

		self.exptime = exptime
		self.aperture_multiply_factor = optfactor
		self.seeing = fwhm_seeing
		self.seeing_pix = fwhm_peeing
		self.aperture_radius_arcsec = r_arcsec
		self.aperture_radius_pix = r_pix
		self.n_aperture_pix = Npix_ptsrc
		self.n_aperture_arcsec = Narcsec_ptsrc
		return (Npix_ptsrc, Narcsec_ptsrc)
	
	def get_depth_table(self, Nsigma=5):

		s = self.sky_table

		#	Empty Table
		unit_SB  = u.nW/(u.m)**2/u.sr
		unit_cntrate = u.electron / u.s

		T_sens = (QTable( 
					names=('band', 'wavelength', 'I_photo_sky', 'mag_sky', 'mag_pts'),
					dtype=(np.int16,float,float,float,float,) )
				)
		for key in T_sens.colnames:
			T_sens[key].info.format = '.4g'

		#	Iteration
		for ii, (cwl, filtername) in enumerate(zip(self.response['cwl'], self.filterNameList)):
			#	Sky brightness
			wave = s['lam']*1e1 # [nm] --> [AA]
			flux = s['fnu'] # [erg/s/cm2/Hz]
			#	Filter response
			wave_sys = self.response['wave']
			resp_sys = self.response[filtername]
			resp_lvf = self.filterset[filtername]

			# photon rate
			photon_rate = synth_phot(wave, flux, wave_sys, resp_sys, return_photonrate=True)
			# SB
			SB_sky = synth_phot(wave, flux, self.lam, resp_lvf)

			# photo-current or count rate
			I_photo = photon_rate * (np.pi/4*self.d_eff**2) * (self.theta_pixel**2)

			# noise in count per obs [e]. 
			Q_photo = (I_photo+self.I_dark)*self.exptime
			dQ_photo = np.sqrt(Q_photo)

			# noise in count rate [e/s]
			# read-noise (indistinguishable from signal) should be added 
			dI_photo = np.sqrt(dQ_photo**2 + self.dQ_RN**2)/self.exptime

			# surface brightness limit [one pixel]
			dSB_sky = (dI_photo/I_photo)*SB_sky
			mag_sky = -2.5*np.log10(Nsigma*dSB_sky) - 48.60

			# point source limit
			dFnu = np.sqrt(self.n_aperture_pix) * dSB_sky*(self.theta_pixel)**2
			# dFnu = Npix_ptsrc * dSB_sky*(theta_pixel)**2
			mag_pts = -2.5*np.log10(Nsigma*dFnu) - 48.60

			# Add data to the table
			T_sens.add_row([ii, cwl, I_photo, mag_sky, mag_pts]) 

		# Put units
		T_sens['wavelength'].unit = u.um
		T_sens['I_photo_sky'].unit = unit_cntrate
		T_sens['mag_sky'].unit = u.mag
		T_sens['mag_pts'].unit = u.mag

		#	Save summary result
		outbl = Table()
		outbl['index'] = np.arange(len(T_sens))
		# outbl['name'] = [f"m{lam.value/10:g}" for lam in T_sens['wavelength']]
		outbl['name'] = self.filterNameList
		outbl['center_wavelength'] = T_sens['wavelength'].value * u.Angstrom
		outbl['fwhm'] = self.bandwidth * u.Angstrom
		outbl['min_wavelength'] = outbl['center_wavelength'] - outbl['fwhm']/2
		outbl['max_wavelength'] = outbl['center_wavelength'] + outbl['fwhm']/2
		outbl['noise_countrate'] = T_sens['I_photo_sky']
		outbl['surface_brightness'] = T_sens['mag_sky']/(u.arcsec**2)
		outbl['5sigma_depth'] = T_sens['mag_pts']
		outbl['exptime'] = self.exptime*u.second
		outbl['seeing'] = self.seeing*u.arcsec
		outbl['x_aperture'] = self.aperture_multiply_factor
		outbl['aperture_radius'] = self.aperture_radius_arcsec
		outbl['aperture_radius_pix'] = self.aperture_radius_pix

		outbl['aperture_radius'].format = '1.3f'
		outbl['aperture_radius_pix'].format = '1.3f'

		self.depth_table_simple = T_sens
		self.depth_table = outbl 
		return outbl
	
	def plot_point_source_depth(self, figsize=(12, 4), text=True, legend=True):
		fig, ax = plt.subplots(1, figsize=figsize)

		T_sens = self.depth_table_simple

		for ii, filtername in enumerate(self.filterNameList):
			if ii == 0:
				ax.plot(T_sens['wavelength'][ii], T_sens['mag_pts'][ii], 'v', mec='k', c=self.color[-ii-1], ms=10, label=f'7DT ({self.exptime}s)')
			else:
				ax.plot(T_sens['wavelength'][ii], T_sens['mag_pts'][ii], 'v', mec='k', c=self.color[-ii-1], ms=10)
			if text:
				ax.text(T_sens['wavelength'][ii].value, T_sens['mag_pts'][ii].value+0.2, filtername, horizontalalignment='center', size=10)

		plt.xlabel(r'wavelength [$\AA$]', fontsize=20)
		plt.ylabel(r'Point source limits (5$\sigma$)', fontsize=20)

		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)

		yl, yu = plt.ylim()
		plt.ylim([yu+0.5, yl])

		# plt.xlim(wmin-(fwhm*2), wmax+(fwhm*2))
		if legend:
			plt.legend(loc='upper center', framealpha=1.0, fontsize=20)
		plt.minorticks_on()
		plt.tight_layout()

	def plot_point_source_depth_comp(self, spherex=True, ps1=True, smss=True,):

		#	SPHEREx
		sphxcsvlist = sorted(glob.glob('../util/SPHEREx*.csv'))
		total_table_list = []
		for ii, sphxcsv in enumerate(sphxcsvlist):
			#	Table info
			part = sphxcsv.split('_')
			obstype = part[1]
			ranges = part[2].split('.')[0]
			#	Table
			sphxtbl = ascii.read(sphxcsv)
			sphxtbl['wavelength'] <<= u.um
			sphxtbl['depth'] <<= u.ABmag
			sphxtbl['obstype'] = obstype
			sphxtbl['range'] = ranges
			#	Append to the list
			total_table_list.append(sphxtbl)
		#	Total table
		tsphxtbl = vstack(total_table_list)

		wavelength = np.unique(tsphxtbl['wavelength'].to(u.Angstrom))
		#	All-sky
		asphxtbl = tsphxtbl[tsphxtbl['obstype']=='allsky']
		asphxup = np.interp(wavelength, asphxtbl['wavelength'][asphxtbl['range']=='upper'], asphxtbl['depth'][asphxtbl['range']=='upper'])
		asphxlo = np.interp(wavelength, asphxtbl['wavelength'][asphxtbl['range']=='lower'], asphxtbl['depth'][asphxtbl['range']=='lower'])
		#	Deep
		dsphxtbl = tsphxtbl[tsphxtbl['obstype']=='deep']
		dsphxup = np.interp(wavelength, dsphxtbl['wavelength'][dsphxtbl['range']=='upper'], dsphxtbl['depth'][dsphxtbl['range']=='upper'])
		dsphxlo = np.interp(wavelength, dsphxtbl['wavelength'][dsphxtbl['range']=='lower'], dsphxtbl['depth'][dsphxtbl['range']=='lower'])

		#	Panstarrs
		# grizy < 22.0, 21.8, 21.5, 20.9, 19.7
		pstbl = Table()
		filterlist = ['g', 'r', 'i', 'z', 'y']
		depthlist = [22.0, 21.8, 21.5, 20.9, 19.7]
		lameff = [4810.16, 6155.47, 7503.03, 8668.36, 9613.60,]
		lamwidth = [1053.08, 1252.41, 1206.62, 997.72, 638.98,]
		pstbl['filter'] = filterlist
		pstbl['wavelength'] = lameff
		pstbl['eqwidth'] = lamwidth
		pstbl['depth'] = depthlist

		#	Skymapper
		smtbl = Table()
		filterlist = ['u', 'v', 'g', 'r', 'i', 'z']
		depthlist = [20.5, 20.5, 21.7, 21.7, 20.7, 19.7]
		lameff = [3500.22, 3878.68, 5016.05, 6076.85, 7732.83, 9120.25]
		lamwidth = [418.86, 319.06, 1450.60, 1414.05, 1246.20, 1158.57]
		smtbl['filter'] = filterlist
		smtbl['wavelength'] = lameff
		smtbl['eqwidth'] = lamwidth
		smtbl['depth'] = depthlist
		#	uv filer only
		smtbl = smtbl[(smtbl['filter']=='u') | (smtbl['filter']=='v')]

		fig, ax = plt.subplots(1, figsize=(10,6))

		#	7DT/7DS
		selfbl = self.depth_table
		# index	name	center_wavelength	fwhm	min_wavelength	max_wavelength	noise_countrate	surface_brightness	5sigma_depth
		ax.errorbar(selfbl['center_wavelength'], selfbl['5sigma_depth'], xerr=selfbl['fwhm'], markersize=10, mfc='none', marker='v', ls='', label=f'7DT ({self.exptime}s)')
		# ax.errorbar(selfbl['center_wavelength'], selfbl['5sigma_depth'], xerr=selfbl['fwhm'], markersize=10, mfc='none', marker='v', ls='', label='7DT (60s)')
		# ax.errorbar(lselfbl['center_wavelength'], lselfbl['5sigma_depth'], xerr=lselfbl['fwhm'], markersize=10, mfc='none', marker='v', ls='', label='7DT (180s)')
		# ax.errorbar(wselfbl['center_wavelength'], wselfbl['5sigma_depth'], xerr=wselfbl['fwhm'], markersize=10, marker='v', mec='k', ls='', label='7DS-Wide')
		# ax.errorbar(dselfbl['center_wavelength'], dselfbl['5sigma_depth'], xerr=dselfbl['fwhm'], markersize=10, marker='v', mec='k', ls='', label='7DS-IMS')

		#	SPHEREx
		if spherex:
			##	SPHEREx - allsky
			ax.plot(asphxtbl['wavelength'].to(u.Angstrom), asphxtbl['depth'], mfc='k', mec='r', marker='o', ls='', label='SPHEREx All-sky')
			ax.fill_between(wavelength.value, asphxlo, asphxup, facecolor='red', alpha=0.25)

			#	SPHEREx - Deep
			ax.plot(dsphxtbl['wavelength'].to(u.Angstrom), dsphxtbl['depth'], mfc='k', mec='orange', marker='.', ls='', label='SPHEREx Deep')
			ax.fill_between(wavelength.value, dsphxlo, dsphxup, facecolor='orange', alpha=0.25)

		#	SkyMapper
		if smss:
			ax.errorbar(smtbl['wavelength'], smtbl['depth'], xerr=smtbl['eqwidth'], markersize=10, mfc='none', marker='s', ls='', label='SkyMapper (uv)')

		#	PanSTARRs
		if ps1:
			ax.errorbar(pstbl['wavelength'], pstbl['depth'], xerr=pstbl['eqwidth'], markersize=10, mfc='none', marker='s', ls='', label='PS1')

		#	Setting
		ax.set_xscale('log')

		xl, xr = ax.set_xlim()
		yl, yu = ax.set_ylim()
		yl = 17.

		ax.set_xlim([3000, xr])
		ax.set_ylim([yu, yl])

		# ax.legend(loc='lower right', fontsize=12, ncol=2)
		ax.legend(loc='best', fontsize=12, ncol=2, framealpha=1.0)
		ax.tick_params(labelsize=14)

		if spherex == True:
			xticks = [4000, 6000, 9000, 15000, 20000, 30000, 40000, 50000]
			ax.set_xticks(xticks, xticks)
		ax.set_xlabel(r'Wavelength [$\rm \AA$]', fontsize=20)
		# ax.set_ylabel(r'Magnitude AB [$\rm 5\sigma$]', fontsize=20)
		ax.set_ylabel(r'$\rm 5\sigma$ Depth [AB]', fontsize=20)
		#
		plt.tight_layout()
		plt.minorticks_on()
		plt.grid('both', ls='--', c='silver', alpha=0.5)

	def calculate_pointsource_snr(self, mag, filtername, exptime):
		s = self.sky_table
		# mag_src = 17.5
		# exptime = 180
		# filtername = 'm6000'

		#	Sky brightness
		wave = s['lam']*1e1 # [nm] --> [AA]
		flux = s['fnu'] # [erg/s/cm2/Hz]

		wave_sys = self.response['wave']
		resp_sys = self.response[filtername]

		Naper = self.n_aperture_pix 

		flux = s['fnu']

		flux_src = flux*0 + 10**(-0.4*(mag + 48.6))	# [erg/s/cm2/Hz]
		flux_sky = flux*(1e-23*1e-6) # [erg/s/cm2/Hz/arcsec2]

		photon_rate_src = synth_phot(wave, flux_src, wave_sys, resp_sys, return_photonrate=True)  # [ph/s/cm2]
		photon_rate_sky = synth_phot(wave, flux_sky, wave_sys, resp_sys, return_photonrate=True)  # [ph/s/cm2/arcsec2]

		I_photo_src = photon_rate_src * (np.pi/4*self.d_eff**2)                     # [e/s] per aperture (no aperture loss)
		I_photo_sky = photon_rate_sky * (np.pi/4*self.d_eff**2) * (self.theta_pixel**2)  # [e/s] per pixel 

		Q_photo_src = I_photo_src * exptime
		Q_photo_sky = I_photo_sky * exptime
		Q_photo_dark = self.I_dark * exptime

		snr = Q_photo_src / np.sqrt(Q_photo_src + Naper*Q_photo_sky + Naper*Q_photo_dark + Naper*self.dQ_RN**2)
		return max(snr, 1e-30)
	
	def calculate_magobs(self, mag, filtername, exptime, zperr=0.01, n=10):

		#	Signal-to-noise ratio (SNR)
		snr = self.calculate_pointsource_snr(mag=mag, filtername=filtername, exptime=exptime)
		#	SNR --> mag error
		merr0 = convert_snr2magerr(snr)
		#	Random obs points
		m = get_random_point(mag, merr0, n=n)
		#	Measured error
		merr = np.sqrt(merr0**2+zperr**2)

		return (snr, m, merr)
	
	def get_speclite(self):
		rsptbl = self.response_table
		filterlist = self.filterNameList
		for filte in filterlist:
			#	Meta
			metadict = dict(
				group_name='sevendt',
				band_name=filte,
				exptime=self.exptime,
			)

			#	Filter Table
			fltbl = rsptbl[rsptbl['name']==filte]
			_ = speclite.filters.FilterResponse(
				wavelength = fltbl['lam'],
				response = fltbl['response'],
				meta=metadict,
			)

		#	New name for speclite class
		speclite_filterlist = [f"sevendt-{filte}" for filte in filterlist]

		#	Medium filters
		bands = speclite.filters.load_filters(*speclite_filterlist)
		self.speclite_bands = bands
		return bands
	
	def get_synphot(self, spflam, splam, figure=True):
		mags = self.speclite_bands.get_ab_magnitudes(spflam, splam)
		synmag = np.array([mags[filte][0] for filte in mags.keys()])
		synlam = self.speclite_bands.effective_wavelengths
		spfnu = convert_flam2fnu(spflam, splam)
		spabmag = spfnu.to(u.ABmag)

		if figure:
			plt.plot(splam, spabmag, c='grey', lw=3, alpha=0.75, label='spectrum')
			plt.plot(synlam, synmag, 'o', markersize=10, c='tomato', alpha=0.75, label=f'{len(self.filterNameList)} 7DT')

			#	x, y ranges
			yu = np.max(synmag)+0.5
			yl = np.min(synmag)-0.5
			plt.ylim(yl, yu)

			xl = np.min(synlam.value)-500
			xr = np.max(synlam.value)+500
			plt.xlim([xl, xr])
		
			plt.xlabel(r'Wavelength ($\rm \AA$)', fontsize=20)
			plt.ylabel(r'Luminosity (AB mag)', fontsize=20)
			plt.grid('both', ls='--', c='silver', alpha=0.5)
			plt.legend()
			plt.tight_layout()

		return synmag
	
	def get_synphot2obs(self, spflam, splam, z, z0=0, figure=False):

		#	Absolute magnitude
		#	pad
		spflam, splam = self.speclite_bands.pad_spectrum(spflam, splam)
	
		#	Handle NaN values
		indx_nan = np.where(np.isnan(spflam))
		indx_not_nan = np.where(~np.isnan(spflam))
		if len(indx_nan[0]) > 0:
			for nindx in indx_nan[0]:
				if nindx == 0:
					spflam[nindx] = spflam[~np.isnan(spflam)][0]
				elif nindx == len(spflam):
					spflam[nindx] = spflam[~np.isnan(spflam)][-1]
				elif (nindx != 0) & (nindx != len(spflam)) & (nindx-1 not in indx_nan[0]) & (nindx+1 not in indx_nan[0]):
					leftone = spflam[nindx-1]
					rightone = spflam[nindx+1]
					spflam[nindx] = np.mean([leftone, rightone])
				else:
					absdiff = np.abs(indx_not_nan[0]-nindx)
					closest_indx = absdiff.min()
					closest_spflam = spflam[indx_not_nan[0][closest_indx]]
					spflam[nindx] = closest_spflam

		mags = self.speclite_bands.get_ab_magnitudes(spflam, splam)
		synabsmag = np.array([mags[filte][0] for filte in mags.keys()])
		synlam = self.speclite_bands.effective_wavelengths
		# spabsfnu = convert_flam2fnu(spflam, splam)
		# spmag = spfnu.to(u.ABmag)

		#	Shifted & Scaled spectrum
		if z!=None:
			(zspappflam, zsplam) = apply_redshift_on_spectrum(spflam, splam, z, z0)
		else:
			zspappflam, zsplam = spflam, splam
		#	pad
		zspappflam, zsplam = self.speclite_bands.pad_spectrum(zspappflam, zsplam)
		# print(zspappflam, zsplam)
		# zspappflam = fill_nan_with_interpolation(zspappflam)

		#	Handle NaN values
		indx_nan = np.where(np.isnan(zspappflam))
		indx_not_nan = np.where(~np.isnan(zspappflam))
		if len(indx_nan[0]) > 0:
			for nindx in indx_nan[0]:
				if nindx == 0:
					zspappflam[nindx] = zspappflam[~np.isnan(zspappflam)][0]
				elif nindx == len(zspappflam):
					zspappflam[nindx] = zspappflam[~np.isnan(zspappflam)][-1]
				elif (nindx != 0) & (nindx != len(zspappflam)) & (nindx-1 not in indx_nan[0]) & (nindx+1 not in indx_nan[0]):
					leftone = zspappflam[nindx-1]
					if nindx < len(zspappflam)-1:
						# print(nindx, len(zspappflam))
						rightone = zspappflam[nindx+1]
					else:
						rightone = leftone
					zspappflam[nindx] = np.mean([leftone.value, rightone.value])*leftone.unit
				else:
					absdiff = np.abs(indx_not_nan[0]-nindx)
					closest_indx = absdiff.min()
					closest_zspappflam = zspappflam[indx_not_nan[0][closest_indx]]
					zspappflam[nindx] = closest_zspappflam

		mags = self.speclite_bands.get_ab_magnitudes(zspappflam, zsplam)
		synappmag = np.array([mags[filte][0] for filte in mags.keys()])
		synappmag[np.isinf(synappmag)] = 999.
		synappmag[np.isnan(synappmag)] = 999.
		# synlam = self.speclite_bands.effective_wavelengths
		# spappfnu = convert_flam2fnu(spflam, splam)
		# spmag = spfnu.to(u.ABmag)

		rsptbl = self.response_table

		outbl = Table()
		outbl['filter'] = self.filterNameList
		outbl['lam'] = synlam
		outbl['bandwidth'] = np.array([rsptbl['bandwidth'][rsptbl['name']==name][0] for name in self.filterNameList])
		outbl['magabs'] = synabsmag
		outbl['snr'] = 0.0
		outbl['magapp'] = 0.0
		outbl['magobs'] = 0.0
		outbl['magerr'] = 0.0
		outbl['fnuobs'] = 0.0
		outbl['fnuerr'] = 0.0
		# outbl['flamobs'] = 0.0
		# outbl['flamerr'] = 0.0

		for ii, (mag, filtername) in enumerate(zip(synappmag, self.filterNameList)):
			# print(synappmag)
			# print(mag, filtername)
			(snr, m, merr) = self.calculate_magobs(mag, filtername, self.exptime)
			#	fnu [uJy]
			fnuobs = (m*u.ABmag).to(u.uJy)
			fnuerr = fnuobs/snr
			# #	flam [erg/s/cm2/AA]
			# flamobs = convert_fnu2flam(fnuobs, splam)
			# flamerr = flamobs/snr
			
			#	To the table
			outbl['snr'][ii] = snr
			outbl['magapp'][ii] = mag
			outbl['magobs'][ii] = m
			outbl['magerr'][ii] = merr
			outbl['fnuobs'][ii] = fnuobs.value
			outbl['fnuerr'][ii] = fnuerr.value
			# outbl['flamobs'][ii] = flamobs.value
			# outbl['flamerr'][ii] = flamerr.value
		#	magapp --> fnu
		outbl['fnu'] = (outbl['magapp']*u.ABmag).to(u.uJy).value
		#	Format
		for key in outbl.keys():
			if key not in ['filter']:
				outbl[key].format = '1.3f'
		#	Unit
		outbl['magapp'].unit = u.ABmag
		outbl['magobs'].unit = u.ABmag
		outbl['magerr'].unit = u.ABmag
		outbl['fnu'].unit = u.uJy
		outbl['fnuobs'].unit = u.uJy
		outbl['fnuerr'].unit = u.uJy

		#	Figure
		if figure:
			#	flam --> fnu --> magnitude
			zspappfnu = convert_flam2fnu(zspappflam, zsplam)
			zspappmag = zspappfnu.to(u.ABmag)

			plt.plot(zsplam, zspappmag, zorder=0, c='grey', lw=3, alpha=0.5, label='spectrum')
			# plt.plot(outbl['lam'], outbl['magapp'], lw=3, zorder=0, marker='o', ls='none', c='tomato', alpha=0.75, label='syn.phot')
			plt.plot(outbl['lam'], outbl['magapp'], lw=3, zorder=0, marker='.', ls='none', c='tomato', alpha=0.75, label='syn.phot')
			plt.errorbar(outbl['lam'], outbl['magobs'], xerr=outbl['bandwidth']/2, yerr=outbl['magerr'], marker='none', c='k', zorder=0, alpha=0.5, ls='none')
			plt.scatter(outbl['lam'], outbl['magobs'], c=outbl['snr'], marker='s', edgecolors='k', s=50, label='obs')
			cbar = plt.colorbar()
			cbar.set_label('SNR', fontsize=12)

			plt.xlim([self.bandcenter[0]-250, self.bandcenter[-1]+250])
			plt.ylim([outbl['magobs'].max()+0.25, outbl['magobs'].min()-0.25])
			plt.legend(loc='lower center', fontsize=10, ncol=3)
			plt.xlabel(r'Wavelength [$\rm \AA$]', fontsize=12)
			plt.ylabel(r'Brightness [AB mag]', fontsize=12)
			plt.tight_layout()

		return outbl


	def get_sdss_filterset(self):
		#	Filter Table List
		sdsslist = sorted(glob.glob('../util/Chroma*eff.txt'))
		#	Re-arange the order (ugriz)
		sdsslist = [sdsslist[3], sdsslist[0], sdsslist[2], sdsslist[1], sdsslist[4]]

		#	Create filter_set definition
		filter_set = {
			'cwl': 0,
			'wave': 0
			}

		cwl = []
		filterNameList = []
		bandwidth = []
		for ss, sdss in enumerate(sdsslist):
			#	Read SDSS filter transmission table
			intbl = ascii.read(sdss)
			#	Get filter name
			filte = sdss.split('_')[1][0]

			if len(intbl) >= 1801:
				#	[nm] --> [Angstrom]
				intbl['lam'] = intbl['col1']*10
				#	Wavelength resolution
				lamres = np.mean(intbl['lam'][1:] - intbl['lam'][:-1])
				#	Wavelength min & max
				lammin = intbl['lam'].min()
				lammax = intbl['lam'].max()
			else:
				reftbl = ascii.read(sdsslist[0])
				rsp = np.interp(reftbl['col1'], intbl['col1'], intbl['col2'])
				intbl = Table()
				#	[nm] --> [Angstrom]
				intbl['lam'] = reftbl['col1']*10
				intbl['col2'] = rsp
				#	Wavelength resolution
				lamres = np.mean(intbl['lam'][1:] - intbl['lam'][:-1])
				#	Wavelength min & max
				lammin = intbl['lam'].min()
				lammax = intbl['lam'].max()

			#	Effective wavelength
			# indx_eff = np.where(intbl['col2']>0.5)
			# cwl.append(np.sum(intbl['lam'][indx_eff]*intbl['col2'][indx_eff]*lamres)/np.sum(intbl['col2'][indx_eff]*lamres))
			cwl.append(np.sum(intbl['lam']*intbl['col2']*lamres)/np.sum(intbl['col2']*lamres))
			filterNameList.append(filte)
			filter_set.update({filte: intbl['col2']})

			#	Half Max
			hm = intbl['col2'].max()/2
			#	Left, Right index
			indx_left = np.where(intbl['lam']<np.sum(intbl['lam']*intbl['col2']*lamres)/np.sum(intbl['col2']*lamres))
			indx_right = np.where(intbl['lam']>np.sum(intbl['lam']*intbl['col2']*lamres)/np.sum(intbl['col2']*lamres))
			#	FWHM
			fwhm = np.interp(hm, intbl['col2'][indx_right], intbl['lam'][indx_right],) - np.interp(hm, intbl['col2'][indx_left], intbl['lam'][indx_left],)
			bandwidth.append(fwhm/2)
		# bandwidth = np.array(bandwidth)
		#	Forced value
		#	https://www.researchgate.net/figure/SDSS-FILTER-CHARACTERISTICS-AND-PHOTOMETRIC-SENSITIVITY-14-AIR-MASSES_tbl2_2655119
		bandwidth = np.array(
			[
				560,
				1377,
				1371,
				1510,
				940,
			]
		)
		cwl = np.array(cwl)

		step = 500
		ticks = np.arange(round(intbl['lam'].min(), -3)-step, round(intbl['lam'].max(), -3)+step, step)

		filter_set['cwl'] = cwl
		filter_set['wave'] = np.array(intbl['lam'])
		self.filterset = filter_set
		self.filterNameList = filterNameList
		self.lam = intbl['lam']
		self.lammin = lammin
		self.lammax = lammax
		self.lamres = lamres
		self.bandcenter = cwl
		self.bandstep = 0
		self.bandwidth = bandwidth
		self.color = makeSpecColors(len(cwl))
		self.ticks = ticks
		return filter_set

	def get_edmund_25nm_filterset(self, path_filter='../data/edmund_med_band/*_25.csv'):
		# path_filter = '../data/edmund_med_band/*_25.csv'
		infilterlist = sorted(glob.glob(path_filter))
		print(f"{len(infilterlist)} Filters found in {path_filter}")

		#	Create filter_set definition
		filter_set = {
			'cwl': 0,
			'wave': 0
			}

		cwl = []
		filterNameList = []
		bandwidth = []
		################################################################
		ff = 0
		filte = infilterlist[ff]
		for ff, filte in enumerate(infilterlist):
			part = os.path.basename(filte).replace('.csv', '').split('_')
			center_wavelength = float(part[0])*1e1
			bw = int(part[1])*1e1
			# print(filte, center_wavelength, bw)

			_intbl = Table.read(filte)
			_intbl = _intbl[_intbl['wavelength']<1000]
			intbl = Table()
			#	[nm] --> [Angstrom]
			intbl['lam'] = _intbl['wavelength']*1e1
			#	[%] --> ratio
			intbl['col2'] = _intbl['transmission']*1e-2
			#	Wavelength resolution
			lamres = np.mean(intbl['lam'][1:] - intbl['lam'][:-1])
			#	Wavelength min & max
			lammin = intbl['lam'].min()
			lammax = intbl['lam'].max()

			# cwl.append(np.sum(intbl['lam']*intbl['col2']*lamres)/np.sum(intbl['col2']*lamres))
			cwl.append(center_wavelength)
			filterNameList.append(f"m{center_wavelength:g}")
			filter_set.update({f"m{center_wavelength:g}": intbl['col2']})

		cwl = np.array(cwl)

		step = 500
		ticks = np.arange(round(intbl['lam'].min(), -3)-step, round(intbl['lam'].max(), -3)+step, step)

		filter_set['cwl'] = cwl
		filter_set['wave'] = np.array(intbl['lam'].data)

		self.filterset = filter_set
		self.filterNameList = filterNameList
		self.lam = intbl['lam'].data
		self.lammin = lammin
		self.lammax = lammax
		# self.lammin = 3500
		# self.lammax = 9250
		self.lamres = lamres
		self.bandcenter = cwl
		self.bandstep = 12.5
		self.bandwidth = bw
		self.color = makeSpecColors(len(cwl))
		self.ticks = ticks
		return filter_set


	def get_edmund_50nm_filterset(self, path_filter='../data/edmund_med_band/*_50.csv'):
		# path_filter = '../data/edmund_med_band/*_25.csv'
		infilterlist = sorted(glob.glob(path_filter))
		print(f"{len(infilterlist)} Filters found in {path_filter}")

		#	Create filter_set definition
		filter_set = {
			'cwl': 0,
			'wave': 0
			}

		cwl = []
		filterNameList = []
		bandwidth = []
		################################################################
		ff = 0
		filte = infilterlist[ff]
		for ff, filte in enumerate(infilterlist):
			part = os.path.basename(filte).replace('.csv', '').split('_')
			center_wavelength = float(part[0])*1e1
			bw = int(part[1])*1e1
			# print(filte, center_wavelength, bw)

			_intbl = Table.read(filte)
			_intbl = _intbl[_intbl['wavelength']<1000]
			intbl = Table()
			#	[nm] --> [Angstrom]
			intbl['lam'] = _intbl['wavelength']*1e1
			#	[%] --> ratio
			intbl['col2'] = _intbl['transmission']*1e-2
			#	Wavelength resolution
			lamres = np.mean(intbl['lam'][1:] - intbl['lam'][:-1])
			#	Wavelength min & max
			lammin = intbl['lam'].min()
			lammax = intbl['lam'].max()

			# cwl.append(np.sum(intbl['lam']*intbl['col2']*lamres)/np.sum(intbl['col2']*lamres))
			cwl.append(center_wavelength)
			filterNameList.append(f"m{center_wavelength:g}")
			filter_set.update({f"m{center_wavelength:g}": intbl['col2']})

		cwl = np.array(cwl)

		step = 500
		ticks = np.arange(round(intbl['lam'].min(), -3)-step, round(intbl['lam'].max(), -3)+step, step)

		filter_set['cwl'] = cwl
		filter_set['wave'] = np.array(intbl['lam'].data)

		self.filterset = filter_set
		self.filterNameList = filterNameList
		self.lam = intbl['lam'].data
		self.lammin = lammin
		self.lammax = lammax
		# self.lammin = 3500
		# self.lammax = 9250
		self.lamres = lamres
		self.bandcenter = cwl
		self.bandstep = 25
		self.bandwidth = bw
		self.color = makeSpecColors(len(cwl))
		self.ticks = ticks
		return filter_set
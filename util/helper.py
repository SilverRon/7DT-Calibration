#	22.09.25
#	Gregory S.H. Paek
#================================================================
#	Library
#----------------------------------------------------------------
from astropy.io import ascii
import os, glob
import numpy as np
import speclite.filters
from astropy import units as u
from astropy.table import Table, vstack, hstack
from astropy import constants as const
import warnings
# from numba import *
warnings.filterwarnings('ignore')
#----------------------------------------------------------------
def get_bandwidth_table():
	#	Bandwidth Table
	bdwtbl = Table()
	grouplist = ['Med']*20+['SDSS']*4+['Johnson Cousin']*4
	filterlist = [f"m{int(filte)}" for filte in np.arange(400, 875+25, 25)]+['g', 'i', 'r', 'u']+['B', 'V', 'R', 'I']
	bandwidths = [250]*20+[1370, 1530, 1370, 500]+[781, 991, 1066, 2892]
	bdwtbl['group'] = grouplist
	bdwtbl['filter'] = filterlist
	bdwtbl['bandwidth'] = bandwidths
	return bdwtbl
#----------------------------------------------------------------
def get_lsst_bandwidth():
	"""
	filterlist : ugrizy
	"""
	return np.array([494.43, 1419.37, 1327.32, 1244.00, 1024.11, 930.04])*u.Angstrom
#----------------------------------------------------------------
def makeSpecColors(n, palette='Spectral'):
	#	Color palette
	import seaborn as sns
	palette = sns.color_palette(palette, as_cmap=True,)
	palette.reversed

	clist_ = [palette(i) for i in range(palette.N)]
	cstep = int(len(clist_)/n)
	clist = [clist_[i*cstep] for i in range(n)]
	return clist
#----------------------------------------------------------------
def convert_lam2nu(lam):
	nu = (const.c/(lam)).to(u.Hz)
	return nu
#----------------------------------------------------------------
def convert_fnu2flam(fnu, lam):
	flam = (fnu*const.c/(lam**2)).to((u.erg/((u.cm**2)*u.second*u.Angstrom)))
	return flam
#----------------------------------------------------------------
def convert_flam2fnu(flam, lam):
	fnu = (flam*lam**2/const.c).to((u.erg/((u.cm**2)*u.second*u.Hz)))
	return fnu
#----------------------------------------------------------------
def convert_app2abs(m, d):
	M = m - (5*np.log10(d)-5)
	return M
#----------------------------------------------------------------
def convert_abs2app(M, d):
	m = M + (5*np.log10(d)-5)
	return m
#----------------------------------------------------------------
def get_speclite_med(filterlist='all'):
	rsptbl = ascii.read('../3.table/7dt.filter.response.realistic_optics.ecsv')
	if filterlist=='all':
		filterlist = np.unique(rsptbl['name'])
	else:
		pass

	for filte in filterlist:
		#	Filter Table
		fltbl = rsptbl[rsptbl['name']==filte]

		_ = speclite.filters.FilterResponse(
			wavelength = fltbl['lam'],
			response = fltbl['response'], meta=dict(group_name='med', band_name=filte)
		)

	#	New name for speclite class
	mfilterlist = [f"med-{filte}" for filte in filterlist]

	#	Medium filters
	meds = speclite.filters.load_filters(*mfilterlist)
	return meds
#----------------------------------------------------------------
def get_speclite_sdss(filterlist=['u', 'g', 'r', 'i']):
	rsptbl = ascii.read('../3.table/sdss.filter.response.realistic_optics.ecsv')

	# filterlist = np.unique(rsptbl['name'])

	# for filte in ['u', 'g', 'r', 'i']:
	for filte in filterlist:
		#	Filter Table
		fltbl = rsptbl[rsptbl['name']==filte]

		_ = speclite.filters.FilterResponse(
			wavelength = fltbl['lam'],
			response = fltbl['response'], meta=dict(group_name='broad', band_name=filte)
		)

	#	New name for speclite class
	bfilterlist = [f"broad-{filte}" for filte in filterlist]

	#	Broad filters
	broads = speclite.filters.load_filters(*bfilterlist)
	return broads
#----------------------------------------------------------------
def get_speclite_jc(filterlist):
	rsptbl = ascii.read('../3.table/kmtnet/kmtnet_filter.csv')

	# filterlist = np.unique(rsptbl.keys()[1:])
	# filterlist = ['B', 'V', 'R', 'I']
	for filte in filterlist:

		rsp = rsptbl[filte]
		rsp = rsp*1e-2 # [%] --> [0.0-1.0]
		rsp[0] = 0.0
		rsp[-1] = 0.0
		rsp[rsp<0] = 0.0

		index = np.where(
			rsp>1e-2	
		)
		
		rsp0 = rsp[index]
		rsp0[0] = 0.0
		rsp0[-1] = 0.0

		#	Filter Table
		_ = speclite.filters.FilterResponse(
			wavelength = rsptbl['wavelength'][index]*u.nm,
			response = rsp0, meta=dict(group_name='kmtnet', band_name=filte)
		)

	#	New name for speclite class
	kfilterlist = [f"kmtnet-{filte}" for filte in filterlist]

	#	KMTNet filters
	kmtns = speclite.filters.load_filters(*kfilterlist)
	return kmtns
#----------------------------------------------------------------
def get_speclite_ztf():
	rsptbl = ascii.read('../3.table/sdss.filter.response.realistic_optics.ecsv')

	# filterlist = np.unique(rsptbl['name'])
	filterlist = ['g', 'r', 'i']

	for filte in filterlist:
	# for filte in ['g', 'r', 'i']:
		#	Filter Table
		fltbl = rsptbl[rsptbl['name']==filte]

		_ = speclite.filters.FilterResponse(
			wavelength = fltbl['lam'],
			response = fltbl['response'], meta=dict(group_name='broad', band_name=filte)
		)

	#	New name for speclite class
	bfilterlist = [f"broad-{filte}" for filte in filterlist]

	#	Broad filters
	broads = speclite.filters.load_filters(*bfilterlist)
	return broads

#----------------------------------------------------------------
def get_speclite_lsst():
	lsst = speclite.filters.load_filters('lsst2016-*')
	# speclite.filters.plot_filters(lsst, wavelength_limits=(3000, 11000), legend_loc='upper left')
	return lsst
#----------------------------------------------------------------
def get_wollaeger():
	kncbtbl = Table.read(f"../3.table/kn_cube.lite.spectrum.summary.fits")
	return kncbtbl
#----------------------------------------------------------------
def get_7dt_depth(exptime=180):
	dptbl = ascii.read(f"../3.table/7dt.filter.realistic_optics.{exptime}s.summary.ecsv")
	return dptbl
#----------------------------------------------------------------
def get_7dt_broadband_depth(exptime=180):
	dptbl = ascii.read(f"../3.table/sdss.filter.realistic_optics.{exptime}s.summary.ecsv")
	return dptbl
#----------------------------------------------------------------
def get_ztf_depth(filte):
	if filte == 'g':
		depth = 20.8
	elif filte == 'r':
		depth = 20.6
	elif filte == 'i':
		depth = 19.9
	else:
		depth = None
	return depth
#----------------------------------------------------------------
def get_decam_depth(filte):
	if filte == 'i':
		depth = 22.5
	elif filte == 'z':
		depth = 21.8
	else:
		depth = None
	return depth
#----------------------------------------------------------------
def get_lsst_depth(filte):
	if filte == 'u':
		depth = 23.6
	elif filte == 'g':
		depth = 24.7
	elif filte == 'r':
		depth = 24.2
	elif filte == 'i':
		depth = 23.8
	elif filte == 'z':
		depth = 23.2
	elif filte == 'y':
		depth = 22.3
	else:
		depth = None
	return depth
#----------------------------------------------------------------
def get_kmtnet_depth(filte, obs='KMTNet', exptime=120):
	'''
	exptime = 480 [s] : default value for calculating the depth
	'''
	offset = 2.5*np.log10(exptime/480)
	dptbl = Table.read('../3.table/kmtnet/kmtnet.depth.fits')
	obstbl = dptbl[dptbl['obs']==obs]
	try:
		return obstbl['ul5_med'][obstbl['filter']==filte].item()+offset
	except:
		return None
#----------------------------------------------------------------
#	Fitting tools
def func(x, a):
	return a*x

def calc_chisquare(obs, exp):
	return np.sum((obs-exp)**2/exp)
#----------------------------------------------------------------
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
#----------------------------------------------------------------
def calc_chisq(r, sigma):
	chisq = sum((r / sigma)**2)
	return chisq

def calc_chisqdof(chisq, dof):
	return chisq/dof
#----------------------------------------------------------------
# @jit(nopython=True, parallel=True)
def calc_snr(m, ul, sigma=5):
	snr = sigma*10**((ul-m)/5)
	return snr
#----------------------------------------------------------------
def convert_snr2magerr(snr):
	merr = 2.5*np.log10(1+1/snr)
	return merr
#----------------------------------------------------------------
def calc_GaussianFraction(seeing, optfactor=0.6731, path_plot=None):
	import numpy as np
	import matplotlib.pyplot as plt
	from scipy.special import erf

	# seeing, optfactor= 1.5, 0.6731

	mu = 0.0
	# sigma = fwhm_seeing/2.355
	fwhm2sigma = seeing*2.355
	# optfactor = 0.6731
	sigma = fwhm2sigma*optfactor

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
	# print(np.sum(y), np.sum(yaper), frac)

	if path_plot != None:
		plt.plot(x, y, alpha=0.5, label=f'PDF of N(0, {sigma:1.3f})', lw=5)
		plt.plot(xaper, yaper, alpha=1.0, label=f'Aperture ({frac*1e2:.1f}%)', lw=5,)
		plt.xlabel('x', fontsize=20)
		plt.ylabel('f(x)', fontsize=20)
		plt.legend(loc='lower center', fontsize=14)
		# plt.show()
		plt.savefig(path_plot, overwrite=True)
	else:
		pass

	return frac

def add_noise(mu, sigma, nseed, n=10, path_plot=None):
	"""
	mu, sigma = 17.5, 0.1
	n = 10
	"""
	from scipy.stats import norm
	import numpy as np
	
	try:
		x = np.arange(mu-sigma*n, mu+sigma*n, sigma*1e-3)
		y = norm(mu, sigma).pdf(x)

		if path_plot != None:
			resultlist = []
			for i in range(10000):
				xobs = np.random.choice(x, p=y/np.sum(y))
				# print(xobs)
				resultlist.append(xobs)
			plt.axvspan(xmin=mu-sigma, xmax=mu+sigma, alpha=0.5, color='tomato',)
			plt.axvline(x=mu, ls='--', alpha=1.0, color='tomato', lw=3)
			plt.plot(x, y, lw=3, alpha=0.75, color='grey')
			plt.hist(resultlist, lw=3, alpha=0.75, color='k', histtype='step', density=True)
			plt.xlabel(r'$\rm m_{obs}$')
			plt.plot(x, y)
			plt.savefig(path_plot, overwrite=True)
		else:
			pass
		#	more complicated choice with the fixed random seed
		np.random.seed(int((nseed+1)+(mu*1e2)))
		return np.random.choice(x, p=y/np.sum(y))
	except:
		# print('Something goes wrong (add_noise function)')
		return None
#----------------------------------------------------------------
def get_random_point(mu, sigma, n=10):
	"""
	mu, sigma = 17.5, 0.1
	n = 10
	"""
	from scipy.stats import norm
	import numpy as np
	
	x = np.arange(mu-sigma*n, mu+sigma*n, sigma*1e-3)
	y = norm(mu, sigma).pdf(x)

	return np.random.choice(x, p=y/np.sum(y))
#----------------------------------------------------------------
# def calc_distance_modulus(d, derr):
# 	"""
# 	d = 40*u.Mpc
# 	derr = d*0.3
# 	"""
# 	#	Distance modulus
# 	mu = 5*np.log10(d.to(u.pc).value)-5
# 	mu_up = 5*np.log10((d+derr).to(u.pc).value)-5	# upperlimit
# 	mu_lo = 5*np.log10((d-derr).to(u.pc).value)-5 # lowerlimit
# 	return (mu, mu_lo, mu_up)
def calc_distance_modulus(d):
	"""
	d = 40*u.Mpc
	"""
	#	Distance modulus
	mu = 5*np.log10(d.to(u.pc).value)-5
	return mu


#----------------------------------------------------------------
from astropy.table import Table, Column

def transpose_table(tab_before, id_col_name='ID'):
    '''Returns a copy of tab_before (an astropy.Table) with rows and columns interchanged
        id_col_name: name for optional ID column corresponding to
        the column names of tab_before'''
    # contents of the first column of the old table provide column names for the new table
    # TBD: check for duplicates in new_colnames & resolve
    new_colnames=tuple(tab_before[tab_before.colnames[0]])
    # remaining columns of old table are row IDs for new table 
    new_rownames=tab_before.colnames[1:]
    # make a new, empty table
    tab_after=Table(names=new_colnames)
    # add the columns of the old table as rows of the new table
    for r in new_rownames:
        tab_after.add_row(tab_before[r])
    if id_col_name != '':
        # add the column headers of the old table as the id column of new table
        tab_after.add_column(Column(new_rownames, name=id_col_name),index=0)
    return(tab_after)


#----------------------------------------------------------------
def normalize_minmax(x, arr):
	min_val = min(arr)
	max_val = max(arr)
	return (x-min_val) / (max_val - min_val)

def denormalize(x, arr):
	y = x*(np.max(arr) - np.min(arr)) + np.min(arr)
	return y

#----------------------------------------------------------------
from astropy.cosmology import z_at_value, WMAP9
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pickle

def interp_gw170817like_kn(path_kntable, path_pickle=None):
	#	Only for the GW170817-like KN
	spectrumlist = sorted(glob.glob(f"{path_kntable}/Run*TP*wind2*.fits"))

	#	All models share the same wavelength array
	inlam = np.unique(Table.read(spectrumlist[0])['lam'])

	print(f"spectrum: {len(spectrumlist)}")
	#	Ejecta mass, velocity
	mdarr = np.array([0.001, 0.003, 0.01, 0.03, 0.1])
	vdarr = np.array([0.05, 0.15, 0.3])
	mwarr = np.array([0.001, 0.003, 0.01, 0.03, 0.1])
	vwarr = np.array([0.05, 0.15, 0.3])
	#	Viewing angle
	angles = np.linspace(0, 180, 54)

	sptbl = Table.read(spectrumlist[0])
	splam = np.unique(np.array(sptbl['lam'])) # no shift
	phase = np.array([float(sptbl.meta['comments'][i].split('= ')[-1]) for i in range(len(sptbl.meta['comments']))])[:int(len(sptbl)/len(splam))]
	mdmean = normalize_minmax(np.mean(mdarr), mdarr)
	vdmean = normalize_minmax(np.mean(vdarr), vdarr)
	mwmean = normalize_minmax(np.mean(mwarr), mwarr)
	vwmean = normalize_minmax(np.mean(vwarr), vwarr)
	angmean = normalize_minmax(np.mean(angles), angles)
	# phasemean = normalize_minmax(np.mean(phase), phase)
	phasemean = normalize_minmax(0.3, phase)
	#	Set min, max = 20, 40
	# mumean = (30-20)/(40-20)
	mumean = (37.5-20)/(40-20)
	# mumean = 0.5

	# print("Rough Initial Guess")
	# print(f"md   ={mdmean:1.3f}")
	# print(f"vd   ={vdmean:1.3f}")
	# print(f"mw   ={mwmean:1.3f}")
	# print(f"vw   ={vwmean:1.3f}")
	# print(f"ang  ={angmean:1.3f}")
	# print(f"phase={phasemean:1.3f}")
	# print(f"mu   ={mumean:1.3f}")
	nsplam = len(splam)
	nmdarr = len(mdarr)
	nvdarr = len(vdarr)
	nmwarr = len(mwarr)
	nvwarr = len(vwarr)
	nangles = len(angles)
	nsplam = len(splam)
	nphase = len(phase)

	# print(f"Number of dynamical masses: {nmdarr}")
	# print(f"Number of dynamical velocities: {nvdarr}")
	# print(f"Number of wind masses: {nmwarr}")
	# print(f"Number of wind velocities: {nvwarr}")
	# print(f"Number of angles: {nangles}")
	# print(f"Number of phases: {nphase}")
	# print(f"Number of wavelengths: {nsplam}")

	#	Redshift Array
	muarr = np.arange(20, 50+0.01, 0.01)
	zarr = z_at_value(WMAP9.distmod, muarr*u.mag).value
	#	7D interpolator
	length_spectrum = 2156544
	fluxarr = np.zeros(length_spectrum*len(spectrumlist))
	#	Each spectrum list
	for ss, spec in enumerate(spectrumlist):
		print(f"[{ss+1}/{len(spectrumlist)}] {os.path.basename(spec)}", end='\r')
		_ = Table.read(spec)
		#	Viewing angle
		for jj, nn in enumerate(np.arange(3, 56+1, 1)):
			key = f'col{nn}'
			st = int((ss*length_spectrum)+jj*(length_spectrum/nangles))
			ed = int((ss*length_spectrum)+(jj+1)*(length_spectrum/nangles))
			fluxarr[st:ed] = np.array(_[key])

	#	Fold the flux array
	data = fluxarr.reshape(nmdarr, nvdarr, nmwarr, nvwarr, nangles, nphase, nsplam)


	# Create the interpolator
	interp = RegularGridInterpolator((mdarr, vdarr, mwarr, vwarr, angles, phase, splam), data, method='linear')
	#	Dump?
	if path_pickle is not None:
		print(f"Dumping interpolator to {path_pickle}...")
		with open(path_pickle, 'wb') as f:
			pickle.dump(interp, f)

	print(f"\nDONE!\n")
	return interp


def get_mean_kn_parameter(path_kntable):
	#	Only for the GW170817-like KN
	spectrumlist = sorted(glob.glob(f"{path_kntable}/Run*TP*wind2*.fits"))

	#	All models share the same wavelength array
	inlam = np.unique(Table.read(spectrumlist[0])['lam'])

	print(f"spectrum: {len(spectrumlist)}")
	#	Ejecta mass, velocity
	mdarr = np.array([0.001, 0.003, 0.01, 0.03, 0.1])
	vdarr = np.array([0.05, 0.15, 0.3])
	mwarr = np.array([0.001, 0.003, 0.01, 0.03, 0.1])
	vwarr = np.array([0.05, 0.15, 0.3])
	#	Viewing angle
	angles = np.linspace(0, 180, 54)

	sptbl = Table.read(spectrumlist[0])
	splam = np.unique(np.array(sptbl['lam'])) # no shift
	phase = np.array([float(sptbl.meta['comments'][i].split('= ')[-1]) for i in range(len(sptbl.meta['comments']))])[:int(len(sptbl)/len(splam))]
	mdmean = normalize_minmax(np.mean(mdarr), mdarr)
	vdmean = normalize_minmax(np.mean(vdarr), vdarr)
	mwmean = normalize_minmax(np.mean(mwarr), mwarr)
	vwmean = normalize_minmax(np.mean(vwarr), vwarr)
	angmean = normalize_minmax(np.mean(angles), angles)
	phasemean = normalize_minmax(0.3, phase)
	#	Set min, max = 20, 40
	mumean = (37.5-20)/(40-20)

	return (inlam, splam, phase, mdmean, vdmean, mwmean, vwmean, angmean, phasemean, mumean)

def register_custom_filters_on_speclite(path):
	#	Declare the speclite filtersets
	import speclite.filters
	#	Medium-band (25nm)
	rsptbl = ascii.read(f'{path}/7dt_filter.m4000_to_m8750.25nm.response.dat')
	filterlist = np.unique(rsptbl['name'])
	for filte in filterlist:
		#	Filter Table
		fltbl = rsptbl[rsptbl['name']==filte]

		_ = speclite.filters.FilterResponse(
			wavelength = fltbl['lam']*u.Angstrom,
			response = fltbl['response'], meta=dict(group_name='med25nm', band_name=filte)
		)
	#	Medium-band (50nm)
	rsptbl = ascii.read(f'{path}/7dt_filter.m3750_to_m9000.50nm.response.dat')
	filterlist = np.unique(rsptbl['name'])
	for filte in filterlist:
		#	Filter Table
		fltbl = rsptbl[rsptbl['name']==filte]

		_ = speclite.filters.FilterResponse(
			wavelength = fltbl['lam']*u.Angstrom,
			response = fltbl['response'], meta=dict(group_name='med50nm', band_name=filte)
		)
	#	Broad-band (SDSS)
	rsptbl = ascii.read(f'{path}/7dt_filter.ugriz.broad.response.dat')
	filterlist = np.unique(rsptbl['name'])
	for filte in filterlist:
		#	Filter Table
		fltbl = rsptbl[rsptbl['name']==filte]

		_ = speclite.filters.FilterResponse(
			wavelength = fltbl['lam']*u.Angstrom,
			response = fltbl['response'], meta=dict(group_name='broad', band_name=filte)
		)
	print(f"Use `med25nm`, `med50nm`, `broad` as `group_name`")

def prepare_rf_train_data(infotbl, param_keys, phasearr, number_of_unique_phase, number_of_unique_wavelength, phase_upper=30., lam_lower=2000., lam_upper=10000.):
	X = []
	y = []

	for mm, model in enumerate(infotbl['model'][:]):
		_mdtbl = Table.read(model)

		indx = np.where(
			(_mdtbl['col1'] <= phase_upper) &
			(_mdtbl['col2'] >= lam_lower) &
			(_mdtbl['col2'] <= lam_upper)
		)
		mdtbl = _mdtbl[indx]

		param_values = []
		for key in param_keys:
			param_values.append(infotbl[key][mm])

		# 해당 모델의 스펙트럼 데이터 추출
		model_spectrum = mdtbl['col3'].reshape(number_of_unique_phase, number_of_unique_wavelength)

		# 데이터 포인트 생성 및 추가
		for pp, param5 in enumerate(phasearr):
			_param_values = param_values.copy()
			_param_values.append(param5)
			X.append(_param_values)
			y.append(model_spectrum[pp])
	X = np.array(X)
	y = np.array(y)
	return X, y

def tablize_sedinfo(path_sedinfo, models):
	infotbl = Table()
	infotbl['model'] = models
	
	with open(path_sedinfo, 'r') as f:
		for line in f:
			#	Find parameter row
			if line.startswith('PARNAMES: '):
				# print(line)
				headers = line.split()[1:]
				break
		#	Generate empty columns
		for header in headers:
			infotbl[header] = 0.0

		for ll, line in enumerate(f):
			if ll < len(infotbl):
				#	Values
				if line.startswith('SED:'):
					vals = line.split()[2:]
					for hh, header in enumerate(headers):
						infotbl[header][ll] = float(vals[hh])

	return infotbl



#==========

filterlist_med25nm = (
	('m4000', '25nm', 'med25nm'),
	('m4250', '25nm', 'med25nm'),
	('m4500', '25nm', 'med25nm'),
	('m4750', '25nm', 'med25nm'),
	('m5000', '25nm', 'med25nm'),
	('m5250', '25nm', 'med25nm'),
	('m5500', '25nm', 'med25nm'),
	('m5750', '25nm', 'med25nm'),
	('m6000', '25nm', 'med25nm'),
	('m6250', '25nm', 'med25nm'),
	('m6500', '25nm', 'med25nm'),
	('m6750', '25nm', 'med25nm'),
	('m7000', '25nm', 'med25nm'),
	('m7250', '25nm', 'med25nm'),
	('m7500', '25nm', 'med25nm'),
	('m7750', '25nm', 'med25nm'),
	('m8000', '25nm', 'med25nm'),
	('m8250', '25nm', 'med25nm'),
	('m8500', '25nm', 'med25nm'),
	('m8750', '25nm', 'med25nm'),
)

filterlist_griz = (
	# ('u', 'broad', 'broad'),
	('g', 'broad', 'broad'),
	('r', 'broad', 'broad'),
	('i', 'broad', 'broad'),
	('z', 'broad', 'broad'),
)

filterlist_ugriz = (
	('u', 'broad', 'broad'),
	('g', 'broad', 'broad'),
	('r', 'broad', 'broad'),
	('i', 'broad', 'broad'),
	('z', 'broad', 'broad'),
)

# http://svo2.cab.inta-csic.es/svo/theory/fps/index.php?mode=browse&gname=SLOAN&asttype=
bandwidtharr_broad_ugriz = np.array([
	#	ugriz
	540.97,
	1064.68,
	1055.51,
	1102.57,
	1164.01,
	#	u'g'r'i'z'
	# 563.56,
	# 1264.52,
	# 1253.71,
	# 1478.93,
	# 4306.72,
])
bandwidtharr_broad_griz = np.array([
	#	griz
	1064.68,
	1055.51,
	1102.57,
	1164.01,
])
bandwidtharr_med25nm = np.array([250]*20)
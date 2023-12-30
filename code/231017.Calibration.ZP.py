# %% [markdown]
# # Library

# %%
# Python Library
import os, glob, sys
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.io import fits
import numpy as np
from astropy.table import Table, vstack, hstack
from astropy import units as u
from astropy.coordinates import SkyCoord
import warnings
warnings.filterwarnings("ignore")

# Plot presetting
import matplotlib.pyplot as plt
import matplotlib as mpl
#
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')

# %%
plt.style.use('dark_background') # Dark 모드 스타일 적용

# %%
from astropy.visualization import ZScaleInterval, LinearStretch, LogStretch
from astropy.visualization import (MinMaxInterval, SqrtStretch, ImageNormalize)
from matplotlib.colors import LogNorm  # LogNorm 추가
from matplotlib.patches import Circle

# %%
from astropy.stats import sigma_clipped_stats

# %%
def calc_weighted_median_zp(zp_arr, zperr_arr):
	weights = [1 / error for error in zperr_arr]

	weighted_sum = sum(value * weight for value, weight in zip(zp_arr, weights))
	total_weight = sum(weights)
	zp = weighted_sum / total_weight

	total_weight = sum(weights)
	weighted_diff_sum = sum(((value - zp) ** 2) * weight for value, weight in zip(zp_arr, weights))
	zperr = (weighted_diff_sum / (total_weight - 1)) ** 0.5

	print(f"zp = {zp:.3f}+/-{zperr:.3f}")
	return zp, zperr

# %%
def check_slope(x, y, yerr, plot=False, verbose=False):
	from sklearn.linear_model import LinearRegression

	model = LinearRegression()
	# x = mtbl[f'{filte}mag'].reshape(-1, 1)
	# y = zp_ps1_arr.reshape(-1, 1)
	x = x.reshape(-1, 1)
	y = y.reshape(-1, 1)
	model.fit(x, y, sample_weight=1 / yerr**2)

	# 회귀 결과 추출
	slope = model.coef_[0][0]  # 기울기
	intercept = model.intercept_[0]  # y 절편

	if plot:
		# 회귀선 그리기
		plt.scatter(x, y, label='Data', alpha=0.5)
		plt.plot(x, model.predict(x), color='red', label=f'Linear Regression (a={slope:.3f}, b={intercept:.3f})')
		plt.xlabel(r"$\rm m_{PS1}$")
		plt.ylabel('ZP')
		plt.ylim([intercept+2, intercept-2])
		plt.title('Weighted Linear Regression')
		plt.legend(loc='upper center')

		plt.show()

	if verbose:
		# 회귀 결과 출력
		print(f"Slope (기울기): {slope}")
		print(f"Intercept (y 절편): {intercept}")
	return (slope, intercept)

# %% [markdown]
# # Input

# %%
# tname = 'LTT7987'
# tra, tdec = 302.7343929, -30.2200720
# radius = 1.5 # [deg]
#
tname = 'FEIGE110'
tra, tdec = 349.9933320184, -5.1656030952 # [deg], [deg]
radius = 1.5 # [deg]
#
# tname = 'LTT1020'
# tra, tdec = 28.709, -27.477 # [deg], [deg]
# radius = 1.5 # [deg]


c_target = SkyCoord(tra, tdec, unit='deg')

# %%
path_save = f'../output/{tname}'
if not os.path.exists(path_save):
    os.makedirs(path_save)

# %% [markdown]
# ## Source EXtractor

# %%
#	Source EXtractor
imlist = sorted(glob.glob(f'../data/spss/{tname}/C*m.fits'))
for _inim in imlist:
	inim = os.path.basename(_inim)
	outcat = inim.replace('.fits', '.cat')
	sexcom = f"sex -c simple.sex -CATALOG_NAME {outcat} {inim}"
	print(sexcom)

# %% [markdown]
# ## Read Tables

# %%
catlist = sorted(glob.glob(f'../data/spss/{tname}/C*cat'))
print(f"{len(catlist)} catalogs found")

# %% [markdown]
# - PS1

# %%
ps1cat = f'../data/spss/{tname}/ps1-{tname}.cat'
if os.path.exists(ps1cat):
	ps1tbl = Table.read(ps1cat, format='ascii')
	c_ps1 = SkyCoord(ra=ps1tbl['RA_ICRS'], dec=ps1tbl['DE_ICRS'], frame='icrs')

# %%
xstbl = Table.read(f'../output/{tname}/7dt.all.phot.csv')

# %% [markdown]
# - Gaia

# %%
gtbl = Table.read(f'../output/gaia-{tname}.cat', format='csv')
# gtbl = Table.read(f'../output/{tname}.pre.csv', format='csv')
c_gaia = SkyCoord(ra=gtbl['ra'], dec=gtbl['dec'], unit='deg', frame='icrs')

# %%
outbl = Table()
outbl['image'] = [os.path.basename(inim) for inim in imlist]
outbl['filter'] = [inim.split('-')[-2] for inim in imlist]
outbl['zp_ps1'] = 0.0
outbl['zperr_ps1'] = 0.0
outbl['zp_gaia'] = 0.0
outbl['zperr_gaia'] = 0.0
outbl['zp_calspec'] = 0.0
outbl['zperr_calspec'] = 0.0
# %% [markdown]
# # Analysis

broad_filterlist = ['g', 'r', 'i', 'z',]
med_filterlist = ['m400', 'm425', 'm650, m675']
# %%
nn = 2
for nn in range(len(catlist)):
	incat = catlist[nn]
	inim = imlist[nn]
	#	Image
	filte = inim.split('-')[-2]
	data = fits.getdata(inim)
	# bkg = np.median(data)
	# print(f"{os.path.basename(inim)} (bkg={bkg:.3f})")
	print(os.path.basename(incat))

	#	Table
	intbl = Table.read(incat, format='ascii.sextractor')
	intbl['SNR'] = intbl['FLUX_AUTO']/intbl['FLUXERR_AUTO']
	c_7dt = SkyCoord(ra=intbl['ALPHA_J2000'], dec=intbl['DELTA_J2000'])

	indx_select = np.where(
		(intbl['FLAGS']==0) &
		(intbl['CLASS_STAR']>0.9)
	)
	plt.close('all')
	plt.plot(tra, tdec, '+', c='r', ms=15, label=tname, zorder=999)
	plt.plot(intbl['ALPHA_J2000'], intbl['DELTA_J2000'], '.', alpha=0.1, c='silver',)
	plt.plot(intbl['ALPHA_J2000'][indx_select], intbl['DELTA_J2000'][indx_select], '.', label='Star-like Sources')
	plt.plot(gtbl['ra'], gtbl['dec'], '.', label='Gaia')
	plt.legend(loc='upper right', framealpha=1.0)

	xl, xr = plt.xlim()
	plt.xlim([xr, xl])
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.xlabel('RA [deg]')
	plt.ylabel('Dec [deg]')
	plt.tight_layout()
	plt.savefig(f"{path_save}/{tname}.radec.png",  transparent=True)

	# ## Matching with Gaia

	indx_match, sep, _ = c_7dt.match_to_catalog_sky(c_gaia)
	_mtbl = hstack([intbl, gtbl[indx_match]])
	_mtbl = _mtbl[(sep.arcsec<2)]
	print(f"Matched {len(_mtbl)} sources")

	_zp_gaia_arr = _mtbl[f'{filte}']-_mtbl['MAG_AUTO']
	_zperr_gaia_arr = _mtbl[f'{filte}err']

	fig = plt.figure(figsize=(10, 6))
	plt.axis('equal')

	_zp_gaia_tmp = np.median(_zp_gaia_arr)

	if nn == 2:
		plt.close('all')
		plt.scatter(_mtbl['X_IMAGE'], _mtbl['Y_IMAGE'], c=_zp_gaia_arr, vmin=_zp_gaia_tmp-0.25, vmax=_zp_gaia_tmp+0.25)
		cbar = plt.colorbar()
		cbar.set_label('ZP')
		plt.xlabel('X (pixels)')
		plt.ylabel('Y (pixels)')
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.tight_layout()
		plt.savefig(f"{path_save}/{tname}.zp_spatial_dist.png",  transparent=True)

	if filte in ['u', 'g', 'r', 'i', 'z']:
		mtbl = _mtbl[
			(_mtbl['FLAGS']==0) &
			(_mtbl['SNR']>50) &
			(_mtbl[f'cflag_{filte}']=='True')
			]

	else:
		mtbl = _mtbl[
			# (sep.arcsec<2)
			# (sep.arcsec<2) &
			# (~_mtbl[f'{filte}'].mask) &
			(_mtbl['FLAGS']==0) &
			#
			# (~_mtbl[f'{filte}'].mask) &
			# (_mtbl[f'{filte}']>14) &
			# (_mtbl[f'{filte}']<18) &
			(_mtbl['SNR']>50)
			]

	print(f"Selected {len(mtbl)} sources")

	zp_gaia_arr = mtbl[f'{filte}']-mtbl['MAG_AUTO']
	zperr_gaia_arr = mtbl[f'{filte}err']

	from astropy.stats import sigma_clip

	sigma = 3.0
	maxiters = None

	#	SIGMA CLIPPING
	zp_gaia_clip_arr = sigma_clip(
		zp_gaia_arr.copy(),
		sigma=sigma,
		maxiters=maxiters,
		cenfunc=np.median,
		copy=False
		)
	indx_alive = np.where( zp_gaia_clip_arr.mask == False )
	indx_exile = np.where( zp_gaia_clip_arr.mask == True )

	print(f"- {sigma} Sigma Clipping")
	print(f"Alive: {len(zp_gaia_arr[indx_alive])}")
	print(f"Exile: {len(zp_gaia_arr[indx_exile])}")

	zp_gaia, zperr_gaia = np.median(zp_gaia_arr[indx_alive]), np.std(zp_gaia_arr[indx_alive])

	plt.close('all')
	plt.title(f"{tname} ({filte})",)
	plt.plot(_mtbl[f'{filte}'], _zp_gaia_arr, '.', c='silver', alpha=0.5, zorder=0)
	plt.errorbar(mtbl[f'{filte}'], zp_gaia_arr, yerr=zperr_gaia_arr, ls='none', c='grey', alpha=0.5, zorder=0)
	plt.plot(mtbl[f'{filte}'][indx_alive], zp_gaia_arr[indx_alive], '.', color='dodgerblue', alpha=0.5)
	plt.plot(mtbl[f'{filte}'][indx_exile], zp_gaia_arr[indx_exile], 'xr', alpha=0.5)

	plt.axhline(y=zp_gaia, ls='-', color='lime', zorder=999, label=f'ZP={zp_gaia:.2f}+/-{zperr_gaia:.2f}')
	plt.axhspan(ymin=zp_gaia-zperr_gaia, ymax=zp_gaia+zperr_gaia, color='lime', zorder=0, alpha=0.5)
	plt.xlabel(r"$\rm m_{GaiaXP}$")
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.ylabel("ZP")
	plt.xlim([10, 21])
	# plt.ylim(zp_gaia-1.5, zp_gaia+1.5)
	plt.ylim(zp_gaia-0.5, zp_gaia+0.5)
	plt.legend(loc='upper center', fontsize=14)
	plt.tight_layout()

	plt.savefig(f"{path_save}/{tname}.gaia.zpcal.{filte}.png",  transparent=True)

	# ## Matching with PS1
	if filte in broad_filterlist:
		indx_match, sep, _ = c_7dt.match_to_catalog_sky(c_ps1)
		_mtbl = hstack([intbl, ps1tbl[indx_match]])
		print(f"Matched {len(_mtbl)} sources")

		_zp_ps1_arr = _mtbl[f'{filte}mag']-_mtbl['MAG_AUTO']
		_zperr_ps1_arr = np.sqrt( (_mtbl[f'e_{filte}mag']**2) + (_mtbl['MAGERR_AUTO']**2) )

		try:
			mtbl = _mtbl[
				(sep.arcsec<2) &
				(~_mtbl[f'{filte}mag'].mask) &
				(_mtbl[f'{filte}mag']>14) &
				(_mtbl[f'{filte}mag']<18) &
				(_mtbl['SNR']>50) &
				(_mtbl['CLASS_STAR']>0.9) &
				(~_mtbl[f'e_{filte}mag'].mask)
				]
		except:
			mtbl = _mtbl[
				(sep.arcsec<2) &
				(_mtbl[f'{filte}mag']>14) &
				(_mtbl[f'{filte}mag']<18) &
				(_mtbl['SNR']>50) &
				(_mtbl['CLASS_STAR']>0.9)
				]

		print(f"Selected {len(mtbl)} sources")

		zp_ps1_arr = mtbl[f'{filte}mag']-mtbl['MAG_AUTO']
		zperr_ps1_arr = np.sqrt( (mtbl[f'e_{filte}mag']**2) + (mtbl['MAGERR_AUTO']**2) )

		from astropy.stats import sigma_clip

		sigma = 3.0
		maxiters = None

		#	SIGMA CLIPPING
		zp_ps1_clip_arr = sigma_clip(
			zp_ps1_arr.copy(),
			sigma=sigma,
			maxiters=maxiters,
			cenfunc=np.median,
			copy=False
			)
		indx_alive = np.where( zp_ps1_clip_arr.mask == False )
		indx_exile = np.where( zp_ps1_clip_arr.mask == True )

		print(f"- {sigma} Sigma Clipping")
		print(f"Alive: {len(zp_ps1_arr[indx_alive])}")
		print(f"Exile: {len(zp_ps1_arr[indx_exile])}")

		zp_ps1, zperr_ps1 = calc_weighted_median_zp(zp_ps1_arr[indx_alive], zperr_ps1_arr[indx_alive])
		plt.close('all')
		plt.plot(_mtbl[f'{filte}mag'], _zp_ps1_arr, '.', c='silver', alpha=0.5, zorder=0)
		# plt.plot(mtbl[f'{filte}mag'], zp_ps1_arr, '.', alpha=0.5)
		plt.plot(mtbl[f'{filte}mag'][indx_alive], zp_ps1_arr[indx_alive], '.', color='dodgerblue', alpha=0.5)
		plt.plot(mtbl[f'{filte}mag'][indx_exile], zp_ps1_arr[indx_exile], 'xr', alpha=0.5)

		plt.axhline(y=zp_ps1, ls='-', color='tomato', zorder=999, label=f'ZP={zp_ps1:.2f}+/-{zperr_ps1:.2f}')
		plt.axhspan(ymin=zp_ps1-zperr_ps1, ymax=zp_ps1+zperr_ps1, color='tomato', zorder=0, alpha=0.5)
		plt.xlabel(r"$\rm m_{PS1}$")
		plt.ylabel("ZP")
		plt.xlim([12, 21])
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.ylim(zp_ps1-1.5, zp_ps1+1.5)
		plt.legend(loc='upper center', fontsize=14)
		plt.tight_layout()
		plt.savefig(f"{path_save}/{tname}.ps1.zpcal.{filte}.png",  transparent=True)

	# ## Xshooter or CALSPEC


	if 'm' in filte:
		refmag = xstbl['magapp'][xstbl['filter']==f"{filte}0"].item()
	else:
		refmag = xstbl['magapp'][xstbl['filter']==filte].item()
	print(f"ref.mag in {filte}: {refmag}")

	indx, sep, _ = c_target.match_to_catalog_sky(c_7dt)

	mag = intbl['MAG_AUTO'][indx]
	magerr = intbl['MAGERR_AUTO'][indx]
	print(f"inst.mag={mag:.3f}+/-{magerr:.3f}")
	print('sep=', sep.arcsec)

	zp = refmag - mag
	print(f"ZP={zp:.3f}")

	# # Result

	plt.close('all')
	plt.axhline(y=zp, label='CALSPEC', lw=3, zorder=0)
	plt.errorbar(0, zp_gaia, yerr=zperr_gaia, ms=10, color='lime', fmt='o', label='Gaia')
	if filte in broad_filterlist:
		plt.errorbar(1, zp_ps1, yerr=zperr_ps1, ms=10, color='tomato', fmt='o', label='PS1')
	plt.ylim(zp-0.25, zp+0.25)
	plt.xlim(-0.5, 1.5)
	xticks = plt.xticks([])
	plt.yticks(fontsize=14)
	plt.xticks(fontsize=14)

	plt.legend(loc='upper center', ncol=3, fontsize=14)
	plt.ylabel('ZP')
	plt.tight_layout()
	plt.savefig(f'{path_save}/{tname}.zpcomp.{filte}.png',  transparent=True)

	plt.close('all')
	plt.axhline(y=zp-zp, label='CALSPEC', lw=3, zorder=0)
	plt.errorbar(0, zp_gaia-zp, yerr=zperr_gaia, ms=10, color='lime', fmt='o', label='Gaia')
	if filte in broad_filterlist:
		plt.errorbar(1, zp_ps1-zp, yerr=zperr_ps1, ms=10, color='tomato', fmt='o', label='PS1')
	# plt.ylim(zp-0.25, zp+0.25)
	plt.ylim(-0.4, +0.4)
	plt.xlim(-0.5, 1.5)
	xticks = plt.xticks([])
	plt.yticks(fontsize=14)
	plt.xticks(fontsize=14)
	plt.legend(loc='upper center', ncol=3, fontsize=14)
	plt.ylabel(r'$\rm ZP-ZP_{CALSPEC}$')
	plt.tight_layout()
	plt.savefig(f'{path_save}/{tname}.zpres.{filte}.png',  transparent=True)

	##
	if filte in broad_filterlist:
		outbl['zp_ps1'][nn] = zp_ps1
		outbl['zperr_ps1'][nn] = zperr_ps1
	outbl['zp_gaia'][nn] = zp_gaia
	outbl['zperr_gaia'][nn] = zperr_gaia
	outbl['zp_calspec'][nn] = zp
	outbl['zperr_calspec'][nn] = magerr

#%%
for key in outbl.keys():
	try:
		outbl[key].format = '.3f'
	except:
		pass


outbl['lam'] = 0.0
# for ff, filte in enumerate(outbl['filter']):
for ff, filte in enumerate(outbl['filter']):
	if 'm' in filte:
		_lam = xstbl['lam'][xstbl['filter']==f"{filte}0"].item()
	else:
		_lam = xstbl['lam'][xstbl['filter']==filte].item()
	outbl['lam'][ff] = _lam

outbl['delta_zp'] = outbl['zp_gaia'] - outbl['zp_calspec']

outbl.write(f"{path_save}/result.csv", format='csv', overwrite=True)

plt.close('all')
plt.plot(outbl['lam'], outbl['zperr_gaia'], 'o')
for ff, filte in enumerate(outbl['filter']):
	plt.text(outbl['lam'][ff], outbl['zperr_gaia'][ff], filte, fontsize=14)
plt.xlabel(r'Wavelength [$\rm \AA$]')
plt.ylabel(r'$ZP_{Gaia}$ Error')
plt.ylim([0, 0.25])
plt.title(os.path.basename(inim), fontsize=10)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(f'{path_save}/{os.path.basename(inim)}.zperr.png',  transparent=True)


#%%
plt.close('all')
delzparr = outbl['zp_gaia']-outbl['zp_calspec']


plt.plot(outbl['lam'][5], delzparr[5], 'o')
plt.plot(outbl['lam'][6], delzparr[6], 'o')
plt.plot(outbl['lam'][7], delzparr[7], 'o')
plt.plot(outbl['lam'][8], delzparr[8], 'o')

for ff, filte in enumerate(outbl['filter']):
	if 'm' in filte:
		plt.text(outbl['lam'][ff], delzparr[ff], filte, fontsize=14)

plt.axhline(y=0, ls='--', color='grey', lw=3, zorder=0)

plt.xlabel(r'Wavelength [$\rm \AA$]')
# plt.ylabel(r'$\rm \Delta ZP$')
plt.ylabel(r'$ZP_{GaiaXP}-ZP_{CALSPEC}$')
yl, yu = plt.ylim()
plt.ylim(-np.max([np.abs(yl), np.abs(yu)]), np.max([np.abs(yl), np.abs(yu)]))
# plt.ylim([-0.05, +0.05])
# plt.yticks(np.arange(-0.06, +0.06, +0.02))
# plt.title(os.path.basename(inim), fontsize=10)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(f'{path_save}/{os.path.basename(inim)}.delzp.png',  transparent=True)
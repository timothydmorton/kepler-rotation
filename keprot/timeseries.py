from __future__ import print_function,division
import os,sys,re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import median_filter
from scipy.optimize import leastsq, curve_fit

from scipy.signal import lombscargle

from pkg_resources import resource_filename


import logging

from keputils import koiutils as ku

import acor

from .findpeaks import peakdetect

try:
    KOI_PHOTOMETRY_DIR = os.environ['KOI_PHOTOMETRY_DIR']
except KeyError:
    logging.warning('KOI_PHOTOMETRY_DIR environment variable not defined.')
    KOI_PHOTOMETRY_DIR = '.'

CADENCE = 0.02043423
QTIMESFILE = resource_filename('keprot','data/qStartStop.txt')
QSTART = {}
QSTOP = {}
for line in open(QTIMESFILE):
    line = line.split()
    if line[0] != 'q':
        QSTART[int(line[0])] = float(line[1])
        QSTOP[int(line[0])] = float(line[2])



class TimeSeries(object):
    def __init__(self, t, f, mask=None, cadence=CADENCE,
                 default_maxlag=200//CADENCE):
        self.t = np.atleast_1d(t)
        self.f = np.atleast_1d(f)
        if mask is None:
            mask = np.isnan(self.f)
        self.mask = mask

        self.cadence = cadence
        self.default_maxlag = default_maxlag

        #set private variables for cached acorr calculation
        self._lag = None  #always should be in cadences
        self._ac = None 

        #private variables for caching pgram calculation
        self._pers = None
        self._pgram = None

    def acorr(self, maxlag=None, smooth=18, days=True,
              recalc=False):
        if maxlag is None:
            maxlag = self.default_maxlag

        #don't recalculate the same thing if not necessary
        if self._ac is not None and not recalc:
            lag = self._lag
            ac = self._ac
        else:
            x = self.f.copy()
            x[self.mask] = 0

            #logging.debug('{} nans in x'.format((np.isnan(x)).sum()))

            ac = acor.function(x, maxlag)
            lag = np.arange(maxlag)

            #smooth AC function
            ac = gaussian_filter(ac, smooth)

            #set private variables for cached calculation
            self._ac = ac
            self._lag = lag
            self._maxlag = maxlag
            self._smooth = smooth

        if days:
            return lag*CADENCE,ac
        else:
            return lag,ac

    def acorr_peaks(self, lookahead=5, days=True, 
                    return_heights=False, **kwargs):
        lag, ac = self.acorr(days=days, **kwargs)
        return peaks_and_lphs(ac, lag, return_heights=return_heights,
                              lookahead=lookahead)
        
    def plot_acorr(self, days=True, smooth=18, maxlag=None,
                   mark_period=True, lookahead=5, fit_npeaks=4,
                   tol=0.2,
                   **kwargs):

        lag, ac = self.acorr(days=days, smooth=smooth, maxlag=maxlag)

        plt.plot(lag, ac, **kwargs)

        pks, lphs = self.acorr_peaks(smooth=smooth,
                                     maxlag=maxlag,
                                     lookahead=lookahead)
        #plt.ylim(ymax=1)

        if mark_period:
            if mark_period is True:
                mark_period = None
            p,e_p,pks,lphs,hts = self.acorr_period_fit(period=mark_period, return_peaks=True,
                                                 fit_npeaks=fit_npeaks, tol=tol,
                                                 smooth=smooth,
                                                 maxlag=maxlag,
                                                 lookahead=lookahead)
            plt.xlim(xmax=min((fit_npeaks+1)*p, lag.max()))

            for pk in pks:
                plt.axvline(pk, ls=':')

    def acorr_period_fit(self, period=None, fit_npeaks=4,
                   smooth=18, maxlag=None, lookahead=5,
                   tol=0.2, return_peaks=False):
        peaks, lphs, hts = self.acorr_peaks(smooth=smooth, maxlag=maxlag,
                                            lookahead=lookahead, return_heights=True)

        if lphs[0] >= lphs[1]:
            firstpeak = peaks[0]
        else:
            firstpeak = peaks[1]
            if lphs[1] < 1.2*lphs[0]:
                logging.warning('Second peak (selected) less than 1.2x height of first peak.')        

        if period is None:
            period = firstpeak

        if fit_npeaks > len(peaks):
            fit_npeaks = len(peaks)
        #peaks = peaks[:fit_npeaks]

        #identify peaks to use in fit: first 'fit_npeaks' peaks closest to integer
        # multiples of period guess

        fit_peaks = []
        fit_lphs = []
        fit_hts = []
        last = 0.
        #used = np.zeros_like(peaks).astype(bool)
        for n in np.arange(fit_npeaks)+1:
            #find highest peak within 'tol' of integer multiple (that hasn't been used)
            close = (np.absolute(peaks - n*period) < (tol*n*period)) & ((peaks-last) > 0.3*period)
            if close.sum()==0:
                fit_npeaks = n-1
                break
                #raise NoPeakError('No peak found near {}*{:.2f}={:.2f} (tol={})'.format(n,period,n*period,tol))
            ind = np.argmax(hts[close])
            last = peaks[close][ind]
            fit_peaks.append(peaks[close][ind])
            fit_lphs.append(lphs[close][ind])
            fit_hts.append(hts[close][ind])
            #used[close][ind] = True
            logging.debug('{}: {}, {}'.format(n*period,peaks[close],peaks[close][ind]))
            #logging.debug(used)

            #ind = np.argmin(np.absolute(peaks - n*period)) #closest peak
            #fit_peaks.append(peaks[ind])
            #fit_lphs.append(lphs[ind])

        logging.debug('fitting peaks: {}'.format(fit_peaks))

        if fit_npeaks < 3:
            return peaks,-1, fit_peaks, fit_lphs, fit_hts


        x = np.arange(fit_npeaks + 1)
        y = np.concatenate([np.array([0]),fit_peaks])

        #x = np.arange(fit_npeaks) + 1
        #y = fit_peaks

        def fn(x,a,b):
            return a*x + b

        fit,cov = curve_fit(fn, x, y, p0=(period,0))
        if return_peaks:
            return fit[0],cov[0][0],fit_peaks,fit_lphs,fit_hts
        else:
            return fit[0],cov[0][0]

    def periodogram(self,pmin=0.5,pmax=60,npts=2000,
              recalc=False):
        
        pers = np.logspace(np.log10(pmin),np.log10(pmax),npts)
        if np.array_equal(pers,self._pers) and not recalc:
            pgram = self._pgram
        else:
            freqs = (2*np.pi)/(pers)
            t = self.t[~self.mask]
            f = self.f[~self.mask]
            
            pgram = lombscargle(t.astype('float64'),
                                f.astype('float64'),
                                freqs.astype('float64'))
            self._pgram = pgram
            self._pers = pers

        return pers,pgram
        
    def pgram_peaks(self, npeaks=10, lookahead=5, **kwargs):
        pers,pgram = self.periodogram(**kwargs)
        maxes,mins = peakdetect(pgram,pers,lookahead=lookahead)
        maxes = np.array(maxes)
        inds = np.argsort(maxes[:,1])
        pks,hts = maxes[inds,0][-npeaks:][::-1],maxes[inds,1][-npeaks:][::-1]
        return  pks,hts

    def save_hdf(self, filename, path=''):
        """Writes data to file, along with acorr and pgram info.
        """
        data = pd.DataFrame({'t':self.t,
                             'f':self.f,
                             'mask':self.mask})
        lag, ac = self.acorr(days=False)
        acorr = pd.DataFrame({'lag':lag,
                              'ac':ac})

        pks, lphs = self.acorr_peaks()
        acorr_peaks = pd.DataFrame({'lag':pks,
                                    'lph':lphs})

        pers,pg = self.periodogram()
        pgram = pd.DataFrame({'period':pers,
                           'pgram':pg})
        
        pks, hts = self.pgram_peaks()
        pgram_peaks = pd.DataFrame({'P':pks,
                                    'height':hts})
        
        data.to_hdf(filename,'{}/data'.format(path))
        acorr.to_hdf(filename,'{}/acorr'.format(path))
        acorr_peaks.to_hdf(filename,'{}/acorr_peaks'.format(path))
        pgram.to_hdf(filename,'{}/pgram'.format(path))
        pgram_peaks.to_hdf(filename,'{}/pgram_peaks'.format(path))

        if hasattr(self,'subseries'):
            for name in self.subseries:
                self.subseries[name].save_hdf(filename, path=name)

    def make_subseries(self,tlist,names=None):
        """Splits up timeseries into chunks, according to tlist

        tlist is a list of (tstart,tstop) tuples.  If names is provided,
        those names will be used; otherwise 'sub1', 'sub2', etc.
        """
        if names is None:
            names = ['sub{}'.format(i) for i in 1+np.arange(len(tlist))]

        self.subseries = {}
        for (tlo,thi),name in zip(tlist,names):
            tok = (self.t > tlo) & (self.t < thi)
            t = self.t[tok]
            f = self.f[tok]
            mask = self.mask[tok]
            self.subseries[name] = TimeSeries(t, f, mask=mask,
                                              default_maxlag=self.default_maxlag)


class TimeSeries_FromH5(TimeSeries):
    def __init__(self, filename, path=''):

        self.filename = filename
        self.path = path

        #store = pd.HDFStore(filename, 'r')
        data = pd.read_hdf(filename, '{}/data'.format(path))
        #data = store['{}/data'.format(path)]
        t = np.array(data['t'])
        f = np.array(data['f'])
        mask = np.array(data['mask'])

        TimeSeries.__init__(self, t,f,mask=mask)

        #acorr = store['{}/acorr'.format(path)]
        acorr = pd.read_hdf(filename, '{}/acorr'.format(path))
        self._lag = np.array(acorr['lag'])
        self._ac = np.array(acorr['ac'])

        #pgram = store['{}/pgram'.format(path)]
        pgram = pd.read_hdf(filename, '{}/pgram'.format(path))
        self._pers = np.array(pgram['period'])
        self._pgram = np.array(pgram['pgram'])

        #store.close()

        i=1
        has_sub = True
        self.subseries = {}
        while has_sub:
            try:
                name = 'sub{}'.format(i)
                self.subseries[name] = TimeSeries_FromH5(filename, path='{}/{}'.format(path,name))
            except KeyError:
                has_sub = False
            i += 1
                

class Kepler_TimeSeries_Petigura(TimeSeries):
    def __init__(self, koi, folder=KOI_PHOTOMETRY_DIR, 
                 pdc=True, median_window=500, sigclip=7,
                 mask_transit=True, 
                 qtr=None,
                 qmin=3, qmax=15,
                 **kwargs):
        """TimeSeries based on Kepler data, as structured in HDF5 files
        created by Erik Petigura
        """

        self.koi = ku.koistar(koi)
        self.folder = folder
        self.filename = '{}/{}.01.h5'.format(folder,self.koi)

        self.pdc = pdc
        self.median_window = median_window
        self.sigclip = sigclip

        if qtr is not None:
            qmin = qtr
            qmax = qtr

        self.qmin = qmin
        self.qmax = qmax

        h5 = h5py.File(self.filename, 'r')

        if pdc:
            f = h5['pp']['mqcal'][:]['fpdc']
        else:
            f = h5['pp']['mqcal'][:]['f']

        mask = h5['pp']['mqcal'][:]['fmask']

        mask = (mask | np.isnan(f))

        t = h5['pp']['mqcal'][:]['t']

        tmin = QSTART[qmin]
        tmax = QSTOP[qmax]
        tok = (t > tmin) & (t < tmax)

        f = f[tok]
        t = t[tok]
        mask = mask[tok]

        #mask region near any transit.
        for num in np.arange(0.01,0.1,0.01): #.01 through .09...no 10-planet systems yet...
            koiname = '{}.{:02n}'.format(self.koi,num*100)
            try:
                per,ep = ku.DATA.ix[koiname,['koi_period','koi_time0bk']]
                dur = ku.DATA.ix[koiname,'koi_duration']/24
                intransit = np.absolute((t - ep + per/2) % per - per/2) < (dur*0.75)
                mask = (mask | intransit)
            except KeyError:
                pass

        #temporarily zero-out mask, apply median filter, then restore masked values,
        # and add outliers to mask
        fmasked = f[mask].copy()
        f[mask] = 0

        filt = median_filter(f,median_window)
        absdev = np.absolute(f - filt)
        filt2 = median_filter(absdev,median_window)
        
        f[mask] = fmasked
        mask = mask | ((absdev-filt2) > sigclip*filt2.std())


        h5.close()

        TimeSeries.__init__(self, t, f, mask=mask, **kwargs)
        
    def make_subseries(self,qlist=[(3,5),(6,8),(9,11),(12,15)]):
        tlist = [(QSTART[tlo],QSTOP[thi]) for tlo,thi in qlist]
        #names = ['q{}q{}'.format(tlo,thi) for tlo,thi in qlist]
        TimeSeries.make_subseries(self, tlist)
        

def peaks_and_lphs(y, x=None, lookahead=5, return_heights=False):
    """Returns locations of peaks and corresponding "local peak heights"
    """
    if x is None:
        x = np.arange(len(y))

    maxes, mins = peakdetect(y, x, lookahead=lookahead)
    maxes = np.array(maxes)
    mins = np.array(mins)
    
    #logging.debug('maxes: {}'.format(maxes))

    #calculate "local heights".  First will always be a minimum.
    try: #this if maxes and mins are same length 
        lphs = np.concatenate([((maxes[:-1,1] - mins[:-1,1]) + (maxes[:-1,1] - mins[1:,1]))/2.,
                               np.array([maxes[-1,1]-mins[-1,1]])])
    except ValueError: #this if mins have one more
        lphs = ((maxes[:,1] - mins[:-1,1]) + (maxes[:,1] - mins[1:,1]))/2.

    if return_heights:
        return maxes[:,0], lphs, maxes[:,1]
    else:
        return maxes[:,0], lphs 

    

def acorr_peaks(fs, mask=None, lookahead=5, smooth=18, maxlag=200//CADENCE,
                return_acorr=False, days=True):
    """Returns positions of acorr peaks, with corresponding local heights.
    """
    fs = np.atleast_1d(fs)
    if mask is None:
        mask = self.mask
    fs[mask] = 0
    
    corr = acor.function(fs,maxlag)
    lag = np.arange(maxlag)

    logging.debug('ac: {}'.format(corr))

    #lag, corr = acorr(fs, mask=mask, maxlag=maxlag)
    maxes, mins = peakdetect(corr, lag, lookahead=lookahead)
    maxes = np.array(maxes)
    mins = np.array(mins)

    logging.debug('maxes: {}'.format(maxes))

    #calculate "local heights".  First will always be a minimum.
    try: #this if maxes and mins are same length 
        lphs = np.concatenate([((maxes[:-1,1] - mins[:-1,1]) + (maxes[:-1,1] - mins[1:,1]))/2.,
                             np.array([maxes[-1,1]-mins[-1,1]])])
    except ValueError: #this if mins have one more
        lphs = ((maxes[:,1] - mins[:-1,1]) + (maxes[:,1] - mins[1:,1]))/2.

    if return_acorr:
        return corr, maxes[:-1,0], lphs[:-1] #leaving off the last one, just in case weirdness...
    else:
        return maxes[:-1,0], lphs[:-1] #leaving off the last one, just in case weirdness...

def fit_period(period, peaks, lphs, fit_npeaks=4, tol=0.2):
    """fits series of fit_npeaks peaks near integer multiples of period to a line

    tol is fractional tolerance in order to select a peak to fit.

    """
    
    #identify peaks to use in fit: first 'fit_npeaks' peaks w/in 
    # 20% of integer multiple of period
    logging.debug(np.absolute((peaks[:fit_npeaks]/period)/(np.arange(fit_npeaks)+1) - 1))
    close_peaks = np.absolute((peaks[:fit_npeaks]/period)/(np.arange(fit_npeaks)+1) - 1) < tol

    x = np.arange(fit_npeaks + 1)
    y = np.concatenate([np.array([0]),peaks[close_peaks]])
    
    def resid(a):
        return y - a*x
    
    return leastsq(resid, period, full_output=1)
    

class NoPeakError(Exception):
    pass

    

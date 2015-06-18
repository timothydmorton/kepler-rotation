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

import acor

from .utils import peaks_and_lphs
from .findpeaks import peakdetect

class TimeSeries(object):
    def __init__(self, t, f, mask=None, cadence=None,
                 default_maxlag_days=200,
                 flatten_order=2):
        self.t = np.atleast_1d(t)
        self.f = np.atleast_1d(f)
        if mask is None:
            mask = np.isnan(self.f)
        self.mask = mask

        if cadence is None:
            cadence = np.median(self.t[1:]-self.t[:-1])
        self.cadence = cadence
        self.default_maxlag_days = default_maxlag_days
        self.default_maxlag = default_maxlag_days//cadence
        self.flatten_order = flatten_order
        
        #set private variables for cached acorr calculation
        self._lag = None  #always should be in cadences
        self._ac = None
        self._ac_poly_coeffs = None #polynomial coefficients subtracted out to flatten acorr

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

            #fit and subtract out quadratic
            if self.flatten_order is not None:
                c = np.polyfit(lag, ac, self.flatten_order)
                ac -= np.polyval(c, lag)
                self._ac_poly_coeffs = c

            #smooth AC function
            ac = gaussian_filter(ac, smooth)

            #set private variables for cached calculation
            self._ac = ac
            self._lag = lag
            self._maxlag = maxlag
            self._smooth = smooth

        if days:
            return lag*self.cadence,ac
        else:
            return lag,ac

    def acorr_peaks(self, lookahead=5, days=True, 
                    return_heights=False, **kwargs):
        lag, ac = self.acorr(days=days, **kwargs)
        return peaks_and_lphs(ac, lag, return_heights=return_heights,
                              lookahead=lookahead)
        
    def plot_acorr(self, days=True, smooth=18, maxlag=None,
                   mark_period=False, lookahead=5, fit_npeaks=4,
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

    def make_chunks(self, nchunks, chunksize=300, step=100):
        tmin, tmax = (self.t.min(), self.t.max())
        tlist = [(t, t+chunksize) for t in np.arange(tmin, tmax+step, step)]
        logging.debug('(start, stop) tlist: {}'.format(tlist))
        self.make_subseries(tlist)
                
    def make_subseries(self, tlist, names=None):
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
                                              default_maxlag_days=self.default_maxlag_days)


    @classmethod
    def load_hdf(cls, filename, path=''):
        
        data = pd.read_hdf(filename, '{}/data'.format(path))
        t = np.array(data['t'])
        f = np.array(data['f'])
        mask = np.array(data['mask'])

        new = cls(t,f,mask=mask)

        acorr = pd.read_hdf(filename, '{}/acorr'.format(path))
        new._lag = np.array(acorr['lag'])
        new._ac = np.array(acorr['ac'])

        pgram = pd.read_hdf(filename, '{}/pgram'.format(path))
        new._pers = np.array(pgram['period'])
        new._pgram = np.array(pgram['pgram'])

        #store.close()

        i=1
        has_sub = True
        new.subseries = {}
        while has_sub:
            try:
                name = 'sub{}'.format(i)
                new.subseries[name] = cls.load_hdf(filename, path='{}/{}'.format(path,name))
            except KeyError:
                has_sub = False
            i += 1

        return new



    

class NoPeakError(Exception):
    pass

    

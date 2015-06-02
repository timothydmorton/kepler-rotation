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

from .timeseries import TimeSeries

try:
    KOI_PHOTOMETRY_DIR = os.environ['KOI_PHOTOMETRY_DIR']
except KeyError:
    logging.warning('KOI_PHOTOMETRY_DIR environment variable not defined.')
    KOI_PHOTOMETRY_DIR = '.'

CADENCE = 0.02043423
QTIMESFILE = resource_filename('rotation','data/qStartStop.txt')
QSTART = {}
QSTOP = {}
for line in open(QTIMESFILE):
    line = line.split()
    if line[0] != 'q':
        QSTART[int(line[0])] = float(line[1])
        QSTOP[int(line[0])] = float(line[2])


class Kepler_TimeSeries_Petigura(TimeSeries):
    def __init__(self, koi, folder=KOI_PHOTOMETRY_DIR, 
                 pdc=True, median_window=500, sigclip=7,
                 mask_transit=True, 
                 qtr=None,
                 qmin=3, qmax=15,
                 cadence=CADENCE,
                 default_maxlag_days=200
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

        TimeSeries.__init__(self, t, f, mask=mask, cadence=cadence,
                            default_maxlag=default_maxlag_days//cadence,
                            **kwargs)
        
    def make_subseries(self,qlist=[(3,5),(6,8),(9,11),(12,15)]):
        tlist = [(QSTART[tlo],QSTOP[thi]) for tlo,thi in qlist]
        #names = ['q{}q{}'.format(tlo,thi) for tlo,thi in qlist]
        TimeSeries.make_subseries(self, tlist)
        

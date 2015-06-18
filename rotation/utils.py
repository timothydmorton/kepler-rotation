import numpy as np
import acor
import logging

from .findpeaks import peakdetect


def peaks_and_lphs(y, x=None, lookahead=5, return_heights=False):
    """Returns locations of peaks and corresponding "local peak heights"
    """
    if x is None:
        x = np.arange(len(y))

    maxes, mins = peakdetect(y, x, lookahead=lookahead)
    maxes = np.array(maxes)
    mins = np.array(mins)

    logging.debug('maxes: {0} (shape={0.shape})'.format(maxes))
    logging.debug('mins: {0} (shape={0.shape})'.format(mins))
    
    if len(maxes) == 0:
        logging.warning('No peaks found in acorr; returning empty')
        if return_heights:
            return [], [], []
        else:
            return [], []

    n_maxes = maxes.shape[0]
    n_mins = mins.shape[0]

    if n_maxes==1 and n_mins==1:
        lphs = maxes[0,1] - mins[0,1]

    elif n_maxes == n_mins+1:
        lphs = np.concatenate([[maxes[0,1] - mins[0,1]],
                               ((maxes[1:-1,1] - mins[1:,1]) + (maxes[1:-1,1] - mins[:-1,1]))/2])
    elif n_mins == n_maxes+1:
        lphs = ((maxes[:,1] - mins[1:,1]) + (maxes[:1,1] - mins[:-1,1]))

    elif n_maxes == n_mins:
        if maxes[0,0] < mins[0,0]:
            lphs = np.concatenate([[maxes[0,1] - mins[0,1]],
                                  ((maxes[1:,1] - mins[:-1,1]) + (maxes[1:,1] + mins[1:,1]))/2])
        else:
            lphs = np.concatenate([((maxes[:-1,1] - mins[:-1,1]) + (maxes[:-1,1] - mins[1:,1]))/2.,
                                  [maxes[-1,1] - mins[-1,1]]])


    else:
        raise RuntimeError('No cases satisfied??')
    
        ##if first extremum is a max, remove it:
        #if maxes[0,0] < mins[0,0]:
        #    logging.debug('first extremum is a max; removing')
        #    maxes = maxes[1:,:]
        #    logging.debug('now, maxes: {}'.format(maxes))
        #    logging.debug('now, mins: {}'.format(mins))    

        ##if last extremum is a max, remove it:
        #if maxes[-1,0] > mins[-1,0]:
        #    logging.debug('last extremum is a max; removing')
        #    maxes = maxes[:-1,:]
        #    logging.debug('now, maxes: {}'.format(maxes))
        #    logging.debug('now, mins: {}'.format(mins))    


        #this should always work now?
        #lphs = ((maxes[:,1] - mins[:-1,1]) + (maxes[:,1] - mins[1:,1]))/2.
    
        
    """
    if maxes.shape[0]==1:
        if return_heights:
            return maxes[:,0], [], []
        else:
            return [], []
    
    #calculate "local heights".  First (used to) always be a minimum.
    try: #this if maxes and mins are same length 
        lphs = np.concatenate([((maxes[:-1,1] - mins[:-1,1]) + (maxes[:-1,1] - mins[1:,1]))/2.,
                               np.array([maxes[-1,1]-mins[-1,1]])])
    except ValueError: #this if mins have one more
        try:
            lphs = ((maxes[:,1] - mins[:-1,1]) + (maxes[:,1] - mins[1:,1]))/2.
        except ValueError: # if maxes have one more (drop first max)
            lphs = np.concatenate([((maxes[1:-1,1] - mins[:-1,1]) + (maxes[1:-1,1] - mins[1:,1]))/2.,
                               np.array([maxes[-1,1]-mins[-1,1]])])

    """

    logging.debug('lphs: {}'.format(lphs))

    if return_heights:
        return maxes[:,0], lphs, maxes[:,1]
    else:
        return maxes[:,0], lphs 

    

def acorr_peaks_old(fs, maxlag, mask=None, lookahead=5, smooth=18,
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

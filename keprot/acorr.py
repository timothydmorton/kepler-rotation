import numpy as np
from scipy.ndimage.filters import gaussian_filter

CADENCE = 0.02043423

def acorr(fs, mask=None, maxlag=100//CADENCE, smooth=18):
    """Autocorrelation function

    Where mask is True, flux set to zero & de-weighted in autocorrelation sum
    """
    if mask is None:
        mask = np.zeros(t.shape)

    maxlag_cad = maxlag/CADENCE
    lags = 1 + np.arange(maxlag_cad-1)
    f[mask] = 0
    weights = np.ones(f.shape)
    weights[mask] = 0
        
    sums = np.zeros(len(lags))
    for i,l in enumerate(lags):
        prod = f[:-l]*f[l:]
        ws = weights[:-l]*weights[l:]
        sums[i] = (prod*ws).sum()
        
    sums /= ((fs - fs.mean())**2).sum()

    if smooth is not None:
        smoothed = gaussian_filter(corr, smooth)
    else:
        smoothed = sums

    return lags, smoothed
    
def acorr_peaks(fs, mask=None, npeaks=10, lookahead=5, smooth=18)
    """Returns positions of acorr peaks, with corresponding local heights.
    """
    lag, corr = acorr(fs, mask=mask, maxlag=maxlag)
    maxes, mins = peakdetect(corr, lag, lookahead=lookahead)
    maxes = np.array(maxes)
    mins = np.array(mins)
    
    try:
        hs = np.concatenate([((maxes[:-1,1] - mins[:-1,1]) + (maxes[:-1,1] - mins[1:,1]))/2.,
                             np.array([maxes[-1,1]-mins[-1,1]])])
    except ValueError:
        hs = ((maxes[:,1] - mins[:-1,1]) + (maxes[:,1] - mins[1:,1]))/2.

    return maxes[:,0],hs


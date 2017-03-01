# AFP_Recovery_with_Diffusion_Processes_in_Equity_Market

Run `findLargestMatrix.py` to create a list of `(size, date)` pair for each of *call* and *put*, sorted by `size` - the number of valid numeric values in the price matrix (days till maturity vs. strike).

Run `loadDataOfDate.py` and use the API function `loadDataOfDate(date, option_type)` to extract price matrix of specified date and option type. Matrix is in format of DataFrame with index being days till maturity, and column names being strike price * 1000.


## Interpolation
Currently, only **call option** data is available.
Only 1 sample data is uploaded to Github (date of 20160429). The rest can be found in Google Drive.

<pre><code>
from interpolateSurface import *

# ts, Ks are arrays of input coordinates to be evaluated
# date is integer, in format of yyyymmdd

# V, Vk, Vkk, Vt are matrix of len(t) x len(K)
# t and K are actual inputs used for evaluation,
#  might be different from input ts and Ks if inputs are out of bound.

V, Vk, Vkk, Vt, t, K = obtainSurface(ts, Ks, date, option_type='call')

# 3D Plot
plotSurface(t, K, V, ['t', 'K', 'V'])
plotSurface(t, K, Vk, ['t', 'K', 'Vk'])
plotSurface(t, K, Vkk, ['t', 'K', 'Vkk'])
plotSurface(t, K, Vt, ['t', 'K', 'Vt'])
</code></pre>

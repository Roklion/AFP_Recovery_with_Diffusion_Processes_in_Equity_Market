# AFP_Recovery_with_Diffusion_Processes_in_Equity_Market

Run `findLargestMatrix.py` to create a list of `(size, date)` pair for each of *call* and *put*, sorted by `size` - the number of valid numeric values in the price matrix (days till maturity vs. strike).

Run `loadDataOfDate.py` and use the API function `loadDataOfDate(date, option_type)` to extract price matrix of specified date and option type. Matrix is in format of DataFrame with index being days till maturity, and column names being strike price * 1000.

# How to read files.
For example, if you want to read `filename.h5`
```Python
import pandas as pd
df = pd.read_hdf("filename.h5", key="data")
```
NOTE: **key is always "data" for all hdf5 files **.

# Here is a short description about the data

| Filename       | Description                                                      |
| -------------- | -----------------------------------------------------------------|
| "daily_pv.h5"  | Split-adjusted daily price and volume data.                      |


# For different data, We have some basic knowledge for them

## Daily price and volume data
All prices are **split-adjusted** to the most recent split date. This means:
- Stock splits (e.g., 10:1, 7:1) are accounted for
- Historical prices are normalized to post-split levels
- No artificial "drops" from split events

$open: open price of the stock on that day (split-adjusted).
$close: close price of the stock on that day (split-adjusted).
$high: high price of the stock on that day (split-adjusted).
$low: low price of the stock on that day (split-adjusted).
$volume: volume of the stock on that day.
$factor: always 1.0 (split adjustment has been applied to prices).
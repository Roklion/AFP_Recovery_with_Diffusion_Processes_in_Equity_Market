###############################################################################
##
## @author: Yao Dong Yu
##
## Description: Smooth transformed option prices vs. strike function using local
##              polynomial smoothing, and create up-to 4th derivatives of the
##              regression function
##              Method is based on Ait-Sahalia and Duarte (2003)
##
## Require: outputs of transform_main() of fitOptionPriceSurface.py:
##              transformed/ms_call_*.csv
##              transformed/ms_put_*.csv
##
###############################################################################
source('./smooth_K_impl.R')

# Some constants
K_step = 1
K_order = 4

input.path <- './data/transformed/'
output.path <- './data/local_poly/'
input.files <- dir(input.path, pattern='.csv')

# Each file represent 1 time snapshot
for(f in 1:length(input.files)) {
    smooth_K_impl(paste0(input.path, input.files[f]),
                  paste0(output.path, input.files[f]),
                  K_step, K_order)
}


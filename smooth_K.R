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
library(locpol)

# Some constants
K_step = 1
K_order = 4

input.path <- './data/transformed/'
output.path <- './data/local_poly/'
input.files <- dir(input.path, pattern='.csv')

# Each file represent 1 time snapshot
for(f in 1:length(input.files)) {
    df = read.csv(paste0(input.path, input.files[f]), header=TRUE, check.names=FALSE)
    
    # Isolate time to maturity column
    ts = df[, 1]
    df[, 1] <- NULL
    
    strikes = as.numeric(colnames(df))
    
    result <- data.frame()
    # Each row(time to maturity)
    for (i in 1:nrow(df)) {
        row <- df[i, ]
        t <- ts[i]
        
        # Remove NA's
        valid_idx = !is.na(row)
        valid_prices <- row[valid_idx]
        valid_Ks <- strikes[valid_idx]
        
        # Enough data points
        if(length(valid_Ks) > 10) {
            x_c <- seq(min(valid_Ks), max(valid_Ks), by=K_step)
            
            #bw <- thumbBw(valid_Ks, valid_prices, K_order, EpaK)
            #bw <- pluginBw(valid_Ks, valid_prices, K_order, EpaK)
            #print(bw)
            # Hard code bandwidth for now
            bw <- 400
            
            res <- locPolSmootherC(valid_Ks, valid_prices, x_c, bw, K_order, EpaK)
            # Truncate tails where convexity condition does not meet
            res <- res[res$beta2 >= 0, ]
            # Remove NA's
            res <- res[apply(is.na(res), c(1), sum) == 0, ]
            res$t <- t
            
            result <- rbind(result, res)
        }
    }
    
    write.csv(result, paste0(output.path, input.files[f]), row.names=FALSE)
}


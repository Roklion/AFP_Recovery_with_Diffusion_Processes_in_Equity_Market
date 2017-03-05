###############################################################################
##
## @author: Yao Dong Yu
##
## Description: Detail implementation of smooth_K.R, see also description of
##                  smooth_K.R
##
###############################################################################
library(locpol, lib.loc = "C:/Users/kcfef/Documents/R/win-library/3.3")

smooth_K_impl <- function(in.path, out.path, K_step, K_order, bandwidth=400, fix_violation=TRUE) {
    
    df = read.csv(in.path, header=TRUE, check.names=FALSE)
    
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
            
            # Hard code bandwidth
            #bw <- thumbBw(valid_Ks, valid_prices, K_order, EpaK)
            #bw <- pluginBw(valid_Ks, valid_prices, K_order, EpaK)
            #print(bw)
            
            res <- locPolSmootherC(valid_Ks, valid_prices, x_c, bandwidth, K_order, EpaK)
            if(fix_violation == TRUE) {
                # Truncate tails where convexity condition does not meet
                res <- res[res$beta2 >= 0, ]
            }
            # Remove NA's
            res <- res[apply(is.na(res), c(1), sum) == 0, ]
            res$t <- t
            
            result <- rbind(result, res)
        }
    }
    
    write.csv(result, out.path, row.names=FALSE)
}
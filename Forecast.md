Forecast
================

## Forecast

Goal and methods are to explore techniques of forecast in the field of
financial markets and major drawbacks of such statistical methods.

## Code

The library used for the project are listed below:

- library(dplyr)

- library(fpp2)

- library(fpp3)

- library(ggplot2)

- library(rugarch)

In the fpp2 is possible to find time-series for study purposes and the
forecasting tools.

## Exploratory analysis

Given the historical data provided by Yahoo Finance we get the last 5
years price of the popular S&P500. Studies can be done for many other
securities. The repository include price quotation from May 19, 2019 -
May 19, 2024 of the S&P500 and the Nasdq.

    ##         Date    Open    High     Low   Close Adj.Close     Volume
    ## 1 2019-05-20 2841.94 2853.86 2831.29 2840.23   2840.23 3293750000
    ## 2 2019-05-21 2854.02 2868.88 2854.02 2864.36   2864.36 3223050000
    ## 3 2019-05-22 2856.06 2865.47 2851.11 2856.27   2856.27 3194000000
    ## 4 2019-05-23 2836.70 2836.70 2805.49 2822.24   2822.24 3899320000
    ## 5 2019-05-24 2832.41 2841.36 2820.19 2826.06   2826.06 2889230000
    ## 6 2019-05-28 2830.03 2840.51 2801.58 2802.39   2802.39 4146980000

There are several reasons why adjusted prices are often preferred over
raw prices for studying time series data with ARIMA models:

**1. Account for Corporate Actions:**

- Raw stock prices can be affected by corporate actions like stock
  splits, dividends, and rights offerings. These events change the
  number of shares outstanding, impacting the raw price but not
  necessarily the underlying value of the company.

- Adjusted prices incorporate these corporate actions, reflecting the
  true value of the stock over time. This is crucial for ARIMA models
  that analyze trends and patterns in the data.

**2. Improved Stationarity:**

- ARIMA models perform best with stationary data, meaning the
  statistical properties (mean, variance) don’t change significantly
  over time.

- Corporate actions can introduce non-stationarity in raw prices due to
  sudden jumps or dips. Adjusted prices help mitigate this effect by
  smoothing out the impact of these events.

**3. Comparability:**

- When analyzing price movements over extended periods, using adjusted
  prices ensures a more accurate comparison. This is because you’re
  comparing the intrinsic value of the stock over time, not just the raw
  price fluctuations.

**4. Consistency with Financial Analysis:**

- Financial analysts typically use adjusted prices for their
  calculations and valuations. By using adjusted prices in ARIMA
  models, you align your analysis with standard financial practices.

**Here are some additional points to consider:**

- While adjusted prices are generally preferred, there might be specific
  research questions where raw prices are relevant (e.g., studying the
  immediate impact of a stock split).

- The type of adjustment applied to the price will depend on the
  specific corporate action.

``` r
dates=as.Date(data$Date)
prices=data$Adj.Close

plot(dates, prices, type="l", main="Adjusted Close price S&P500 in US $")
```

![](Forecast_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

Is possible to identify a upwards trends that is not a characteristic of
a time-serie with seasonality. Seasonal ts refers to recurring patterns
within a specific time period, typically within a year
(e.g.,monthly, quarterly). but seams impossible to detect seasonality.

``` r
arima=auto.arima(prices, seasonal = FALSE)
summary(arima)
```

    ## Series: prices 
    ## ARIMA(0,1,2) with drift 
    ## 
    ## Coefficients:
    ##           ma1     ma2   drift
    ##       -0.0955  0.0547  1.9573
    ## s.e.   0.0281  0.0287  1.2485
    ## 
    ## sigma^2 = 2136:  log likelihood = -6605.96
    ## AIC=13219.92   AICc=13219.95   BIC=13240.46
    ## 
    ## Training set error measures:
    ##                       ME     RMSE      MAE         MPE      MAPE      MASE
    ## Training set 0.002490136 46.14639 32.86629 -0.01090744 0.8751402 0.9974752
    ##                      ACF1
    ## Training set -0.001829568

``` r
checkresiduals(arima)
```

![](Forecast_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

    ## 
    ##  Ljung-Box test
    ## 
    ## data:  Residuals from ARIMA(0,1,2) with drift
    ## Q* = 57.035, df = 8, p-value = 1.773e-09
    ## 
    ## Model df: 2.   Total lags used: 10

- **ARIMA(0,1,2):** This indicates an ARIMA model with:

  - No autoregressive terms (AR=0), meaning the prediction doesn’t rely
    on past values of the price series.

  - A moving average term of order 1 (MA1), suggesting the model
    considers the previous error term (difference between predicted and
    actual value) in its forecast.

  - A moving average term of order 2 (MA2), meaning it also takes into
    account the error term from two periods back.

- **Drift:** The presence of drift signifies a long-term linear trend in
  the price series.

  ``` r
  log_return= diff(log(prices))
  plot(log_return, type="l", main="Log returns S&P500")
  ```

  ![](Forecast_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

  ``` r
  plot(forecast(arima, h=50))
  ```

  ![](Forecast_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->

With the transformations is possible to get a stationary ts which wasn’t
the case with the previous state of data.

Calculating the logarithm of the index returns can improve the study of
the ts under the aspects of:

**Stationarity:** Financial time series data often exhibits
non-stationarity, meaning the statistical properties (mean, variance)
change over time. Log returns help achieve stationarity by focusing on
the proportional changes in prices.

**Heteroskedasticity:** meaning the variance is not constant over
time. Log returns can lead to more consistent variance and improving the
performance of some models.

``` r
acf(log_return)
```

![](Forecast_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
acf(log_return^2)
```

![](Forecast_files/figure-gfm/unnamed-chunk-5-2.png)<!-- -->

``` r
plot(log_return, type="l", main="Log returns S&P500")
```

![](Forecast_files/figure-gfm/unnamed-chunk-5-3.png)<!-- -->

As shown in the the Auto-correlation function, the data distribution has
an evident lag .

Log returns at the power of 2 can help highlighting volatility cluster
over time as the returns are even more correlated with previous
observation. As a matter of fact this findings violate the assumption of
Arima models:

- **Stationarity:** means that the mean, variance, and autocorrelation
  of the time series data are constant over time.

We try other models more suitable for our purpose.

``` r
install.packages("rugarch")
```

    ## Installing package into '/cloud/lib/x86_64-pc-linux-gnu-library/4.4'
    ## (as 'lib' is unspecified)

    ## Warning: unable to access index for repository [http://cran.us.r-project.org](http://cran.us.r-project.org)/src/contrib:
    ##   cannot open URL '[http://cran.us.r-project.org](http://cran.us.r-project.org)/src/contrib/PACKAGES'

    ## Warning: package 'rugarch' is not available for this version of R
    ## 
    ## A version of this package for your version of R might be available elsewhere,
    ## see the ideas at
    ## https://cran.r-project.org/doc/manuals/r-patched/R-admin.html#Installing-packages

``` r
library(rugarch)
```

    ## Loading required package: parallel

    ## 
    ## Attaching package: 'rugarch'

    ## The following object is masked from 'package:fabletools':
    ## 
    ##     report

    ## The following object is masked from 'package:stats':
    ## 
    ##     sigma

``` r
gar= ugarchspec(variance.model = list(model="sGARCH",garchOrder=c(1,1)),
                                   mean.model = list(armaOrder=c(0,0)),
                                   distribution.model = "norm")
# fittiamo il modello
gar= ugarchfit(spec = gar,data = log_return)
gar
```

    ## 
    ## *---------------------------------*
    ## *          GARCH Model Fit        *
    ## *---------------------------------*
    ## 
    ## Conditional Variance Dynamics    
    ## -----------------------------------
    ## GARCH Model  : sGARCH(1,1)
    ## Mean Model   : ARFIMA(0,0,0)
    ## Distribution : norm 
    ## 
    ## Optimal Parameters
    ## ------------------------------------
    ##         Estimate  Std. Error  t value Pr(>|t|)
    ## mu      0.000920    0.000243   3.7857 0.000153
    ## omega   0.000004    0.000003   1.4243 0.154366
    ## alpha1  0.174253    0.025545   6.8214 0.000000
    ## beta1   0.800685    0.033122  24.1737 0.000000
    ## 
    ## Robust Standard Errors:
    ##         Estimate  Std. Error  t value Pr(>|t|)
    ## mu      0.000920    0.000246  3.73951 0.000184
    ## omega   0.000004    0.000014  0.31858 0.750045
    ## alpha1  0.174253    0.034669  5.02613 0.000001
    ## beta1   0.800685    0.111168  7.20247 0.000000
    ## 
    ## LogLikelihood : 3983.287 
    ## 
    ## Information Criteria
    ## ------------------------------------
    ##                     
    ## Akaike       -6.3264
    ## Bayes        -6.3100
    ## Shibata      -6.3264
    ## Hannan-Quinn -6.3202
    ## 
    ## Weighted Ljung-Box Test on Standardized Residuals
    ## ------------------------------------
    ##                         statistic p-value
    ## Lag[1]                     0.2204  0.6387
    ## Lag[2*(p+q)+(p+q)-1][2]    0.2297  0.8362
    ## Lag[4*(p+q)+(p+q)-1][5]    1.3345  0.7805
    ## d.o.f=0
    ## H0 : No serial correlation
    ## 
    ## Weighted Ljung-Box Test on Standardized Squared Residuals
    ## ------------------------------------
    ##                         statistic p-value
    ## Lag[1]                     0.3117  0.5766
    ## Lag[2*(p+q)+(p+q)-1][5]    1.0101  0.8574
    ## Lag[4*(p+q)+(p+q)-1][9]    1.2157  0.9755
    ## d.o.f=2
    ## 
    ## Weighted ARCH LM Tests
    ## ------------------------------------
    ##             Statistic Shape Scale P-Value
    ## ARCH Lag[3]    0.1491 0.500 2.000  0.6994
    ## ARCH Lag[5]    0.1780 1.440 1.667  0.9706
    ## ARCH Lag[7]    0.2231 2.315 1.543  0.9962
    ## 
    ## Nyblom stability test
    ## ------------------------------------
    ## Joint Statistic:  3.2317
    ## Individual Statistics:              
    ## mu     0.06939
    ## omega  0.16121
    ## alpha1 0.15376
    ## beta1  0.14574
    ## 
    ## Asymptotic Critical Values (10% 5% 1%)
    ## Joint Statistic:          1.07 1.24 1.6
    ## Individual Statistic:     0.35 0.47 0.75
    ## 
    ## Sign Bias Test
    ## ------------------------------------
    ##                    t-value    prob sig
    ## Sign Bias           2.1214 0.03409  **
    ## Negative Sign Bias  0.3833 0.70159    
    ## Positive Sign Bias  0.2138 0.83071    
    ## Joint Effect        7.6665 0.05343   *
    ## 
    ## 
    ## Adjusted Pearson Goodness-of-Fit Test:
    ## ------------------------------------
    ##   group statistic p-value(g-1)
    ## 1    20     47.91     0.000264
    ## 2    30     65.34     0.000128
    ## 3    40     69.12     0.002086
    ## 4    50     78.65     0.004572
    ## 
    ## 
    ## Elapsed time : 0.3380864

``` r
# sGARCH(1,1)
# grafici diagnostici
plot(gar, which=8)
```

![](Forecast_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
## ripeto l'analisi con un modello con innovazioni t di student
# specifichiamo il modello
gar_t= ugarchspec(variance.model = list(model="sGARCH",garchOrder=c(1,1)),
                                   mean.model = list(armaOrder=c(0,0)),
                                   distribution.model = "std")
# fittiamo il modello
gar_t= ugarchfit(spec = gar_t, data = log_return)

# grafici diagnostici
plot(gar_t,which=8)
```

![](Forecast_files/figure-gfm/unnamed-chunk-6-2.png)<!-- -->

``` r
plot(gar_t@fit$z,type ="l")
```

![](Forecast_files/figure-gfm/unnamed-chunk-6-3.png)<!-- -->

``` r
## calcolo del Value at Risk al 99%
alpha = 0.99
## con gaussiana
VaR_gauss= -quantile(gar, probs = 1-alpha)
# plot(VaR_gauss)
plot(log_return,type = "l", main = "VaR99% GARCH(1,1)Gauss")+
lines(-as.numeric(VaR_gauss),col = "red")
```

![](Forecast_files/figure-gfm/unnamed-chunk-6-4.png)<!-- -->

    ## integer(0)

``` r
# verifichiamo il numero di violazioni
mean(log_return < -VaR_gauss)
```

    ## [1] 0.02146264

``` r
## con Student's t
VaR_t= -quantile(gar_t,probs = 1-alpha)
# plot(VaR_t)
plot(log_return,type = "l", main = "VaR99% GARCH(1,1)t")+
lines(-as.numeric(VaR_t),col = "red")
```

![](Forecast_files/figure-gfm/unnamed-chunk-6-5.png)<!-- -->

    ## integer(0)

``` r
# verifichiamo il numero di violazioni
```

## **Implications of VaR with GARCH Models**

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models
are used to estimate the volatility of financial returns, which is
crucial for accurate VaR calculations. Unlike standard volatility
models, GARCH models account for the fact that volatility is often not
constant over time but varies, exhibiting periods of high and low
volatility.

VaR provides a quantifiable measure of potential loss in a portfolio,
and incorporating GARCH models into VaR calculations allows for a more
dynamic and realistic assessment of risk by accounting for time-varying
volatility.

``` r
#Value at Risk
a=0.05
quantile(log_return,a)
```

    ##          5% 
    ## -0.01865175

``` r
# This might become more accurate for risk management assessing with alpha=0.01

qplot(log_return , geom = 'histogram') + geom_histogram(fill='lightblue') +
geom_histogram(aes(log_return[log_return < quantile(log_return, a)]), fill='red') + labs(x = 'Daily Returns S&P500')
```

    ## Warning: `qplot()` was deprecated in ggplot2 3.4.0.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
    ## generated.

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](Forecast_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

## Benchmarking Naïve

``` r
naive=naive(prices, h = 20)  # h = 20 days
# Plot forecasts
autoplot(naive)
```

![](Forecast_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
# Plot forecast with an xlim
autoplot(naive, include=200)
```

![](Forecast_files/figure-gfm/unnamed-chunk-8-2.png)<!-- -->

As for **Hewamalage et al.(2022) practitioners can check the goodness of
a model benchmarking results with** Naïve **model, simple Random Walk
equivalent to** ARIMA(0,1,0).

``` r
library(fpp3)

data$Date <- as.Date(data$Date)
tsib= as_tsibble(data, index = Date)

tsib_filled <- tsib %>%
  fill_gaps() %>%
  mutate(Adj.Close = ifelse(is.na(Adj.Close), mean(Adj.Close, na.rm = TRUE), Adj.Close))


#fit=model(NAIVE(tsib$Adj.Close))
#autoplot(fit)
?NAIVE
```

## Forecast using Neural Networks

``` r
fitnn=nnetar(prices, lambda = 0)
autoplot(forecast(fitnn, h=20, col="red"))
```

![](Forecast_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
sim= ts(matrix(0, nrow=30L, ncol=9L),
start=end(prices)[1L]+1L)

for(i in seq(9))
  sim[,i]= simulate(fitnn, nsim=30L)
p= autoplot(ts(prices))
p + autolayer(sim, series = "Forecast")
```

    ## For a multivariate time series, specify a seriesname for each time series. Defaulting to column names.

![](Forecast_files/figure-gfm/unnamed-chunk-10-2.png)<!-- -->

``` r
fcast= forecast(fitnn, PI=TRUE, h=20)
autoplot(fcast)
```

![](Forecast_files/figure-gfm/unnamed-chunk-10-3.png)<!-- -->

Seems possible to try to model the time series of adjusted prices, with
lambda=0 is performed a Box-Cox transformation. It’s possible to create
a cycle that iterate the forecast for several times and distribute the
results plotting and graphically seeing the results.

\################

Figure 11.15: Forecasts with prediction intervals for the annual sunspot
data. Prediction intervals are computed using simulated future sample
paths.

Because it is a little slow, `PI=FALSE` is the default, so prediction
intervals are not computed unless requested. The `npaths` argument
in `forecast()` controls how many simulations are done (default 1000).
By default, the errors are drawn from a normal distribution.
The `bootstrap` argument allows the errors to be “bootstrapped” (i.e.,
randomly drawn from the historical errors)

\################

After modelization we try backtest with real observation of the index
from May to June.

With the function tsoutliers() cosidered a TS composed of: - Trend -
Seasonality - Residuals

It’s possible to leave out of the equation a series of outliers and get
a clean time series with the function is possible to replace missing
through replacement and values coming from interpolation of data.

This may be particularly important if people want to evaluate and study
VIX due to the fact that Yahoo Finance and other data provider don’t
give clean data.

With indices and stocks we know that markets are usually oper from
monday to friday afternoon, there may be bank holidays. Other than that
is news, report about quarter prifit usually comes off in the last
market day in order to avoid rapid changes and make it possible for
analysts to process information, data and metadata.

``` r
tsoutliers(prices)
```

    ## $index
    ##  [1] 206 208 209 210 211 212 213 214 215 220
    ## 
    ## $replacements
    ##  [1] 2726.200 2702.026 2693.031 2684.037 2675.042 2666.048 2657.053 2648.059
    ##  [9] 2639.065 2555.745

``` r
tsclean(prices)
```

    ##    [1] 2840.230 2864.360 2856.270 2822.240 2826.060 2802.390 2783.020 2788.860
    ##    [9] 2752.060 2744.450 2803.270 2826.150 2843.490 2873.340 2886.730 2885.720
    ##   [17] 2879.840 2891.640 2886.980 2889.670 2917.750 2926.460 2954.180 2950.460
    ##   [25] 2945.350 2917.380 2913.780 2924.920 2941.760 2964.330 2973.010 2995.820
    ##   [33] 2990.410 2975.950 2979.630 2993.070 2999.910 3013.770 3014.300 3004.040
    ##   [41] 2984.420 2995.110 2976.610 2985.030 3005.470 3019.560 3003.670 3025.860
    ##   [49] 3020.970 3013.180 2980.380 2953.560 2932.050 2844.740 2881.770 2883.980
    ##   [57] 2938.090 2918.650 2882.700 2926.320 2840.600 2847.600 2888.680 2923.650
    ##   [65] 2900.510 2924.430 2922.950 2847.110 2878.380 2869.160 2887.940 2924.580
    ##   [73] 2926.460 2906.270 2937.780 2976.000 2978.710 2978.430 2979.390 3000.930
    ##   [81] 3009.570 3007.390 2997.960 3005.700 3006.730 3006.790 2992.070 2991.780
    ##   [89] 2966.600 2984.870 2977.620 2961.790 2976.740 2940.250 2887.610 2910.630
    ##   [97] 2952.010 2938.790 2893.060 2919.400 2938.130 2970.270 2966.150 2995.680
    ##  [105] 2989.690 2997.950 2986.200 3006.720 2995.990 3004.520 3010.290 3022.550
    ##  [113] 3039.420 3036.890 3046.770 3037.560 3066.910 3078.270 3074.620 3076.780
    ##  [121] 3085.180 3093.080 3087.010 3091.840 3094.040 3096.630 3120.460 3122.030
    ##  [129] 3120.180 3108.460 3103.540 3110.290 3133.640 3140.520 3153.630 3140.980
    ##  [137] 3113.870 3093.200 3112.760 3117.430 3145.910 3135.960 3132.520 3141.630
    ##  [145] 3168.570 3168.800 3191.450 3192.520 3191.140 3205.370 3221.220 3224.010
    ##  [153] 3223.380 3239.910 3240.020 3221.290 3230.780 3257.850 3234.850 3246.280
    ##  [161] 3237.180 3253.050 3274.700 3265.350 3288.130 3283.150 3289.290 3316.810
    ##  [169] 3329.620 3320.790 3321.750 3325.540 3295.470 3243.630 3276.240 3273.400
    ##  [177] 3283.660 3225.520 3248.920 3297.590 3334.690 3345.780 3327.710 3352.090
    ##  [185] 3357.750 3379.450 3373.940 3380.160 3370.290 3386.150 3373.230 3337.750
    ##  [193] 3225.890 3128.210 3116.390 2978.760 2954.220 3090.230 3003.370 3130.120
    ##  [201] 3023.940 2972.370 2746.560 2882.230 2741.380 2726.200 2711.020 2702.026
    ##  [209] 2693.031 2684.037 2675.042 2666.048 2657.053 2648.059 2639.065 2630.070
    ##  [217] 2541.470 2626.650 2584.590 2555.745 2526.900 2488.650 2663.680 2659.410
    ##  [225] 2749.980 2789.820 2761.630 2846.060 2783.360 2799.550 2874.560 2823.160
    ##  [233] 2736.560 2799.310 2797.800 2836.740 2878.480 2863.390 2939.510 2912.430
    ##  [241] 2830.710 2842.740 2868.440 2848.420 2881.190 2929.800 2930.190 2870.120
    ##  [249] 2820.000 2852.500 2863.700 2953.910 2922.940 2971.610 2948.510 2955.450
    ##  [257] 2991.770 3036.130 3029.730 3044.310 3055.730 3080.820 3122.870 3112.350
    ##  [265] 3193.930 3232.390 3207.180 3190.140 3002.100 3041.310 3066.590 3124.740
    ##  [273] 3113.490 3115.340 3097.740 3117.860 3131.290 3050.330 3083.760 3009.050
    ##  [281] 3053.240 3100.290 3115.860 3130.010 3179.720 3145.320 3169.940 3152.050
    ##  [289] 3185.040 3155.220 3197.520 3226.560 3215.570 3224.730 3251.840 3257.300
    ##  [297] 3276.020 3235.660 3215.630 3239.410 3218.440 3258.440 3246.220 3271.120
    ##  [305] 3294.610 3306.510 3327.770 3349.160 3351.280 3360.470 3333.690 3380.350
    ##  [313] 3373.430 3372.850 3381.990 3389.780 3374.850 3385.510 3397.160 3431.280
    ##  [321] 3443.620 3478.730 3484.550 3508.010 3500.310 3526.650 3580.840 3455.060
    ##  [329] 3426.960 3331.840 3398.960 3339.190 3340.970 3383.540 3401.200 3385.490
    ##  [337] 3357.010 3319.470 3281.060 3315.570 3236.920 3246.590 3298.460 3351.600
    ##  [345] 3335.470 3363.000 3380.800 3348.420 3408.600 3360.970 3419.440 3446.830
    ##  [353] 3477.140 3534.220 3511.930 3488.670 3483.340 3483.810 3426.920 3443.120
    ##  [361] 3435.560 3453.490 3465.390 3400.970 3390.680 3271.030 3310.110 3269.960
    ##  [369] 3310.240 3369.160 3443.440 3510.450 3509.440 3550.500 3545.530 3572.660
    ##  [377] 3537.010 3585.150 3626.910 3609.530 3567.790 3581.870 3557.540 3577.590
    ##  [385] 3635.410 3629.650 3638.350 3621.630 3662.450 3669.010 3666.720 3699.120
    ##  [393] 3691.960 3702.250 3672.820 3668.100 3663.460 3647.490 3694.620 3701.170
    ##  [401] 3722.480 3709.410 3694.920 3687.260 3690.010 3703.060 3735.360 3727.040
    ##  [409] 3732.040 3756.070 3700.650 3726.860 3748.140 3803.790 3824.680 3799.610
    ##  [417] 3801.190 3809.840 3795.540 3768.250 3798.910 3851.850 3853.070 3841.470
    ##  [425] 3855.360 3849.620 3750.770 3787.380 3714.240 3773.860 3826.310 3830.170
    ##  [433] 3871.740 3886.830 3915.590 3911.230 3909.880 3916.380 3934.830 3932.590
    ##  [441] 3931.330 3913.970 3906.710 3876.500 3881.370 3925.430 3829.340 3811.150
    ##  [449] 3901.820 3870.290 3819.720 3768.470 3841.940 3821.350 3875.440 3898.810
    ##  [457] 3939.340 3943.340 3968.940 3962.710 3974.120 3915.460 3913.100 3940.590
    ##  [465] 3910.520 3889.140 3909.520 3974.540 3971.090 3958.550 3972.890 4019.870
    ##  [473] 4077.910 4073.940 4079.950 4097.170 4128.800 4127.990 4141.590 4124.660
    ##  [481] 4170.420 4185.470 4163.260 4134.940 4173.420 4134.980 4180.170 4187.620
    ##  [489] 4186.720 4183.180 4211.470 4181.170 4192.660 4164.660 4167.590 4201.620
    ##  [497] 4232.600 4188.430 4152.100 4063.040 4112.500 4173.850 4163.290 4127.830
    ##  [505] 4115.680 4159.120 4155.860 4197.050 4188.130 4195.990 4200.880 4204.110
    ##  [513] 4202.040 4208.120 4192.850 4229.890 4226.520 4227.260 4219.550 4239.180
    ##  [521] 4247.440 4255.150 4246.590 4223.700 4221.860 4166.450 4224.790 4246.440
    ##  [529] 4241.840 4266.490 4280.700 4290.610 4291.800 4297.500 4319.940 4352.340
    ##  [537] 4343.540 4358.130 4320.820 4369.550 4384.630 4369.210 4374.300 4360.030
    ##  [545] 4327.160 4258.490 4323.060 4358.690 4367.480 4411.790 4422.300 4401.460
    ##  [553] 4400.640 4419.150 4395.260 4387.160 4423.150 4402.660 4429.100 4436.520
    ##  [561] 4432.350 4436.750 4442.410 4460.830 4468.000 4479.710 4448.080 4400.270
    ##  [569] 4405.800 4441.670 4479.530 4486.230 4496.190 4470.000 4509.370 4528.790
    ##  [577] 4522.680 4524.090 4536.950 4535.430 4520.030 4514.070 4493.280 4458.580
    ##  [585] 4468.730 4443.050 4480.700 4473.750 4432.990 4357.730 4354.190 4395.640
    ##  [593] 4448.980 4455.480 4443.110 4352.630 4359.460 4307.540 4357.040 4300.460
    ##  [601] 4345.720 4363.550 4399.760 4391.340 4361.190 4350.650 4363.800 4438.260
    ##  [609] 4471.370 4486.460 4519.630 4536.190 4549.780 4544.900 4566.480 4574.790
    ##  [617] 4551.680 4596.420 4605.380 4613.670 4630.650 4660.570 4680.060 4697.530
    ##  [625] 4701.700 4685.250 4646.710 4649.270 4682.850 4682.800 4700.900 4688.670
    ##  [633] 4704.540 4697.960 4682.940 4690.700 4701.460 4594.620 4655.270 4567.000
    ##  [641] 4513.040 4577.100 4538.430 4591.670 4686.750 4701.210 4667.450 4712.020
    ##  [649] 4668.970 4634.090 4709.850 4668.670 4620.640 4568.020 4649.230 4696.560
    ##  [657] 4725.790 4791.190 4786.350 4793.060 4778.730 4766.180 4796.560 4793.540
    ##  [665] 4700.580 4696.050 4677.030 4670.290 4713.070 4726.350 4659.030 4662.850
    ##  [673] 4577.110 4532.760 4482.730 4397.940 4410.130 4356.450 4349.930 4326.510
    ##  [681] 4431.850 4515.550 4546.540 4589.380 4477.440 4500.530 4483.870 4521.540
    ##  [689] 4587.180 4504.080 4418.640 4401.670 4471.070 4475.010 4380.260 4348.870
    ##  [697] 4304.760 4225.500 4288.700 4384.650 4373.940 4306.260 4386.540 4363.490
    ##  [705] 4328.870 4201.090 4170.700 4277.880 4259.520 4204.310 4173.110 4262.450
    ##  [713] 4357.860 4411.670 4463.120 4461.180 4511.610 4456.240 4520.160 4543.060
    ##  [721] 4575.520 4631.600 4602.450 4530.410 4545.860 4582.640 4525.120 4481.150
    ##  [729] 4500.210 4488.280 4412.530 4397.450 4446.590 4392.590 4391.690 4462.210
    ##  [737] 4459.450 4393.660 4271.780 4296.120 4175.200 4183.960 4287.500 4131.930
    ##  [745] 4155.380 4175.480 4300.170 4146.870 4123.340 3991.240 4001.050 3935.180
    ##  [753] 3930.080 4023.890 4008.010 4088.850 3923.680 3900.790 3901.360 3973.750
    ##  [761] 3941.480 3978.730 4057.840 4158.240 4132.150 4101.230 4176.820 4108.540
    ##  [769] 4121.430 4160.680 4115.770 4017.820 3900.860 3749.630 3735.480 3789.990
    ##  [777] 3666.770 3674.840 3764.790 3759.890 3795.730 3911.740 3900.110 3821.550
    ##  [785] 3818.830 3785.380 3825.330 3831.390 3845.080 3902.620 3899.380 3854.430
    ##  [793] 3818.800 3801.780 3790.380 3863.160 3830.850 3936.690 3959.900 3998.950
    ##  [801] 3961.630 3966.840 3921.050 4023.610 4072.430 4130.290 4118.630 4091.190
    ##  [809] 4155.170 4151.940 4145.190 4140.060 4122.470 4210.240 4207.270 4280.150
    ##  [817] 4297.140 4305.200 4274.040 4283.740 4228.480 4137.990 4128.730 4140.770
    ##  [825] 4199.120 4057.660 4030.610 3986.160 3955.000 3966.850 3924.260 3908.190
    ##  [833] 3979.870 4006.180 4067.360 4110.410 3932.690 3946.010 3901.350 3873.330
    ##  [841] 3899.890 3855.930 3789.930 3757.990 3693.230 3655.040 3647.290 3719.040
    ##  [849] 3640.470 3585.620 3678.430 3790.930 3783.280 3744.520 3639.660 3612.390
    ##  [857] 3588.840 3577.030 3669.910 3583.070 3677.950 3719.980 3695.160 3665.780
    ##  [865] 3752.750 3797.340 3859.110 3830.600 3807.300 3901.060 3871.980 3856.100
    ##  [873] 3759.690 3719.890 3770.550 3806.800 3828.110 3748.570 3956.370 3992.930
    ##  [881] 3957.250 3991.730 3958.790 3946.560 3965.340 3949.940 4003.580 4027.260
    ##  [889] 4026.120 3963.940 3957.630 4080.110 4076.570 4071.700 3998.840 3941.260
    ##  [897] 3933.920 3963.510 3934.380 3990.560 4019.650 3995.320 3895.750 3852.360
    ##  [905] 3817.660 3821.620 3878.440 3822.390 3844.820 3829.250 3783.220 3849.280
    ##  [913] 3839.500 3824.140 3852.970 3808.100 3895.080 3892.090 3919.250 3969.610
    ##  [921] 3983.170 3999.090 3990.970 3928.860 3898.850 3972.610 4019.810 4016.950
    ##  [929] 4016.220 4060.430 4070.560 4017.770 4076.600 4119.210 4179.760 4136.480
    ##  [937] 4111.080 4164.000 4117.860 4081.500 4090.460 4137.290 4136.130 4147.600
    ##  [945] 4090.410 4079.090 3997.340 3991.050 4012.320 3970.040 3982.240 3970.150
    ##  [953] 3951.390 3981.350 4045.640 4048.420 3986.370 3992.010 3918.320 3861.590
    ##  [961] 3855.760 3919.290 3891.930 3960.280 3916.640 3951.570 4002.870 3936.970
    ##  [969] 3948.720 3970.990 3977.530 3971.270 4027.810 4050.830 4109.310 4124.510
    ##  [977] 4100.600 4090.380 4105.020 4109.110 4108.940 4091.950 4146.220 4137.640
    ##  [985] 4151.320 4154.870 4154.520 4129.790 4133.520 4137.040 4071.630 4055.990
    ##  [993] 4135.350 4169.480 4167.870 4119.580 4090.750 4061.220 4136.250 4138.120
    ## [1001] 4119.170 4137.640 4130.620 4124.080 4136.280 4109.900 4158.770 4198.050
    ## [1009] 4191.980 4192.630 4145.580 4115.240 4151.280 4205.450 4205.520 4179.830
    ## [1017] 4221.020 4282.370 4273.790 4283.850 4267.520 4293.930 4298.860 4338.930
    ## [1025] 4369.010 4372.590 4425.840 4409.590 4388.710 4365.690 4381.890 4348.330
    ## [1033] 4328.820 4378.410 4376.860 4396.440 4450.380 4455.590 4446.820 4411.590
    ## [1041] 4398.950 4409.530 4439.260 4472.160 4510.040 4505.420 4522.790 4554.980
    ## [1049] 4565.720 4534.870 4536.340 4554.640 4567.460 4566.750 4537.410 4582.230
    ## [1057] 4588.960 4576.730 4513.390 4501.890 4478.030 4518.440 4499.380 4467.710
    ## [1065] 4468.830 4464.050 4489.720 4437.860 4404.330 4370.360 4369.710 4399.770
    ## [1073] 4387.550 4436.010 4376.310 4405.710 4433.310 4497.630 4514.870 4507.660
    ## [1081] 4515.770 4496.830 4465.480 4451.140 4457.490 4487.460 4461.900 4467.440
    ## [1089] 4505.100 4450.320 4453.530 4443.950 4402.200 4330.000 4320.060 4337.440
    ## [1097] 4273.530 4274.510 4299.700 4288.050 4288.390 4229.450 4263.750 4258.190
    ## [1105] 4308.500 4335.660 4358.240 4376.950 4349.610 4327.780 4373.630 4373.200
    ## [1113] 4314.600 4278.000 4224.160 4217.040 4247.680 4186.770 4137.230 4117.370
    ## [1121] 4166.820 4193.800 4237.860 4317.780 4358.340 4365.980 4378.380 4382.780
    ## [1129] 4347.350 4415.240 4411.550 4495.700 4502.880 4508.240 4514.020 4547.380
    ## [1137] 4538.190 4556.620 4559.340 4550.430 4554.890 4550.580 4567.800 4594.630
    ## [1145] 4569.780 4567.180 4549.340 4585.590 4604.370 4622.440 4643.700 4707.090
    ## [1153] 4719.550 4719.190 4740.560 4768.370 4698.350 4746.750 4754.630 4774.750
    ## [1161] 4781.580 4783.350 4769.830 4742.830 4704.810 4688.680 4697.240 4763.540
    ## [1169] 4756.500 4783.450 4780.240 4783.830 4765.980 4739.210 4780.940 4839.810
    ## [1177] 4850.430 4864.600 4868.550 4894.160 4890.970 4927.930 4924.970 4845.650
    ## [1185] 4906.190 4958.610 4942.810 4954.230 4995.060 4997.910 5026.610 5021.840
    ## [1193] 4953.170 5000.620 5029.730 5005.570 4975.510 4981.800 5087.030 5088.800
    ## [1201] 5069.530 5078.180 5069.760 5096.270 5137.080 5130.950 5078.650 5104.760
    ## [1209] 5157.360 5123.690 5117.940 5175.270 5165.310 5150.480 5117.090 5149.420
    ## [1217] 5178.510 5224.620 5241.530 5234.180 5218.190 5203.580 5248.490 5254.350
    ## [1225] 5243.770 5205.810 5211.490 5147.210 5204.340 5202.390 5209.910 5160.640
    ## [1233] 5199.060 5123.410 5061.820 5051.410 5022.210 5011.120 4967.230 5010.600
    ## [1241] 5070.550 5071.630 5048.420 5099.960 5116.170 5035.690 5018.390 5064.200
    ## [1249] 5127.790 5180.740 5187.700 5187.670 5214.080 5222.680 5221.420 5246.680
    ## [1257] 5308.150 5297.100 5303.270

``` r
plot(tsclean(prices), type="l")
```

![](Forecast_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

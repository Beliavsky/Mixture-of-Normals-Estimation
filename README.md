# Mixture-of-Normals-Estimation
Estimate a mixture of normals distribution using moments and the EM algorithm

Output:

```
Number of samples simulated: 100000

True parameters of the normal mixture:
Component 1: Weight = 0.3000, Mean = -2.0000, SD = 0.5000
Component 2: Weight = 0.4000, Mean = 0.0000, SD = 1.0000
Component 3: Weight = 0.3000, Mean = 3.0000, SD = 1.5000

Empirical moments from simulated data:
Mean: 0.3027, Std: 2.2234, Skew: 0.6176, Excess Kurtosis: -0.4706

Estimated parameters from moment matching:
Component 1: Weight = 0.3000, Mean = -1.7539, SD = 0.5532
Component 2: Weight = 0.4000, Mean = 0.6047, SD = 1.9148
Component 3: Weight = 0.3000, Mean = 1.9597, SD = 2.0249

Theoretical moments of the moment-estimated mixture:
Mean: 0.3036, Std: 2.2175, Skew: 0.5163, Excess Kurtosis: -0.4214

EM algorithm estimated parameters:
Component 1: Weight = 0.2794, Mean = -1.9960, SD = 0.4881
Component 2: Weight = 0.3945, Mean = -0.0704, SD = 1.1231
Component 3: Weight = 0.3261, Mean = 2.7232, SD = 1.6956

Theoretical moments of the EM-fitted mixture:
Mean: 0.3027, Std: 2.2234, Skew: 0.6372, Excess Kurtosis: -0.3606

Timing Information:
Data simulation time: 0.0051 seconds
Moment-based estimation time: 0.0556 seconds
EM algorithm estimation time: 0.4135 seconds

Theoretical moments of the true mixture:
Mean: 0.3000, Std: 2.2271, Skew: 0.6187, Excess Kurtosis: -0.4687
```

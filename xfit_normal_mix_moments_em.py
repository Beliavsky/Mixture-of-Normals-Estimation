"""
Mixture of Normals Parameter Estimation

This program simulates data from a mixture of normal distributions, computes empirical moments,
and estimates the mixture parameters using a moment-matching approach. Optionally, it can further
refine the estimates using the EM algorithm via scikit-learn. The program supports skipping 
moment fitting or EM estimation, and prints the true and estimated parameters, the corresponding 
theoretical moments, and timing information for each step.
"""

import time
import numpy as np
from scipy import stats, optimize
from sklearn.mixture import GaussianMixture

def compute_mixture_moments(means, sds, weights):
    """
    Compute overall moments of a mixture of normals.
    
    Parameters:
      means: array of component means
      sds:   array of component standard deviations
      weights: array of mixture weights (should sum to 1)
      
    Returns:
      overall_mean, overall_std, skew, excess_kurtosis
    """
    weights = np.asarray(weights)
    means = np.asarray(means)
    sds = np.asarray(sds)
    
    # Overall mean
    mu = np.sum(weights * means)
    
    # Overall second moment: E[X^2] = sum_i w_i*(s_i^2 + m_i^2)
    second_moment = np.sum(weights * (sds**2 + means**2))
    var = second_moment - mu**2
    std = np.sqrt(var)
    
    # Third central moment: For a normal component,
    # E[(X-mu)^3] = (m_i - mu)**3 + 3 s_i**2 (m_i - mu)
    third_central = np.sum(weights * ((means - mu)**3 + 3 * sds**2 * (means - mu)))
    skew = third_central / (std**3)
    
    # Fourth central moment: E[(X-mu)^4] = (m_i - mu)**4 + 6 (m_i - mu)**2 s_i**2 + 3 s_i**4.
    fourth_central = np.sum(weights * ((means - mu)**4 + 6 * (means - mu)**2 * sds**2 + 3 * sds**4))
    kurtosis = fourth_central / (var**2)
    excess_kurtosis = kurtosis - 3
    return mu, std, skew, excess_kurtosis

def fit_mixture_from_moments(target_mean, target_std, target_skew, target_exkurt, weights, reg_lambda=1e-2):
    """
    Fit the component means and standard deviations (via log-sds) for a mixture of normals
    with given weights so that the overall mixture moments match the target moments.
    
    Parameters:
      target_mean, target_std, target_skew, target_exkurt: target overall moments.
      weights: array-like, mixture weights (should sum to 1)
      reg_lambda: regularization strength
      
    Returns:
      (est_means, est_sds): arrays of estimated component means and standard deviations.
    """
    weights = np.asarray(weights)
    ncomp = len(weights)
    
    def residuals(p, p0):
        # p[0:ncomp] are means; p[ncomp:] are log(sds)
        means = p[:ncomp]
        sds = np.exp(p[ncomp:])
        
        mu, std, skew, exkurt = compute_mixture_moments(means, sds, weights)
        res = np.empty(4)
        res[0] = mu - target_mean
        res[1] = std - target_std
        res[2] = skew - target_skew
        res[3] = exkurt - target_exkurt
        
        # Regularization: penalize deviation from initial guess p0.
        reg = np.sqrt(reg_lambda) * (p - p0)
        return np.concatenate([res, reg])
    
    # Initial guess: spread means linearly around the target mean; set log-sds = log(target_std)
    spread = np.linspace(-(ncomp-1)/2, (ncomp-1)/2, ncomp)
    init_means = target_mean + spread * target_std * 0.5  # adjust scale as needed
    init_log_sds = np.log(np.full(ncomp, target_std))
    p0 = np.concatenate([init_means, init_log_sds])
    
    result = optimize.least_squares(residuals, p0, args=(p0,))
    est_means = result.x[:ncomp]
    est_sds = np.exp(result.x[ncomp:])
    return est_means, est_sds

def simulate_mixture(n_samples, weights, true_means, true_sds, random_state=None):
    """
    Simulate samples from a mixture of normals.
    
    Parameters:
      n_samples: number of samples to generate
      weights: mixture weights (summing to 1)
      true_means: component means
      true_sds: component standard deviations
      random_state: optional random state for reproducibility
      
    Returns:
      samples: 1D numpy array of simulated data.
    """
    rng = np.random.default_rng(random_state)
    weights = np.asarray(weights)
    true_means = np.asarray(true_means)
    true_sds = np.asarray(true_sds)
    ncomp = len(weights)
    
    # Randomly choose a component for each sample
    components = rng.choice(ncomp, size=n_samples, p=weights)
    samples = rng.normal(loc=true_means[components], scale=true_sds[components])
    return samples

if __name__ == '__main__':
    # =======================
    # ----- OPTIONS -------
    # =======================
    # Set these flags to choose which estimation method(s) to run.
    use_moments = True  # If True, use moment-based fitting. If False, skip moment fitting.
    use_em = True       # If True, run the EM algorithm. If False, use moment estimates only.
    
    # -----------------------
    # Mixture true parameters
    true_weights = np.array([0.3, 0.4, 0.3])
    true_means = np.array([-2.0, 0.0, 3.0])
    true_sds   = np.array([0.5, 1.0, 1.5])
    n_samples = 100000
    ncomp = len(true_weights)
    
    print("Number of samples simulated:", n_samples)
    
    # Print the true parameters of the normal mixture.
    print("\nTrue parameters of the normal mixture:")
    for i, (w, m, s) in enumerate(zip(true_weights, true_means, true_sds)):
        print(f"Component {i+1}: Weight = {w:.4f}, Mean = {m:.4f}, SD = {s:.4f}")
    
    # =======================
    # --- Data Simulation ---
    t0 = time.time()
    data = simulate_mixture(n_samples, true_weights, true_means, true_sds, random_state=42)
    sim_time = time.time() - t0
    
    # Compute empirical moments from the simulated data.
    emp_mean = np.mean(data)
    emp_std = np.std(data, ddof=1)
    emp_skew = stats.skew(data)
    emp_exkurt = stats.kurtosis(data)  # excess kurtosis (Fisher's definition)
    
    print("\nEmpirical moments from simulated data:")
    print(f"Mean: {emp_mean:.4f}, Std: {emp_std:.4f}, Skew: {emp_skew:.4f}, Excess Kurtosis: {emp_exkurt:.4f}")
    
    # =======================
    # Moment-based Estimation
    if use_moments:
        t1 = time.time()
        est_means, est_sds = fit_mixture_from_moments(emp_mean, emp_std, emp_skew, emp_exkurt, true_weights)
        moment_fit_time = time.time() - t1
        
        print("\nEstimated parameters from moment matching:")
        for i, (m, s) in enumerate(zip(est_means, est_sds)):
            print(f"Component {i+1}: Weight = {true_weights[i]:.4f}, Mean = {m:.4f}, SD = {s:.4f}")
        
        est_moments = compute_mixture_moments(est_means, est_sds, true_weights)
        print("\nTheoretical moments of the moment-estimated mixture:")
        print(f"Mean: {est_moments[0]:.4f}, Std: {est_moments[1]:.4f}, Skew: {est_moments[2]:.4f}, Excess Kurtosis: {est_moments[3]:.4f}")
    else:
        moment_fit_time = 0.0
        print("\nSkipping moment-based fitting; initial parameters will be set randomly for EM.")
    
    # =======================
    # EM Algorithm Estimation (using scikit-learn)
    if use_em:
        t2 = time.time()
        # Determine initial parameters for EM.
        if use_moments:
            # Use the moment-based estimates as initialization.
            init_means = est_means.reshape(-1, 1)
            init_weights = true_weights  # known weights
            init_precisions = (1.0 / (est_sds**2)).reshape(-1, 1)
        else:
            # Set random initialization parameters.
            rng = np.random.default_rng(123)
            # Random means uniformly in the range of the data.
            init_means = rng.uniform(np.min(data), np.max(data), size=ncomp).reshape(-1, 1)
            # Use uniform weights.
            init_weights = np.full(ncomp, 1.0/ncomp)
            # Set initial sds to sample std.
            sample_std = np.std(data, ddof=1)
            init_sds = np.full(ncomp, sample_std)
            init_precisions = (1.0 / (init_sds**2)).reshape(-1, 1)
        
        gmm = GaussianMixture(n_components=ncomp, covariance_type='diag', 
                              weights_init=init_weights, means_init=init_means,
                              precisions_init=init_precisions, max_iter=100, random_state=42)
        data_reshaped = data.reshape(-1, 1)
        gmm.fit(data_reshaped)
        em_time = time.time() - t2
        
        em_weights = gmm.weights_
        em_means = gmm.means_.flatten()
        em_variances = gmm.covariances_.flatten()  # for 'diag' covariance, each is a variance
        em_sds = np.sqrt(em_variances)
        
        print("\nEM algorithm estimated parameters:")
        for i, (w, m, s) in enumerate(zip(em_weights, em_means, em_sds)):
            print(f"Component {i+1}: Weight = {w:.4f}, Mean = {m:.4f}, SD = {s:.4f}")
        
        em_moments = compute_mixture_moments(em_means, em_sds, em_weights)
        print("\nTheoretical moments of the EM-fitted mixture:")
        print(f"Mean: {em_moments[0]:.4f}, Std: {em_moments[1]:.4f}, Skew: {em_moments[2]:.4f}, Excess Kurtosis: {em_moments[3]:.4f}")
    else:
        em_time = 0.0
        print("\nSkipping EM algorithm; using moment-based estimates only.")
    
    # =======================
    # Print Timing Information
    print("\nTiming Information:")
    print(f"Data simulation time: {sim_time:.4f} seconds")
    print(f"Moment-based estimation time: {moment_fit_time:.4f} seconds")
    print(f"EM algorithm estimation time: {em_time:.4f} seconds")
    
    # -----------------------
    # Also print theoretical moments of the true mixture for reference.
    true_moments = compute_mixture_moments(true_means, true_sds, true_weights)
    print("\nTheoretical moments of the true mixture:")
    print(f"Mean: {true_moments[0]:.4f}, Std: {true_moments[1]:.4f}, Skew: {true_moments[2]:.4f}, Excess Kurtosis: {true_moments[3]:.4f}")

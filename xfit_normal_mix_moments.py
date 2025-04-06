import numpy as np
from scipy import stats, optimize

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
    
    # Third central moment: For a normal component, E[(X-mu)^3] = (m_i - mu)^3 + 3 s_i^2 (m_i - mu)
    third_central = np.sum(weights * ((means - mu)**3 + 3 * sds**2 * (means - mu)))
    skew = third_central / (std**3)
    
    # Fourth central moment: E[(X-mu)^4] = (m_i - mu)**4 + 6 (m_i - mu)**2 s_i**2 + 3 s_i**4.
    fourth_central = np.sum(weights * ((means - mu)**4 + 6 * (means - mu)**2 * sds**2 + 3 * sds**4))
    kurtosis = fourth_central / (var**2)
    excess_kurtosis = kurtosis - 3
    return mu, std, skew, excess_kurtosis

def fit_mixture_from_moments(target_mean, target_std, target_skew, target_excess_kurtosis, weights, reg_lambda=1e-2):
    """
    Fit the component means and standard deviations (via log-sds) for a mixture of normals
    with given weights so that the overall mixture moments match the target moments.
    
    Parameters:
      target_mean, target_std, target_skew, target_excess_kurtosis: target overall moments.
      weights: array-like, mixture weights (should sum to 1)
      reg_lambda: regularization strength (penalizes deviation from initial guess)
      
    Returns:
      (est_means, est_sds): arrays of estimated component means and standard deviations.
    """
    weights = np.asarray(weights)
    ncomp = len(weights)
    
    # Define the objective: residuals on moment matching plus regularization
    def residuals(p, p0):
        # p[0:ncomp] are means; p[ncomp:] are log(sds)
        means = p[:ncomp]
        sds = np.exp(p[ncomp:])
        
        # Compute moments from the candidate parameters
        mu, std, skew, exkurt = compute_mixture_moments(means, sds, weights)
        
        # Residuals for the 4 moments
        res = np.empty(4)
        res[0] = mu - target_mean
        res[1] = std - target_std
        res[2] = skew - target_skew
        res[3] = exkurt - target_excess_kurtosis
        
        # Regularization: penalize deviation from the initial guess p0.
        # (This helps to pick one solution among many for under-determined cases.)
        reg = np.sqrt(reg_lambda) * (p - p0)
        return np.concatenate([res, reg])
    
    # Build an initial guess p0:
    # For the means, we start with a linear spread around the target mean.
    # For log(sds), we start at log(target_std).
    spread = np.linspace(-(ncomp-1)/2, (ncomp-1)/2, ncomp)
    init_means = target_mean + spread * target_std * 0.5  # scale factor can be adjusted
    init_log_sds = np.log(np.full(ncomp, target_std))
    p0 = np.concatenate([init_means, init_log_sds])
    
    # Use least squares optimization:
    result = optimize.least_squares(residuals, p0, args=(p0,))
    
    # Extract estimated parameters
    est_means = result.x[:ncomp]
    est_sds = np.exp(result.x[ncomp:])
    return est_means, est_sds

# ----- Testing the function with simulated data -----

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
    
    # Choose component for each sample
    components = rng.choice(ncomp, size=n_samples, p=weights)
    samples = rng.normal(loc=true_means[components], scale=true_sds[components])
    return samples

if __name__ == '__main__':
    # Define a mixture with an arbitrary number of components.
    # For example, here we use 3 components.
    true_weights = np.array([0.3, 0.4, 0.3])
    true_means = np.array([-2.0, 0.0, 3.0])
    true_sds   = np.array([0.5, 1.0, 1.5])
    
    # Simulate data
    n_samples = 100000
    print("#obs:", n_samples, end="\n\n")
    data = simulate_mixture(n_samples, true_weights, true_means, true_sds, random_state=42)
    
    # Compute empirical moments
    emp_mean = np.mean(data)
    emp_std = np.std(data, ddof=1)
    emp_skew = stats.skew(data)
    emp_exkurt = stats.kurtosis(data)  # by default, fisher=True, so this is excess kurtosis
    
    print("Empirical moments from simulated data:")
    print(f"Mean: {emp_mean:.4f}, Std: {emp_std:.4f}, Skew: {emp_skew:.4f}, Excess Kurtosis: {emp_exkurt:.4f}")
    
    # Fit the mixture parameters from the moments
    est_means, est_sds = fit_mixture_from_moments(emp_mean, emp_std, emp_skew, emp_exkurt, true_weights)
    
    print("\nEstimated component parameters:")
    for i, (m, s) in enumerate(zip(est_means, est_sds)):
        print(f"Component {i+1}: Mean = {m:.4f}, SD = {s:.4f}")
    
    print("\nTrue component parameters:")
    for i, (m, s, w) in enumerate(zip(true_means, true_sds, true_weights)):
        print(f"Component {i+1}: Weight = {w:.4f}, Mean = {m:.4f}, SD = {s:.4f}")
    
    # Compute theoretical moments for the true and estimated mixtures
    true_moments = compute_mixture_moments(true_means, true_sds, true_weights)
    est_moments = compute_mixture_moments(est_means, est_sds, true_weights)
    
    print("\nTheoretical moments of the true mixture:")
    print(f"Mean: {true_moments[0]:.4f}, Std: {true_moments[1]:.4f}, Skew: {true_moments[2]:.4f}, Excess Kurtosis: {true_moments[3]:.4f}")
    
    print("\nTheoretical moments of the estimated mixture:")
    print(f"Mean: {est_moments[0]:.4f}, Std: {est_moments[1]:.4f}, Skew: {est_moments[2]:.4f}, Excess Kurtosis: {est_moments[3]:.4f}")

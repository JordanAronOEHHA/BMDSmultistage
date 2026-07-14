import nlopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import brentq
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from tqdm import tqdm
import numdifftools as nd

class MultistageModel:
    def __init__(self, data, degree: int = 1,
                 bmr: float = 0.05, confidence_level: float = 0.95,
                 bmd_grad: int = 1, bmdl_grad: int = 1,
                 lambda_reg: float = 0.0, ms1_b1: float = 0.0):
        self.data = data
        self.doses = self.data[:, 0]
        self.Ns = self.data[:, 1]
        self.n = self.data[:, 2]
        self.degree = degree
        self.num_params = 1 + degree

        self.bmr = bmr
        self.confidence_level = confidence_level
        self.chi2_val = chi2.ppf(2 * self.confidence_level - 1, 1)/2
        self.bmd_grad = bmd_grad
        self.bmdl_grad = bmdl_grad

        self.parameters = None
        self.lambda_reg = lambda_reg
        self.ms1_b1 = ms1_b1

        self.bmd = None
        self.bmdl = None
        self.nll = None 
        self.unreg_nll = None

        self.standard_errors = None
        
        # Attribute to store CV results for plotting
        self.cv_results_ = None

    def UpdateLambda(self,lambda_reg: float):
        self.lambda_reg = lambda_reg

    @staticmethod
    def unsummarize_data(arr):
        """Transforms summarized data into long format (binary outcomes)."""
        rows = []
        for dose, n, y in arr:
            ones = int(y)
            zeros = int(n - y)
            for _ in range(ones): rows.append([dose, 1])
            for _ in range(zeros): rows.append([dose, 0])
        return pd.DataFrame(rows, columns=['dose', 'outcome'])

    @staticmethod
    def summarize_data(df):
        """Transforms unsummarized (long) format back to summarized format."""
        summary = df.groupby('dose')['outcome'].agg(['count', 'sum']).reset_index()
        summary.columns = ['dose', 'sample_size', 'observed']
        return summary.values

    def CalcMLE(self):
        """Performs Maximum Likelihood Estimation."""
        if self.bmd_grad == 1:
            opt = nlopt.opt(nlopt.LD_SLSQP, self.num_params)
        else:
            opt = nlopt.opt(nlopt.LN_COBYLA, self.num_params)

        lb = [0.0] * self.num_params
        ub = [0.999] + [10000] * self.degree
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        opt.set_xtol_rel(1e-8)
        opt.set_ftol_rel(1e-8)
        opt.set_maxeval(20000)

        opt.set_min_objective(lambda x, g: neg_log_likelihood(x, g, self.doses, self.Ns, self.n, self.degree, self.ms1_b1, self.lambda_reg))

        initial_guess = [0.01] * self.num_params
        try:
            self.parameters = opt.optimize(initial_guess)
            self.nll = -opt.last_optimum_value()
            self.unreg_nll = -opt.last_optimum_value() - self.lambda_reg * (self.ms1_b1 - self.parameters[1])**2
        except nlopt.RoundoffLimited:
            pass

    def CalculateBMD(self):
        target_val = np.log(1 - self.bmr)
        f = lambda d: sum(b * (d ** (i + 1)) for i, b in enumerate(self.parameters[1:])) + target_val
        self.bmd = brentq(f, 0, max(self.doses))


    # Profile Likelihood with Constraint + maybe gradient
    def _profile_likelihood_at_dose(self, test_dose:float) -> float:
        if self.bmdl_grad:
            prof_opt = nlopt.opt(nlopt.LD_SLSQP, self.num_params)
        else:
            prof_opt = nlopt.opt(nlopt.LN_COBYLA, self.num_params)

        lb = [0.0] * self.num_params
        ub = [0.999] + [10000] * self.degree # g_br < 1.0 strict for log stability
        
        prof_opt.set_lower_bounds(lb)
        prof_opt.set_upper_bounds(ub)
        #TODO pass through MS1 B1
        prof_opt.set_min_objective(lambda x, g: neg_log_likelihood(x, g, self.doses, self.Ns, self.n, self.degree, self.ms1_b1, self.lambda_reg))
        prof_opt.set_xtol_rel(1e-6)
        
        # CONSTRAINT DEFINITION WITH GRADIENT
        # Constraint: sum(beta * dose^i) - log(1-BMR) = 0
        def bmr_constraint(x, grad):
            betas = x[1:]
            
            # Gradient of constraint
            if grad.size > 0:
                grad[0] = 0.0 # d/dg_br is 0 
                for i in range(self.degree):
                    # d/dBeta_i is dose^(i+1)
                    grad[i+1] = test_dose ** (i + 1)
                    
            # val of constraint
            poly_sum = sum(b * (test_dose ** (i + 1)) for i, b in enumerate(betas))
            return poly_sum + np.log(1 - self.bmr)

        # Equality constraint
        prof_opt.add_equality_constraint(bmr_constraint, 1e-4)
        
        try:
            # Start search from MLE params for speed
            prof_opt.optimize(self.parameters) 
            return -prof_opt.last_optimum_value()
        except Exception:
            return -np.inf

    def bootstrap_bmd(self, n_bootstrap: int = 1000, confidence_level: float = 0.95, method: str = 'p'):
        """
        Calculates the BMDL using bootstrapping with a visual progress bar.
        
        Args:
            n_bootstrap (int): Number of bootstrap iterations.
            confidence_level (float): Confidence level for the lower bound.
            method (str): 'parametric' (model-based) or 'non-parametric' (data-resampling).
        """
        if self.parameters is None:
            raise ValueError("Model must be fitted via CalcMLE() before bootstrapping.")

        boot_bmds = []
        
        # Setup for the selected method
        if method == 'p':
            # Use multistage_dr to get probabilities from the fitted parameters.
            predicted_probs = multistage_dr(self.doses, self.parameters[0], self.parameters[1:])
        else:
            # Prepare the long-form version of the data for resampling.
            df_long = self.unsummarize_data(self.data)
        
        # The tqdm wrapper adds a progress bar to the console output
        # for i in tqdm(range(n_bootstrap), desc=f"Bootstrap ({method})"):
        for i in range(n_bootstrap):
            if method == 'p':
                # Generate new synthetic incidence using the Binomial distribution.
                boot_n = np.random.binomial(self.Ns.astype(int), predicted_probs)
                boot_data = np.column_stack((self.doses, self.Ns, boot_n))
            else:
                # Resample observations with replacement within each dose group.
                boot_sample = df_long.groupby('dose').sample(frac=1, replace=True)
                boot_data = self.summarize_data(boot_sample)
            
            # Re-fit the model on the bootstrapped data.
            tmp_model = MultistageModel(
                data=boot_data,
                degree=self.degree,
                bmr=self.bmr,
                lambda_reg=self.lambda_reg,
                ms1_b1=self.ms1_b1,
                bmd_grad=self.bmd_grad
            )
            
            try:
                tmp_model.CalcMLE()
                tmp_model.CalculateBMD()
                if tmp_model.bmd and not np.isnan(tmp_model.bmd):
                    boot_bmds.append(tmp_model.bmd)
            except Exception:
                continue

        # Calculate the lower quantile (e.g., 5th percentile for 95% confidence).
        lower_quantile = (1.0 - confidence_level) * 100
        self.bmdl_boot = np.percentile(boot_bmds, lower_quantile)
        self.all_boot_bmds = np.array(boot_bmds)
        self.boot_method_ = method
        
        return self.bmdl_boot

    def plot_bootstrap_distribution(self):
        """
        Visualizes the distribution of bootstrapped BMD values and the resulting BMDL.
        """
        if not hasattr(self, 'all_boot_bmds'):
            raise ValueError("You must run bootstrap_bmd() before plotting.")

        plt.figure(figsize=(10, 6))
        plt.hist(self.all_boot_bmds, bins=30, color='#a8dadc', edgecolor='#457b9d', alpha=0.8)
        
        # Original BMD estimate and the new Bootstrap BMDL
        plt.axvline(self.bmd, color='#1d3557', linestyle='--', linewidth=2, label=f'Original BMD: {self.bmd:.4f}')
        plt.axvline(self.bmdl_boot, color='#e63946', linestyle='-', linewidth=2, label=f'Bootstrap BMDL: {self.bmdl_boot:.4f}')
        
        plt.title(f'Parametric Bootstrap BMD Distribution (n={len(self.all_boot_bmds)})', fontsize=14)
        plt.xlabel('BMD Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(axis='y', alpha=0.2)
        plt.show()
    # Objective function in root finding function
    def _find_bmdl_root(self, d:float) -> float:
        self.ll_at_d = self._profile_likelihood_at_dose(d)
        return self.ll_at_d - self.target_ll

    def CalculateBMDL(self):
        self.target_ll = self.nll - self.chi2_val
        try:
            self.bmdl = brentq(self._find_bmdl_root, 1e-4, self.bmd)
            print("-" * 30)
            print(f"BMDL (95% Lower Bound): {self.bmdl:.4f}")
            print("-" * 30)
        except Exception as e:
            print(f"Could not converge: {e}")

    def select_lambda_cv(self, lambda_values=range(0, 200, 2), n_splits=None, random_state=1):
        """
        Calculates the optimal lambda using the One Standard Error Rule via 
        Stratified K-Fold Cross-Validation.
        """
        df_long = self.unsummarize_data(self.data)
        if n_splits is None:
            n_splits = len(self.data)
            
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        rmse_results = defaultdict(list)

        for train_idx, test_idx in skf.split(df_long, df_long['outcome']):
            train_fold = self.summarize_data(df_long.iloc[train_idx])
            test_fold = self.summarize_data(df_long.iloc[test_idx])

            # Baseline fit (Degree 1) to get ms1_b1
            ms1_baseline = MultistageModel(train_fold, degree=1, lambda_reg=0, ms1_b1=0, bmd_grad = self.bmd_grad)
            ms1_baseline.CalcMLE()
            b1_prior = ms1_baseline.parameters[1]

            for lam in lambda_values:
                ms2_cv = MultistageModel(train_fold, degree=2, lambda_reg=lam, ms1_b1=b1_prior, bmd_grad = self.bmd_grad)
                ms2_cv.CalcMLE()
                
                # RMSE Calculation
                preds = multistage_dr(test_fold[:, 0], ms2_cv.parameters[0], ms2_cv.parameters[1:])
                actuals = test_fold[:, 2] / test_fold[:, 1]
                rmse = np.sqrt(np.mean((preds - actuals)**2))
                rmse_results[lam].append(rmse)

        # Process Results
        lams = sorted(rmse_results.keys())
        avg_rmses = np.array([np.mean(rmse_results[l]) for l in lams])
        se_rmses = np.array([np.std(rmse_results[l], ddof=1) / np.sqrt(n_splits) for l in lams])

        min_idx = np.argmin(avg_rmses)
        threshold = avg_rmses[min_idx] + se_rmses[min_idx]
        
        # Apply 1-SE Rule: Largest lambda within 1 SE of the minimum
        best_1se_lam = max([lams[i] for i, val in enumerate(avg_rmses) if val <= threshold])
        
        # Store for plotting
        self.cv_results_ = {
            'lams': lams, 'avg_rmses': avg_rmses, 'se_rmses': se_rmses,
            'min_lam': lams[min_idx], 'best_1se_lam': best_1se_lam, 'threshold': threshold
        }
        
        self.lambda_reg = best_1se_lam
        return best_1se_lam

    def plot_cv_results(self):
        """Creates the Lambda vs RMSE plot with the 1-SE threshold."""
        if self.cv_results_ is None:
            raise ValueError("CV must be run via select_lambda_cv() before plotting.")
            
        res = self.cv_results_
        plt.figure(figsize=(10, 6))
        plt.errorbar(res['lams'], res['avg_rmses'], yerr=res['se_rmses'], 
                     fmt='o-', color='black', ecolor='lightgray', label='Avg RMSE $\\pm$ 1 SE')
        
        plt.axhline(y=res['threshold'], color='red', linestyle='--', label='1-SE Threshold')
        plt.plot(res['best_1se_lam'], res['avg_rmses'][res['lams'].index(res['best_1se_lam'])], 
                 'ro', markersize=10, label=f'1-SE Selection ($\\lambda$={res["best_1se_lam"]})')
        
        plt.title('$\\lambda$ Selection: One Standard Error Rule')
        plt.xlabel('$\\lambda$')
        plt.ylabel('Average RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_dose_response(self, title: str = "Multistage Dose-Response Fit"):
        """
        Plots the observed data with binomial error bars and the fitted model curve.
        """
        if self.parameters is None:
            raise ValueError("Model must be fitted via CalcMLE() before plotting.")

        # 1. Prepare Data Points
        # Observed probability = incidence / sample_size
        obs_probs = self.n / self.Ns
        
        # Standard Error for Binomial Distribution: sqrt(p(1-p)/n)
        # We handle p=0 or p=1 carefully, though mathematically SE is 0 there.
        se = np.sqrt(obs_probs * (1 - obs_probs) / self.Ns)

        # 2. Generate Smooth Curve
        # Create a dense range of doses from 0 to max dose for a smooth line
        x_smooth = np.linspace(0, np.max(self.doses), 200)
        
        # Calculate predicted probabilities using the fitted parameters
        # self.parameters[0] is background (g), self.parameters[1:] are betas
        y_smooth = multistage_dr(x_smooth, self.parameters[0], self.parameters[1:])

        # 3. Create Plot
        plt.figure(figsize=(10, 6))
        
        # Plot the smooth fitted curve
        plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label=f'Fitted Degree {self.degree} Model')
        
        z = 1.96
        probs = self.n / self.Ns

        lb = (2 * self.Ns * probs + z**2 - 1) - z * np.sqrt(z**2 - (2 + 1/self.Ns) + 4 * probs * (self.Ns * (1 - probs) + 1))
        lb = lb / (2 * (self.Ns + z**2))
        lb = probs - lb
        lb[lb<0] = 0

        ub = (2 * self.Ns * probs + z**2 + 1) + z * np.sqrt(z**2 + (2 - 1/self.Ns) + 4 * probs * (self.Ns * (1 - probs) - 1))
        ub = ub / (2 * (self.Ns + z**2))
        ub = ub - probs

        error_mat = np.array([lb, ub])

        # Plot the data points with error bars
        plt.errorbar(self.doses, obs_probs, yerr=error_mat, fmt='ko', capsize=5, 
                     elinewidth=1.5, markeredgewidth=1.5, label='Observed Data ± 1 SE')

        # 4. Add BMD and BMDL lines if they exist
        if self.bmd is not None:
            plt.axvline(self.bmd, color='green', linestyle='--', alpha=0.7, 
                        label=f'BMD: {self.bmd:.3f}')
            
            # Highlight the BMR response level at the BMD
            bmd_response = multistage_dr(np.array([self.bmd]), self.parameters[0], self.parameters[1:])
            plt.hlines(bmd_response, 0, self.bmd, colors='green', linestyles=':', alpha=0.5)

        if self.bmdl is not None:
            plt.axvline(self.bmdl, color='red', linestyle='--', alpha=0.7, 
                        label=f'BMDL: {self.bmdl:.3f}')

        # Formatting
        plt.title(title, fontsize=14)
        plt.xlabel('Dose', fontsize=12)
        plt.ylabel('Probability of Response', fontsize=12)
        plt.ylim(-0.05, 1.05)  # Keep y-axis neat
        plt.legend(loc='best')
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

    def calculate_goodness_of_fit(self):
        """
        Calculates Scaled Residuals and the Chi-squared GOF statistic.
        """
        P_hat = multistage_dr(self.doses,self.parameters[0],self.parameters[1:])
        E = self.Ns * P_hat  # Expected number of responses
        
        scaled_residuals = []
        chi2_gof = 0.0
        
        for i in range(len(self.doses)):
            # Observed response (n) - Expected response (E)
            O_minus_E = self.n[i] - E[i]
            
            # Denominator: Expected variance for binomial distribution
            V_hat = self.Ns[i] * P_hat[i] * (1 - P_hat[i])
            
            # Scaled Residual: (O - E) / sqrt(V_hat)
            if V_hat > 1e-12:
                residual = O_minus_E / np.sqrt(V_hat)
            else:
                residual = 0.0
            
            scaled_residuals.append(residual)
            
            # Contribution to Chi-squared: (O - E)^2 / V_hat
            if V_hat > 1e-12:
                chi2_gof += (O_minus_E ** 2) / V_hat
                
        # Degrees of Freedom (DF) = Number of Dose Groups - Number of Parameters
        df = len(self.doses) - sum(self.parameters > 1e-8)
        
        # P-value for the GOF test
        p_value = 1.0 - chi2.cdf(chi2_gof, df) if df > 0 else 1.0

        self.scaled_residuals = scaled_residuals
        self.chi2_gof = chi2_gof
        self.df = df
        self.p_value = p_value

    def CalculateStandardErrors(self):
        """
        Calculates the standard errors for the parameters using the 
        inverse of the observed Fisher Information Matrix (Negative Hessian).
        """
        if self.parameters is None:
            raise ValueError("Model must be fitted via CalcMLE() before calculating standard errors.")

        # 1. Define the objective function for the Hessian
        # We use the UNREGULATED negative log-likelihood for standard errors 
        # consistent with standard BMDS output.
        def objective(params):
            # We wrap your existing neg_log_likelihood
            # We set lambda_reg to 0 to get the Fisher Information of the model fit itself
            return neg_log_likelihood(
                params, np.array([]), self.doses, self.Ns, self.n, 
                self.degree, ms1_b1=0, lambda_reg=0
            )

        # 2. Calculate the Hessian matrix
        hessian_func = nd.Hessian(objective)
        hessian_mat = hessian_func(self.parameters)

        # 3. Calculate Variance-Covariance Matrix
        # Fisher Information (I) = -Hessian of Log-Likelihood 
        # Since our objective is NEGATIVE Log-Likelihood, I = Hessian
        try:
            # We use the pseudo-inverse in case the matrix is near-singular
            # (common when parameters are near the 0 boundary)
            var_covar = np.linalg.pinv(hessian_mat)
            diag = np.diag(var_covar)
            
            # 4. Extract SEs (Square root of the diagonal)
            # We mask parameters that are essentially 0 (at the boundary) 
            # as their SEs are not statistically meaningful in this framework.
            se = []
            for i, p in enumerate(self.parameters):
                if p < 1e-7: # Parameter is at the lower bound
                    se.append(np.nan)
                elif diag[i] < 0: # Numerical instability check
                    se.append(np.inf)
                else:
                    se.append(np.sqrt(diag[i]))
            
            self.standard_errors = np.array(se)

        except np.linalg.LinAlgError:
            print("Hessian matrix inversion failed. Cannot calculate Standard Errors.")

    def CalculateStandardErrorsExact(self):
        """
        Calculates the parameter standard errors by analytically computing 
        the Hessian, inverting it, and taking the square root of the diagonal.
        """
        if self.parameters is None:
            raise ValueError("Model must be fitted via CalcMLE() first.")

        # 1. Setup variables
        g = self.parameters[0]
        betas = self.parameters[1:]
        probs = multistage_dr(self.doses, g, betas)
        
        # Precompute the weight term: n * (1-P) / P^2
        # This reflects the binomial curvature at each dose point
        weight = (self.n * (1 - probs)) / (probs**2)
        
        # 2. Construct the Hessian (H)
        H = np.zeros((self.num_params, self.num_params))
        
        # Background-Background (g, g)
        denom_g = (1 - g) if (1 - g) > 1e-10 else 1e-10
        H[0, 0] = np.sum(weight / (denom_g**2))
        
        # Beta and Cross terms
        for j in range(1, self.num_params):
            # H[0, j] is the Background-Beta_j cross term
            H[0, j] = H[j, 0] = np.sum((weight * (self.doses**j)) / denom_g)
            
            for k in range(1, self.num_params):
                # H[j, k] is the Beta_j-Beta_k term
                H[j, k] = np.sum(weight * (self.doses**(j + k)))

        # Add regularization for Beta_1 (params[1])
        H[1, 1] += 2 * self.lambda_reg

        # 3. Invert the Hessian to get the Variance-Covariance Matrix
        try:
            # We use pinv (pseudo-inverse) to handle matrices that are 
            # singular due to parameters hitting the 0 boundary.
            var_covar_matrix = np.linalg.pinv(H)
            
            # 4. Extract the square root of the diagonal
            variances = np.diag(var_covar_matrix)
            
            # Use max(0, v) to prevent tiny negative numbers from precision issues
            # then identify boundary parameters where SE is technically undefined (NA)
            se = []
            for i, v in enumerate(variances):
                if self.parameters[i] < 1e-7: # Boundary check
                    se.append(np.nan)
                else:
                    se.append(np.sqrt(max(0, v)))
                    
            self.standard_errors = np.array(se)
            

        except np.linalg.LinAlgError:
            print("Matrix inversion failed: Singular Hessian.")

# Helper functions provided in the original file
def multistage_dr(dose, g_br, betas):
    poly_sum = np.clip(sum(b * (dose ** (i + 1)) for i, b in enumerate(betas)), 0, 700)
    probs = 1 - (1 - g_br) * np.exp(-poly_sum)
    return np.clip(probs, 1e-16, 1 - 1e-16)

def neg_log_likelihood(params:  np.ndarray, grad:  np.ndarray, data_doses:  np.ndarray, samplesize_vec:  np.ndarray, incidence_vec:  np.ndarray, degree: int, ms1_b1: float, lambda_reg: float) -> float:
    g_br = params[0]
    betas = params[1:]
    
    probs = multistage_dr(data_doses, g_br, betas)
    
    # Gradient Calculation
    # NLopt expects us to modify grad array in-place if it exists
    if grad.size > 0:
        # Common term: (NP - n) / P
        # Derived from d(NLL)/dP * dP/dTheta simplifications
        com_term = (samplesize_vec * probs - incidence_vec) / probs
        
        # Partial wrt g_br: sum( common term * 1/(1-g_br) )
        # if error (bc g = 1), may need to add small value to denominator for stability
        grad[0] = np.sum(com_term * (1.0 / (1.0 - g_br)))
        
        # Partial wrt Betas: sum( common term * d^i )
        # Because dP/dBeta = (1-P) * d^i
        for i in range(degree):
            grad[i+1] = np.sum(com_term * (data_doses ** (i + 1)))
        
        grad[1] += -2*lambda_reg*(ms1_b1-betas[0])  # regularization term gradient

    # Likelihood Calculation
    ll = np.sum(incidence_vec * np.log(probs) + (samplesize_vec - incidence_vec) * np.log(1 - probs))

    penalty = lambda_reg * (ms1_b1-betas[0])**2  # regularization term
    return -ll + penalty




############################################################

class MultiTumorModel:
    def __init__(self, models: list, bmr: float = 0.05, confidence_level: float = 0.95):
        """
        Initializes the Multi-Tumor analysis.
        
        Args:
            models (list): A list of fitted MultistageModel instances.
            bmr (float): The Benchmark Response for the combined risk (usually 0.1 for cancer).
            confidence_level (float): Confidence level for the lower bound.
        """
        self.models = models
        self.bmr = bmr
        self.confidence_level = confidence_level
        # BMDS uses Chi-square with 1 df for profile likelihood bounds
        self.chi2_val = chi2.ppf(2 * self.confidence_level - 1, 1) / 2
        
        # Extract combined bounds and capacities
        self.max_degree = max(m.degree for m in self.models)
        self.total_params = sum(m.num_params for m in self.models)
        
        self.bmd = None
        self.bmdl = None
        self.combined_nll = None
        self.target_nll = None
        self.combined_parameters = None

    def CalculateCombinedMLE(self):
        """
        Calculates combined MLE parameters and BMD based on individual fitted models.
        """
        combined_betas = np.zeros(self.max_degree)
        g_inv_product = 1.0
        
        # We rename this to combined_ll to be mathematically accurate
        # since m.nll in the base class actually stores the Log-Likelihood
        self.combined_ll = 0.0

        for m in self.models:
            if m.parameters is None:
                raise ValueError("All individual models must be fitted via CalcMLE() first.")
            
            g_inv_product *= (1 - m.parameters[0])
            for j in range(m.degree):
                combined_betas[j] += m.parameters[j + 1]
            
            self.combined_ll += m.nll

        g_comb = 1.0 - g_inv_product
        self.combined_parameters = np.concatenate(([g_comb], combined_betas))

    def CalculateBMD(self):
        """
        Calculates the MLE BMD for the combined multi-tumor response.
        """
        if self.combined_parameters is None:
            self.CalculateCombinedMLE()

        # Extra risk target: BMR = 1 - exp(-f(d)) => f(d) = -ln(1 - BMR)
        target_val = np.log(1 - self.bmr)
        betas = self.combined_parameters[1:]
        
        # f(d) represents the polynomial sum across combined betas
        f = lambda d: sum(b * (d ** (i + 1)) for i, b in enumerate(betas)) + target_val
        
        max_dose = max(max(m.doses) for m in self.models)
        
        try:
            # Brentq finds the root where f(d) == 0
            self.bmd = brentq(f, 0, max_dose * 100) 
        except ValueError:
            self.bmd = np.nan
            print("Could not calculate combined BMD. Curve may not cross BMR.")


    def _profile_likelihood_at_dose(self, test_dose: float) -> float:
        """
        Optimizes individual tumor parameters to maximize the Log-Likelihood
        subject to the combined extra risk equating to the BMR constraint.
        """
        opt = nlopt.opt(nlopt.LD_SLSQP, self.total_params)
        lb, ub, initial_guess = [], [], []
        
        for m in self.models:
            lb.extend([0.0] * m.num_params)
            ub.extend([0.999] + [100000] * m.degree) 
            initial_guess.extend(m.parameters)
            
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        opt.set_xtol_rel(1e-6)
        
        def combined_objective(x, grad):
            idx = 0
            total_nll = 0.0
            
            for m in self.models:
                num_p = m.num_params
                m_params = x[idx:idx + num_p]
                m_grad = np.zeros(num_p) if grad.size > 0 else np.array([])
                
                nll = neg_log_likelihood(
                    m_params, m_grad, m.doses, m.Ns, m.n, 
                    m.degree, m.ms1_b1, m.lambda_reg
                )
                total_nll += nll
                
                if grad.size > 0:
                    grad[idx:idx + num_p] = m_grad
                idx += num_p
                
            return total_nll
            
        opt.set_min_objective(combined_objective)
        
        def bmr_constraint(x, grad):
            poly_sum = 0.0
            idx = 0
            if grad.size > 0:
                grad.fill(0.0)
                
            for m in self.models:
                for i in range(m.degree):
                    param_index = idx + 1 + i 
                    beta_val = x[param_index]
                    poly_sum += beta_val * (test_dose ** (i + 1))
                    if grad.size > 0:
                        grad[param_index] = test_dose ** (i + 1)
                idx += m.num_params
                
            return poly_sum + np.log(1 - self.bmr)

        opt.add_equality_constraint(bmr_constraint, 1e-4)
        
        try:
            opt.optimize(initial_guess)
            # CRITICAL FIX: Return the negative to convert NLL back to Log-Likelihood (LL)
            return -opt.last_optimum_value()
        except nlopt.RoundoffLimited:
            return -opt.last_optimum_value()
        except Exception:
            try: return -opt.last_optimum_value()
            except: return -1e6 # Return a heavily penalized negative LL on total failure

    def _find_bmdl_root(self, d: float) -> float:
        """Root-finding objective for the BMDL."""
        ll_at_d = self._profile_likelihood_at_dose(d)
        return ll_at_d - self.target_ll

    def CalculateBMDL(self):
        if self.bmd is None:
            self.CalculateBMD()
            
        # Target LL is MLE Log-Likelihood MINUS chi2 (1 degree of freedom penalty)
        self.target_ll = self.combined_ll - self.chi2_val
        
        upper_bound_dose = self.bmd
        lower_bound_dose = self.bmd * 1e-4 
        
        f_b = self._find_bmdl_root(upper_bound_dose)
        f_a = self._find_bmdl_root(lower_bound_dose)
        
        # If the lower bound doesn't evaluate negative, walk it backward
        attempts = 0
        while f_a > 0 and attempts < 10:
            lower_bound_dose /= 10
            f_a = self._find_bmdl_root(lower_bound_dose)
            attempts += 1

        try:
            self.bmdl = brentq(self._find_bmdl_root, lower_bound_dose, upper_bound_dose)
            print("=" * 40)
            print("MULTI-TUMOR ANALYSIS RESULTS")
            print("=" * 40)
            print(f"Combined BMD (MLE):     {self.bmd:.4f}")
            print(f"Combined BMDL (95% LB): {self.bmdl:.4f}")
            print("=" * 40)
        except ValueError as e:
            print(f"Could not converge for Combined BMDL: {e}")
            print(f"Diagnostic -> f({lower_bound_dose:.4e}) = {f_a:.4f}, f({upper_bound_dose:.4f}) = {f_b:.4f}")


if __name__ == "__main__":
    # data = np.array([
    #     [0.00, 39,  1],
    #     [0.38, 40,  1],
    #     [1.40, 46,  4],
    #     [3.10, 42, 17]
    # ])

    # data = np.array([
    #     [0, 40, 0],
    #     [5,40,2],
    #     [10,40,4],
    #     [20,40,20],
    #     # [40,40,38]
    # ])

    # ms1 = MultistageModel(data, degree=1, lambda_reg=0, ms1_b1=0, bmd_grad=1)
    # ms1.CalcMLE()
    # ms1.CalculateBMD()
    # ms1.CalculateBMDL()
    # ms1.bootstrap_bmd(n_bootstrap=1000, confidence_level=0.95, method='p')
    # # ms1.plot_bootstrap_distribution()
    # ms1.plot_dose_response(title="Multistage Degree 1 Fit")
    # ms1.calculate_goodness_of_fit()

    # # Initialize model
    # ms2 = MultistageModel(data, degree=2, lambda_reg=0, ms1_b1=0, bmd_grad=1)
    # ms2.CalcMLE()
    # ms2.CalculateBMD()
    # ms2.CalculateBMDL()
    # # ms2.bootstrap_bmd(n_bootstrap=1000, confidence_level=0.95, method='p')
    # # ms2.plot_bootstrap_distribution()
    # ms2.plot_dose_response(title="Multistage Degree 2 Fit")
    # ms2.calculate_goodness_of_fit()

    # # Calculate optimal lambda using 1-SE rule via CV
    # best_lam = ms2.select_lambda_cv(lambda_values=range(0,100000000,100000))
    # ms2.plot_cv_results()


    # ms2r = MultistageModel(data, degree=2, lambda_reg=ms2.lambda_reg*1, ms1_b1=ms1.parameters[1], bmd_grad=1)
    # ms2r.CalcMLE()
    # ms2r.CalculateBMD()
    # ms2r.CalculateBMDL()
    # ms2r.bootstrap_bmd(n_bootstrap=1000, confidence_level=0.95, method='p')
    # # ms2r.plot_bootstrap_distribution()
    # ms2r.plot_dose_response(title="Multistage Degree 2 Reg Fit")
    # ms2r.calculate_goodness_of_fit()


    # ms2r.p_value






    # data = np.array([
    #     [0.00, 39,  1],
    #     [0.38, 40,  1],
    #     [1.40, 46,  4],
    #     [3.10, 42, 17]
    # ])

    # ms1 = MultistageModel(data, degree=1, lambda_reg=0, ms1_b1=0, bmd_grad=1)
    # ms1.CalcMLE()


    # ms1.parameters = [0.7825612, 0.05977]

    
    # ms1.nll = -neg_log_likelihood(
    #     np.array(ms1.parameters),
    #     np.array([]),
    #     ms1.doses,
    #     ms1.Ns,
    #     ms1.n,
    #     ms1.degree,
    #     ms1.ms1_b1,
    #     ms1.lambda_reg
    # )

    # ms1.CalculateBMD()
    # ms1.CalculateBMDL()
    # print(ms1.bmdl)



    ######
    # data_t1 = np.array([
    #     [0.00, 50, 39],
    #     [28.57, 50, 47],
    #     [57.14, 50, 50],
    #     [114.29, 49, 49]
    # ])

    # data_t2 = np.array([
    #     [0.00,   47, 1],
    #     [28.57,  49, 6],
    #     [57.14,  49, 8],
    #     [114.29, 47, 7]
    # ])

    #####


    # data_t1 = np.array([
    #     [0.00, 50, 39],
    #     [28.6, 50, 47],
    #     [57.1, 50, 50],
    #     [114.29, 49, 49]
    # ])

    # data_t2 = np.array([
    #     [0.00,   46, 1],
    #     [28.6,  41, 5],
    #     [57.1,  43, 7],
    #     [114.29, 47, 7]
    # ])

    # # Fit Tumor 1 (Degree 1)
    # ms1 = MultistageModel(data_t1, degree=1, lambda_reg=0, ms1_b1=0, bmd_grad=1)
    # ms1.bmr = 0.05 # Ensure BMR is set uniformly
    # ms1.CalcMLE()
    # ms1.CalculateBMD()
    # ms1.CalculateBMDL()


    # # Fit Tumor 2 (Degree 2)
    # ms2 = MultistageModel(data_t2, degree=1, lambda_reg=0, ms1_b1=0, bmd_grad=1)
    # ms2.bmr = 0.05
    # ms2.CalcMLE()
    # ms2.CalculateBMD()
    # ms2.CalculateBMDL()
    

    # # Run Multi-Tumor Analysis
    # multi_tumor = MultiTumorModel(models=[ms1, ms2], bmr=0.05, confidence_level=0.95)
    # multi_tumor.CalculateBMD()
    # multi_tumor.CalculateBMDL()

    #####

    # data = np.array([
    #     [0.00, 45, 31],
    #     [15, 50, 49]
    # ])

    # ms1 = MultistageModel(data, degree=1, lambda_reg=0, ms1_b1=0, bmd_grad=1)
    # ms1.bmr = 0.05 # Ensure BMR is set uniformly
    # ms1.CalcMLE()
    # ms1.CalculateBMD()
    # ms1.CalculateBMDL()

    #####


    data_t1 = np.array([
        [0.00, 48, 28],
        [4.00, 49, 41],
        [45.0, 48, 43],
        [87.0, 50, 48]
    ])

    data_t2 = np.array([
        [0.00, 46, 7],
        [4.00, 49, 5],
        [45.0, 48, 17],
        [87.0, 48, 12]
    ])

    # Fit Tumor 1 (Degree 1)
    # ms1 = MultistageModel(data_t1, degree=1, lambda_reg=0, ms1_b1=0, bmd_grad=1, bmr = 0.05)
    ms1 = MultistageModel(data_t1, degree=1, bmr = 0.05)
    ms1.CalcMLE()
    ms1.CalculateBMD()
    ms1.CalculateBMDL()


    # Fit Tumor 2 (Degree 2)
    ms2 = MultistageModel(data_t2, degree=1, bmr = 0.05)
    ms2.CalcMLE()
    ms2.CalculateBMD()
    ms2.CalculateBMDL()
    

    # Run Multi-Tumor Analysis
    multi_tumor = MultiTumorModel(models=[ms1, ms2], bmr=0.05, confidence_level=0.95)
    multi_tumor.CalculateBMD()
    multi_tumor.CalculateBMDL()

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torchtuples as tt
from pycox.models import CoxPH
from lifelines import CoxPHFitter
import warnings
from typing import List, Dict, Tuple, Optional, Union
import random

warnings.filterwarnings("ignore")


# ==============  TRADITIONAL COX MODEL ==============
class tCoxPH:
    """
    traditional Cox proportional hazards model.
    """

    def __init__(self, data: pd.DataFrame, time_var: str, delta_var: str, treat_var: str):
        """
        Initialize Cox model.

        Args:
            data: Input DataFrame
            time_var: Time variable name
            delta_var: Event indicator variable name
            treat_var: Treatment variable name
        """
        self.data = data
        self.time_var = time_var
        self.delta_var = delta_var
        self.treat_var = treat_var
        self.params = None
        self.p_ci_ind = None

    def compute_p_value(self) -> bool:
        """
        Fit Cox model and compute p-value based on confidence interval.

        Returns:
            True if confidence interval contains 0 (treatment not significant),
            False otherwise
        """
        # Fit Cox proportional hazards model
        cph = CoxPHFitter()
        cph.fit(self.data, duration_col=self.time_var, event_col=self.delta_var)

        # Extract coefficients
        self.params = np.array(cph.params_)

        # Check if confidence interval for treatment contains 0
        ci = cph.confidence_intervals_.loc[self.treat_var]
        self.p_ci_ind = ci[0] <= 0 <= ci[1]

        return 1.0 - self.p_ci_ind

    def compute_g_error(
        self,
        data_generator_class,
        case: int,
        gamma: float,
        binary: bool,
        rho_cor: float = 0.1,
        p_dim: int = 10,
    ) -> float:
        """
        Compute normalized mean squared error for g function estimation.

        Args:
            data_generator_class: Class for generating new data
            case: Data generation case
            gamma: Mixing parameter
            binary: Whether covariates are binary

        Returns:
            Normalized MSE: Var(ĝ - g) / Var(g)
        """
        # Generate new independent dataset
        data_gen = data_generator_class(5000, case, binary=binary, rho_cor=rho_cor, p=p_dim)
        new_data = data_gen.generate_sample(gamma)

        # Extract covariates and compute true g values
        covariates = new_data.loc[:, ~new_data.columns.isin([self.time_var, self.delta_var])].values
        true_g = np.array([data_gen.g0_values[i] for i in range(len(covariates))])
        true_g = true_g.reshape(-1, 1)

        # Compute predicted g values using Cox model coefficients
        predicted_g = covariates @ self.params.reshape(-1, 1)

        # Compute normalized MSE
        delta_g = predicted_g - true_g
        return np.var(delta_g) / np.var(true_g)


# ==============  NEURAL NETWORK COX MODEL ==============
class NeuralCoxModel:
    """
    neural network Cox model with cross-fitting for hypothesis testing.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        time_var: str,
        delta_var: str,
        treat_var: str,
        n_splits: int = 10,
        val_size: float = 0.2,
        num_nodes: List[int] = None,
        batch_norm: bool = True,
        dropout: float = 0.1,
        output_bias: bool = False,
        batch_size: int = 516,
        epochs: int = 128,
        random_state: Optional[int] = None,
    ):
        """
        Initialize neural Cox model.

        Args:
            df: Input DataFrame
            time_var: Time variable name
            delta_var: Event indicator variable name
            treat_var: Treatment variable name
            n_splits: Number of cross-validation folds
            val_size: Validation set size ratio
            num_nodes: List of hidden layer sizes
            batch_norm: Whether to use batch normalization
            dropout: Dropout rate
            output_bias: Whether to use bias in output layer
            batch_size: Batch size for training
            epochs: Number of training epochs
            random_state: Random seed for reproducibility
        """
        # Set default hidden layer sizes
        if num_nodes is None:
            num_nodes = [32] * 1

        # Store parameters
        self.time_var = time_var
        self.delta_var = delta_var
        self.treat_var = treat_var
        self.n_splits = n_splits
        self.val_size = val_size
        self.num_nodes = num_nodes
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.output_bias = output_bias
        self.batch_size = batch_size
        self.epochs = epochs

        # Store data
        self.df = df.copy()

        # Set random seeds
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
            random.seed(random_state)

        # Initialize model storage
        self.full_models = []  # Models with treatment variable
        self.sub_models = []  # Models without treatment variable
        self.rho_seq = None  # Sequence of rho values
        self.rho_seq_pvalues = []  # P-values for each rho
        self.opt_rho = None  # Optimal rho value
        self.single_pvals = None  # Single-split p-value
        self.pVals = None  # Permutation p-values

    def _prepare_survival_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract time and event data for survival analysis.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (time_array, event_array)
        """
        return df[self.time_var].values, df[self.delta_var].values

    def _prepare_features(self, df: pd.DataFrame, include_treatment: bool = True) -> np.ndarray:
        """
        Prepare and standardize features.

        Args:
            df: Input DataFrame
            include_treatment: Whether to include treatment variable

        Returns:
            Standardized feature matrix
        """
        # Define columns to exclude
        exclude_cols = [self.time_var, self.delta_var]
        if not include_treatment:
            exclude_cols.extend(self.treat_var)

        # Extract features
        features = df.loc[:, ~df.columns.isin(exclude_cols)]
        return StandardScaler().fit_transform(features).astype("float32")

    def _train_neural_cox_model(
        self,
        x_train: np.ndarray,
        y_train: Tuple[np.ndarray, np.ndarray],
        x_val: np.ndarray,
        y_val: Tuple[np.ndarray, np.ndarray],
    ) -> CoxPH:
        """
        Train a neural network Cox model.

        Args:
            x_train, y_train: Training data
            x_val, y_val: Validation data

        Returns:
            Trained CoxPH model
        """
        input_dim = x_train.shape[1]

        # Build neural network
        net = tt.practical.MLPVanilla(
            input_dim,
            self.num_nodes,
            1,
            self.batch_norm,
            self.dropout,
            output_bias=self.output_bias,
        )
        model = CoxPH(net, tt.optim.Adam)

        # Find optimal learning rate
        lr_finder = model.lr_finder(x_train, y_train, self.batch_size, tolerance=100)
        lr = lr_finder.get_best_lr() / 2.0
        model.optimizer.set_lr(lr)

        # Train with early stopping
        callbacks = [tt.callbacks.EarlyStopping()]
        model.fit(
            x_train,
            y_train,
            self.batch_size,
            self.epochs,
            callbacks,
            verbose=False,
            val_data=(x_val, y_val),
            val_batch_size=self.batch_size,
        )

        return model

    def compute_cv_pvalue(
        self,
        df: pd.DataFrame = None,
        rho: float = 0.1,
        rho_seq: List[float] = None,
        mse_error: bool = False,
    ) -> Tuple[float, float]:
        """
        Compute cross-validated p-value for treatment effect and single-split p-value.

        Args:
            df: Input DataFrame (uses self.df if None)
            rho: Perturbation size
            rho_seq: Sequence of rho values to evaluate

        Returns:
            Tuple of (cross-validated p-value, single-split p-value)
        """
        if df is None:
            df = self.df

        # Set default rho sequence
        if rho_seq is None:
            rho_seq = np.logspace(
                -3.0, 1.0, 201
            ).tolist()  # [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        self.rho_seq = rho_seq

        # Initialize cross-validation
        kf = RepeatedKFold(n_splits=self.n_splits, n_repeats=1)

        # Reset model storage
        self.full_models = []
        self.sub_models = []

        # Store likelihood differences for all test samples
        n_total = len(df)
        all_likelihood_diffs = np.zeros(n_total)

        # Variables for single-split p-value (last fold)
        last_fold_data = None

        if mse_error:
            mse_count = 0

        # Cross-validation loop
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df)):
            if mse_error:
                mse_count += 1
                if mse_count > 1:
                    continue
            # Split data
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            # Further split training data into train and validation
            train_df, val_df = train_test_split(train_df, test_size=self.val_size)

            # Prepare data for full model (with treatment)
            x_train_full = self._prepare_features(train_df, include_treatment=True)
            x_val_full = self._prepare_features(val_df, include_treatment=True)
            x_test_full = self._prepare_features(test_df, include_treatment=True)

            # Prepare data for sub model (without treatment)
            x_train_sub = self._prepare_features(train_df, include_treatment=False)
            x_val_sub = self._prepare_features(val_df, include_treatment=False)
            x_test_sub = self._prepare_features(test_df, include_treatment=False)

            # Prepare survival data
            y_train = self._prepare_survival_data(train_df)
            y_val = self._prepare_survival_data(val_df)
            y_test = self._prepare_survival_data(test_df)

            # Train full model
            full_model = self._train_neural_cox_model(x_train_full, y_train, x_val_full, y_val)

            # Train sub model
            sub_model = self._train_neural_cox_model(x_train_sub, y_train, x_val_sub, y_val)

            # Store models with their scalers
            scaler_full = StandardScaler().fit(
                train_df.loc[:, ~train_df.columns.isin([self.time_var, self.delta_var])]
            )
            scaler_sub = StandardScaler().fit(
                train_df.loc[
                    :, ~train_df.columns.isin([self.time_var, self.delta_var, self.treat_var])
                ]
            )

            self.full_models.append([scaler_full, full_model.net])
            self.sub_models.append([scaler_sub, sub_model.net])

            # Compute likelihood differences for test set
            ll_full = full_model.partial_log_likelihood(x_test_full, y_test)
            ll_sub = sub_model.partial_log_likelihood(x_test_sub, y_test)
            total_diff = ll_full - ll_sub

            # Only event times contribute to partial likelihood
            event_mask = test_df[self.delta_var] == 1
            n_events = event_mask.sum()

            if n_events > 0:
                likelihood_diffs = np.zeros(len(test_df))
                # Distribute total difference among events
                likelihood_diffs[event_mask] = total_diff / n_events
                all_likelihood_diffs[test_idx] = likelihood_diffs

            # Store last fold data for single-split p-value
            if fold_idx == self.n_splits - 1:  # Last fold
                last_fold_data = {
                    "test_indices": test_idx,
                    "test_df": test_df,
                    "n_test": len(test_df),
                    "n_events": n_events,
                    "total_diff": total_diff,
                    "x_test_full": x_test_full,
                    "x_test_sub": x_test_sub,
                    "y_test": y_test,
                }

        # Generate perturbation noise for cross-validated p-value
        noise = np.random.randn(n_total)

        # Compute cross-validated p-value for given rho
        perturbed_likelihood = all_likelihood_diffs + rho * noise
        _, cv_pval = stats.ttest_1samp(perturbed_likelihood, popmean=0.0, alternative="greater")

        # Compute single-split p-value using the last fold
        if last_fold_data is not None:
            test_df = last_fold_data["test_df"]
            n_test = last_fold_data["n_test"]
            n_events = last_fold_data["n_events"]
            total_diff = last_fold_data["total_diff"]

            # Create single-split test array
            single_test_array = np.zeros(n_test)

            if n_events > 0:
                # Get event indices in the test set
                event_indices = np.where(test_df[self.delta_var] == 1)[0]

                # Calculate per-event difference
                per_event_diff = total_diff / n_events

                # Ensure event indices are within bounds
                valid_event_indices = event_indices[event_indices < n_test]

                # Assign per-event difference to event positions
                if len(valid_event_indices) > 0:
                    single_test_array[valid_event_indices] = per_event_diff

            # Generate noise for single-split test
            single_noise = np.random.randn(n_test)
            single_perturbed = single_test_array + rho * single_noise

            # Compute single-split p-value
            if n_test > 1 and np.std(single_perturbed) > 0:
                _, self.single_pvals = stats.ttest_1samp(
                    single_perturbed, popmean=0.0, alternative="greater"
                )
            else:
                self.single_pvals = 1.0
        else:
            self.single_pvals = 1.0

        # Compute p-values for all rho values in sequence
        self.rho_seq_pvalues = []
        for rho_val in rho_seq:
            t_stat = all_likelihood_diffs + rho_val * noise
            _, rho_pval = stats.ttest_1samp(t_stat, popmean=0.0, alternative="greater")
            self.rho_seq_pvalues.append(rho_pval)

        return cv_pval  # , self.single_pvals

    def select_optimal_rho(
        self, alpha: float = 0.05, permutation_nRep: int = 20, rho_seq: List[float] = None
    ) -> float:
        """
        Select optimal rho using permutation test.

        Args:
            alpha: Nominal significance level
            permutation_nRep: Number of permutations

        Returns:
            Optimal rho value
        """
        # Set default rho sequence
        if rho_seq is None:
            rho_seq = np.logspace(-3.0, 1.0, 201).tolist()
        self.rho_seq = rho_seq

        # Store p-values from permutations
        pvals_list = []

        for _ in range(permutation_nRep):
            # Permute treatment variable
            permuted_df = self.df.copy()
            permuted_df[self.treat_var] = np.random.choice(
                permuted_df[self.treat_var].values, size=len(permuted_df), replace=False
            )

            # Compute p-values for rho sequence on permuted data
            model_copy = NeuralCoxModel(
                permuted_df,
                self.time_var,
                self.delta_var,
                self.treat_var,
                n_splits=self.n_splits,
                val_size=self.val_size,
                num_nodes=self.num_nodes,
                batch_norm=self.batch_norm,
                dropout=self.dropout,
                output_bias=self.output_bias,
                batch_size=self.batch_size,
                epochs=self.epochs,
            )
            _ = model_copy.compute_cv_pvalue(rho_seq=self.rho_seq)
            pvals_list.append(model_copy.rho_seq_pvalues)

        self.pVals = np.array(pvals_list)

        # Compute coverage probabilities (proportion of p-values < alpha)
        cover_prob = np.mean(self.pVals < alpha, axis=0)

        # Select rho that gives coverage closest to alpha
        diff_from_alpha = np.abs(cover_prob - alpha)
        self.opt_rho = self.rho_seq[np.argmin(diff_from_alpha)]

        return self.opt_rho

    def compute_final_pvalue(self, permutation_nRep: int = 20) -> float:
        """
        Compute final p-value using optimal rho selected via permutation.

        Args:
            permutation_nRep: Number of permutations for rho selection

        Returns:
            Final p-value
        """
        # Select optimal rho
        self.opt_rho = self.select_optimal_rho(permutation_nRep=permutation_nRep)

        # Compute final p-value with optimal rho
        return self.compute_cv_pvalue(rho=self.opt_rho)

    def compute_g_error(
        self,
        data_generator_class,
        case: int,
        gamma: float,
        binary: bool,
        rho_cor: float = 0.1,
        p_dim: int = 10,
    ) -> Tuple[float, float]:
        """
        Compute normalized MSE for g function estimation.

        Args:
            data_generator_class: Class for generating new data
            case: Data generation case
            gamma: Mixing parameter
            binary: Whether covariates are binary

        Returns:
            Tuple of (MSE_full, MSE_sub) - MSE for models with and without treatment
        """
        mse_full_list = []
        mse_sub_list = []

        # Device for neural network computation
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for scaler_full, net_full in self.full_models:
            # Generate new independent dataset
            data_gen = data_generator_class(5000, case, binary=binary, rho_cor=rho_cor, p=p_dim)
            new_data = data_gen.generate_sample(gamma)
            true_g = data_gen.g0_values.reshape(-1, 1)

            # Prepare features for full model
            features_full = new_data.loc[:, ~new_data.columns.isin([self.time_var, self.delta_var])]
            features_scaled = scaler_full.transform(features_full).astype("float32")

            # Predict g values using neural network
            features_tensor = torch.from_numpy(features_scaled).to(device)
            with torch.no_grad():
                predicted_g = net_full(features_tensor).cpu().numpy()

            # Compute normalized MSE
            delta_g = true_g - predicted_g
            mse_full_list.append(np.var(delta_g) / np.var(true_g))

        for scaler_sub, net_sub in self.sub_models:
            # Generate new independent dataset
            data_gen = data_generator_class(5000, case, binary=binary, rho_cor=rho_cor, p=p_dim)
            new_data = data_gen.generate_sample(gamma)
            true_g = data_gen.g0_values.reshape(-1, 1)

            # Prepare features for sub model (without treatment)
            features_sub = new_data.loc[
                :, ~new_data.columns.isin([self.time_var, self.delta_var, self.treat_var])
            ]
            features_scaled = scaler_sub.transform(features_sub).astype("float32")

            # Predict g values using neural network
            features_tensor = torch.from_numpy(features_scaled).to(device)
            with torch.no_grad():
                predicted_g = net_sub(features_tensor).cpu().numpy()

            # Compute normalized MSE
            delta_g = true_g - predicted_g
            mse_sub_list.append(np.var(delta_g) / np.var(true_g))

        return np.mean(mse_full_list), np.mean(mse_sub_list)

    def compute_multi_split_pvalue(
        self, df: pd.DataFrame = None, rho: float = 0.1, rho_seq: List[float] = None
    ) -> bool:
        """
        Compute p-value using multiple random splits method.

        Args:
            df: Input DataFrame (uses self.df if None)
            rho: Perturbation size
            rho_seq: Sequence of rho values (not used in this method but kept for compatibility)

        Returns:
            Boolean indicating whether to reject null hypothesis (True = reject)
        """
        if df is None:
            df = self.df

        # Store rho sequence for compatibility
        if rho_seq is None:
            rho_seq = np.logspace(-3, 1.0, 201).tolist()
        self.rho_seq = rho_seq

        # Reset model storage
        self.full_models = []
        self.sub_models = []

        # Generate multiple random splits
        n_total = len(df)
        multi_splits = []
        for _ in range(self.n_splits):
            multi_splits.append(random_split_range(n_total, ratio=(self.n_splits - 1, 1)))

        # Initialize test statistic accumulator
        aggregated_test_statistic = 0.0

        # Process each split
        for split_idx, (train_idx, test_idx) in enumerate(multi_splits):
            # Split data
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            # Further split training data into train and validation
            train_df, val_df = train_test_split(train_df, test_size=self.val_size)

            # Prepare survival data
            y_train = self._prepare_survival_data(train_df)
            y_val = self._prepare_survival_data(val_df)
            y_test = self._prepare_survival_data(test_df)

            # Initialize likelihood difference
            likelihood_diff = 0.0

            # Train both full and sub models
            for include_treatment in [True, False]:
                # Prepare features
                x_train = self._prepare_features(train_df, include_treatment=include_treatment)
                x_val = self._prepare_features(val_df, include_treatment=include_treatment)
                x_test = self._prepare_features(test_df, include_treatment=include_treatment)

                # Train model
                model = self._train_neural_cox_model(x_train, y_train, x_val, y_val)

                # Compute test likelihood
                test_likelihood = model.partial_log_likelihood(x_test, y_test)

                # Store model with scaler
                if include_treatment:
                    scaler = StandardScaler().fit(
                        train_df.loc[:, ~train_df.columns.isin([self.time_var, self.delta_var])]
                    )
                    self.full_models.append([scaler, model.net])
                    likelihood_diff = test_likelihood - likelihood_diff
                else:
                    scaler = StandardScaler().fit(
                        train_df.loc[
                            :,
                            ~train_df.columns.isin([self.time_var, self.delta_var, self.treat_var]),
                        ]
                    )
                    self.sub_models.append([scaler, model.net])
                    likelihood_diff = likelihood_diff - test_likelihood

            # Process test results
            n_test = len(test_df)
            n_events = test_df[self.delta_var].sum()

            # Create array of likelihood differences (only events contribute)
            likelihood_array = np.zeros(n_test)
            if n_events > 0:
                # Note: This assumes all events have the same likelihood difference
                # In practice, you might want to distribute the difference among events
                likelihood_array[: min(n_events, n_test)] = likelihood_diff / max(n_events, 1)

            # Generate perturbation noise
            noise = np.random.randn(n_test)
            perturbed_likelihood = likelihood_array + rho * noise

            # Compute test statistic for this split
            if n_test > 1 and perturbed_likelihood.std() > 0:
                split_statistic = (
                    np.sqrt(n_test) * perturbed_likelihood.mean() / perturbed_likelihood.std()
                )
                aggregated_test_statistic += split_statistic

        # Compute average test statistic
        if self.n_splits > 0:
            average_statistic = aggregated_test_statistic / self.n_splits
        else:
            average_statistic = 0.0

        # Critical value
        critical_value = 2.0642913018240976

        # Return decision (True = reject null hypothesis)
        return average_statistic > critical_value


def random_split_range(n: int, ratio: Tuple[int, int] = (8, 2)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly split a range of indices into two parts based on given ratio.

    Args:
        n: Total number of indices
        ratio: Tuple of (size_first_part, size_second_part)

    Returns:
        Tuple of (first_part_indices, second_part_indices)
    """
    # Create list of indices and shuffle
    numbers = list(range(n))
    random.shuffle(numbers)

    # Calculate partition sizes
    total_parts = sum(ratio)
    first_part_size = (n * ratio[0]) // total_parts
    second_part_size = n - first_part_size

    # Split indices
    first_part = numbers[:first_part_size]
    second_part = numbers[first_part_size:]

    return np.array(first_part), np.array(second_part)

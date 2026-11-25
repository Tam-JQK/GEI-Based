from sklearn.metrics import roc_auc_score
import numpy as np
from typing import Union, List, Dict, Optional
import warnings

##############ranking metrics #############################
# def Recall(ranking_list, label_list, k):
#     r"""Recall_ (also known as sensitivity) measures the proportion of 
#         relevant items that are successfully recommended in the top-k list.

#         .. math::
#             \mathrm {Recall@K} = \frac{|\hat{R}(u) \cap R(u)|}{|R(u)|}

#         :math:`\hat{R}(u)` is the top-k recommended items.
#         :math:`R(u)` is the set of all relevant items for user u.
#     """
#     # Number of relevant items retrieved.
#     hits = np.sum(ranking_list[:k])
#     # Total number of relevant items.
#     total_positives = np.sum(label_list)
    
#     if total_positives == 0:
#         return 0.0
    
#     return hits / total_positives


def Recall(ranking_list, k, total_pos_count):
    """
    Recall@K = relevant items retrieved / all relevant items for the user.
    
    :param ranking_list: Binary relevance indicators in ranked order.
    :param k: top-k
    :param total_pos_count: Total number of relevant items for the user.
    """
    hits = np.sum(ranking_list[:k])
    
    if total_pos_count == 0:
        return 0.0
    
    return hits / total_pos_count


def AUC_score(y_scores, y_true):
    r"""AUC_ (also known as Area Under Curve) is used to evaluate the two-class model, referring to
        the area under the ROC curve.

        .. _AUC: https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

        Note:
            This metric does not calculate group-based AUC which considers the AUC scores
            averaged across users. It is also not limited to k. Instead, it calculates the
            scores on the entire prediction results regardless the users. We call the interface
            in `scikit-learn`, and code calculates the metric using the variation of following formula.

        .. math::
            \mathrm {AUC} = \frac {{{M} \times {(N+1)} - \frac{M \times (M+1)}{2}} -
            \sum\limits_{i=1}^{M} rank_{i}} {{M} \times {(N - M)}}

        :math:`M` denotes the number of positive items.
        :math:`N` denotes the total number of user-item interactions.
        :math:`rank_i` denotes the descending rank of the i-th positive item.
    """
    auc_score = roc_auc_score(y_true, y_scores)
    return auc_score

def dcg(scores, k):
    """
    Calculate the Discounted Cumulative Gain (DCG) at rank k.
    """
    scores = np.array(scores)[:k]
    if scores.size:
        return np.sum(scores / np.log2(np.arange(2, scores.size + 2)))
    return 0.0


def NDCG(ranking_list, label_list, k):
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality,
    where positions are discounted logarithmically. It accounts for the position of the hit by assigning
    higher scores to hits at top ranks.

    .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    .. math::
        \mathrm {NDCG@K} = \frac{1}{|U|}\sum_{u \in U} (\frac{1}{\sum_{i=1}^{\min (|R(u)|, K)}
        \frac{1}{\log _{2}(i+1)}} \sum_{i=1}^{K} \delta(i \in R(u)) \frac{1}{\log _{2}(i+1)})

    :math:`\delta(·)` is an indicator function.
    """
    sorted_label = np.sort(label_list)[::-1]
    label_dcg = dcg(sorted_label, k)
    ranking_dcg = dcg(ranking_list, k)
    ndcg = ranking_dcg/label_dcg

    return ndcg

def HR(ranking_list, label_list, k):
    r"""Return standard Hit Rate for one user: 1 if Top-K contains any positive,
        otherwise 0. Evaluators average these indicators across users.
        ``ranking_list`` contains binary relevance labels in recommendation order.
        ``label_list`` is retained for compatibility with existing callers.

        .. math::
            \mathrm {HR@K} = \frac{1}{|U|}\sum_{u \in U} \delta(\hat{R}(u) \cap R(u) \neq \emptyset),

        :math:`\delta(·)` is an indicator function. :math:`\delta(b)` = 1 if :math:`b` is true and 0 otherwise.
        :math:`\emptyset` denotes the empty set.
    """
    return float(np.any(np.asarray(ranking_list)[:k] > 0))


def MRR(ranking_list, k):
    r"""The MRR_ (also known as Mean Reciprocal Rank) computes the reciprocal rank
        of the first relevant item found by an algorithm.

        .. _MRR: https://en.wikipedia.org/wiki/Mean_reciprocal_rank

        .. math::
           \mathrm {MRR@K} = \frac{1}{|U|}\sum_{u \in U} \frac{1}{\operatorname{rank}_{u}^{*}}

        :math:`{rank}_{u}^{*}` is the rank position of the first relevant item found by an algorithm for a user :math:`u`.
    """
    mrr = 0
    for index, i in enumerate(ranking_list[:k]):
        if i > 0:
            mrr = i/(index+1)
            break
    return mrr


##################fairness metric#######################

def reconstruct_utility(utility_list, weights, group_mask):
    """
        Reconstruct utility by re-weighting them and masking the utility of certain unused groups.
        :param utility_list: array for item/user utilities
        :param weights: array for item/user utilities weights
        :param group_mask: bool array for whether computed the group utilityes
        :return: re-constructed utility array
    """

    if not weights:
        weights = np.ones_like(utility_list)

    utility_list = np.array(utility_list)
    weights = np.array(weights)
    weighted_utility = utility_list * weights

    if group_mask:
        weighted_utility = mask_utility(weighted_utility, group_mask)


    return np.array(weighted_utility)

def mask_utility(utility, group_mask):
    """
    Mask the utility values based on the provided group mask.

    This function filters out the utility values where the corresponding
    group mask element is zero, effectively removing them from the output.


    :param utility: array for item/user utilities
    :param group_mask: bool array for whether computed the group utilityes
    :return: masked utility array
    """


    masked_utility = []
    for i, m in enumerate(group_mask):
        if m == 0:
            masked_utility.append(utility[i])

    return np.array(masked_utility)

def MMF(utility_list, ratio=0.5, weights=None, group_mask = None):
    """
    Calculate the Max-min Fairness (MMF) index based on a given utility list.

    Parameters
    :param utility: array-like
        A list or array representing the utilities of resources or users.
    :param ratio: float, optional
        The fraction of the minimum utilities to consider for the MMF calculation. Defaults to 0.5.
    :param weights : array-like, optional
        An optional list or array of weights corresponding to each utility in `utility_list`.
        If provided, utilities are multiplied by their respective weights before sorting.
        Defaults to None, implying equal weighting.
    :param group_mask : array-like, optional
        An optional list or array used to selectively apply weights. If provided, it must have the same length as
        `utility_list` and `weights`. Defaults to None, indicating no group-based weighting.

    :return: The computed MMF index, representing the fairness of the allocation.
    """
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)

    MMF_length = int(ratio * len(utility_list))
    utility_sort = np.sort(weighted_utility)

    mmf = np.sum(utility_sort[:MMF_length])/np.sum(weighted_utility)

    return mmf

def MinMaxRatio(utility_list, weights=None, group_mask = None):
    """
    This function computes the minimum-to-maximum ratio of a list of utilities, optionally weighted and grouped.

    :param utility_list (list of float): A list containing numerical utility values.
    :param weights (list of float, optional): A list of weights corresponding to the utilities in utility_list. If provided,
      each utility is multiplied by its respective weight. If None, all utilities are considered with equal weight.
    :param group_mask (list of int or bool, optional): A mask indicating groups within the utility_list. If provided, it must
      be of the same length as utility_list. Groups are defined by consecutive True or 1 values. If None, no grouping is applied.

    :return: float: The computed minimum-to-maximum ratio of the (weighted) utilities.
    """
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)
    return np.min(weighted_utility) / np.max(weighted_utility)
    
def Gini(utility_list, weights=None, group_mask = None):
    """
    This function computes the Gini coefficient, a measure of statistical dispersion intended to represent income inequality within a nation or social group.
    The Gini coefficient is calculated based on the cumulative distribution of values in `utility_list`, which can optionally be weighted and masked.

    :param utility_list: array_like
        A 1D array representing individual utilities. The utilities are used to compute the Gini coefficient.
    :param weights: array_like, optional
        A 1D array of weights corresponding to `utility_list`. If provided, each utility value is multiplied by its respective weight before calculating the Gini coefficient. Defaults to None, implying equal weighting.
    :param group_mask: array_like, optional
        A 1D boolean array used to selectively include elements from `utility_list`. If provided, only the elements where the mask is True are considered in the calculation. Defaults to None, meaning all elements are included.

    :return: float: The computed Gini coefficient, ranging from 0 (perfect equality) to 1 (maximal inequality).
    """
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)

    values = np.sort(weighted_utility)
    n = len(values)

    # gini compute
    cumulative_sum = np.cumsum(values)
    gini = (n + 1 - 2 * (np.sum(cumulative_sum) / cumulative_sum[-1])) / n

    
    
    return gini

def Entropy(utility_list, weights=None, group_mask = None):
    """
    Calculate the entropy of a distribution given by `utility_list`, optionally
    weighted by `weights` and filtered by `group_mask`. Entropy measures the
    disorder or uncertainty in the distribution.

    :param utility_list: list or array-like
        A list or array representing utility values for each item.
    :param weights: list or array-like, optional
        A list or array of weights corresponding to each utility value. If not provided,
        all utilities are considered equally weighted. Defaults to None.
    :param group_mask : list or array-like, optional
        A boolean mask indicating which utilities to include in the calculation.
        If not provided, all utilities are included. Defaults to None.

    :return: float: The calculated entropy of the (potentially weighted and masked) distribution.

    Notes
    - Entropy is calculated as H = -sum(p * log2(p)), where p is the probability of each event.
    - Probabilities are normalized to ensure their sum equals 1.
    - Zero-probability terms use the exact limit 0 * log2(0) = 0.
    """
    weighted_utility = reconstruct_utility(utility_list, weights, group_mask)

    values = np.array(weighted_utility)
    values = values / np.sum(values)

    # Shannon entropy on the original exposure shares; GE-only smoothing must
    # not change this metric. Masking also makes a single exposed provider H=0.
    positive = values > 0
    entropy_value = -np.sum(values[positive] * np.log2(values[positive]))
    return entropy_value

def ElasticFair(utils,t):
    """
       Calculate the elasticfair metric from the paper of SIGIR 2025:
       Understanding Accuracy-Fairness Trade-offs in Re-ranking through Elasticity in Economics


       :param utils: list or array-like
           A list or array representing utility values for each item.
       :param t: taxation rate
           different t will align with different fairness metric
           for example t=0: Entropy, t=1: Nash fairness, t=\infty: max-min fairness

       :return: float: The fairness metric value under specific t.
    """

    if t != 0:
        sign = np.sign(1-t)
        utils_g = np.sum(np.power(utils,1-t))
        utils_g = np.power(utils_g, 1/t)
        return sign * utils_g
    else:
        entropy = - np.sum(utils * np.log(utils))
        return np.exp(entropy)

def EF(utility_list, M = 50, weights=None, group_mask = None):
    """
       Calculate the EF metric from the paper of SIGIR 2025:
       Understanding Accuracy-Fairness Trade-offs in Re-ranking through Elasticity in Economics


       :param utils: list or array-like
           A list or array representing utility values for each item.
       :param M: bound of the integral

       :return: float: The fairness metric EF value.
    """

    utility_list = reconstruct_utility(utility_list, weights, group_mask)
    utility_list = utility_list / np.sum(utility_list)

    t = np.linspace(1 - M, 1 + M, 200)
    fair = []
    for i in t:
        fair.append(ElasticFair(utility_list, i))
    integral = np.trapz(fair, t)
    return integral/(2*M*len(utility_list))




# =============================================================================
# Generalized entropy index (GEI)
# Scale-relative smoothing
# =============================================================================
# ---------------------------------------------------------------------------
# Equation (7) adds epsilon = eta times the original mean exposure.
# GE_EPS is an alias for the dimensionless coefficient GE_ETA.
GE_ETA = 1e-12
GE_EPS = GE_ETA


def _smooth_ge_exposures(E, eta):
    """Apply Eq. (7) once to the complete provider exposure vector."""
    E = np.asarray(E, dtype=float)
    if E.ndim != 1 or not np.all(np.isfinite(E)) or np.any(E < 0):
        raise ValueError('GE exposures must be a finite, nonnegative 1-D vector')
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError('GE smoothing eta must be finite and positive')
    if E.size == 0:
        return E.copy()
    scale = E.max()
    if scale == 0:
        return E.copy()
    # GE and its decomposition depend only on ratios. Work in scaled units to
    # apply exactly the same relative smoothing without overflow/underflow in
    # the original exposure units (e.g. all exposures near 1e308 or 1e-300).
    scaled = E / scale
    epsilon = eta * scaled.mean()
    if epsilon == 0:
        raise ValueError('GE smoothing eta is too small for floating-point arithmetic')
    smoothed = scaled + epsilon
    return smoothed / smoothed.max()


def GE_alpha(E: Union[np.ndarray, List[float]], alpha: float,
             eta: float = GE_ETA) -> float:
    """Compute the generalized entropy index of nonnegative exposures.

    For exposure E_i and mean μ:
        GE(α) = (mean((E_i/μ)^α) - 1) / (α(α - 1)) for α outside {0, 1}.
        GE(0) = mean(log(μ/E_i)), the mean log deviation.
        GE(1) = mean((E_i/μ) * log(E_i/μ)), the Theil index.

    Scale-relative smoothing adds eta times the original mean to every
    exposure. The calculation includes entries with zero original exposure.

    Args:
        E: Nonnegative exposure values.
        alpha: Sensitivity parameter. Lower values emphasize low exposures,
            while larger values emphasize concentration at the upper end.
        eta: Dimensionless coefficient for scale-relative smoothing.

    Returns:
        The generalized entropy index. Equal exposures give zero.
    """
    return _ge_alpha_raw(_smooth_ge_exposures(E, eta), alpha)

# ---------------------------------------------------------------------------
# Common special cases
# ---------------------------------------------------------------------------

def MLD(E: Union[np.ndarray, List[float]], eta: float = GE_ETA) -> float:
    """Return mean log deviation, GE(0)."""
    return GE_alpha(E, alpha=0, eta=eta)


def Theil(E: Union[np.ndarray, List[float]], eta: float = GE_ETA) -> float:
    """Return the Theil index, GE(1)."""
    return GE_alpha(E, alpha=1, eta=eta)


def HalfCV2(E: Union[np.ndarray, List[float]], eta: float = GE_ETA) -> float:
    """Return half the squared coefficient of variation, GE(2)."""
    return GE_alpha(E, alpha=2, eta=eta)

# ---------------------------------------------------------------------------
# Generalized entropy on an already smoothed vector
# ---------------------------------------------------------------------------

def _ge_alpha_raw(E: np.ndarray, alpha: float) -> float:
    """Compute generalized entropy from already smoothed exposures."""
    if not np.isscalar(alpha) or not np.isfinite(alpha):
        raise ValueError('GE alpha must be finite')
    if len(E) <= 1:
        return 0.0

    mu = E.mean()

    if mu <= 0:
        return 0.0

    ratio = E / mu

    if abs(alpha) < 1e-9:
        return float(-np.mean(np.log(ratio)))

    if abs(alpha - 1.0) < 1e-9:
        return float(np.mean(ratio * np.log(ratio)))

    term = np.mean(ratio ** alpha)
    return float((term - 1.0) / (alpha * (alpha - 1.0)))


# ---------------------------------------------------------------------------
# Provider-level decomposition: GE_total = GE_between + GE_within.
# ---------------------------------------------------------------------------
def compute_GE_decomposition(
    provider_exposures: np.ndarray,
    pid2mgid: dict,
    alpha: float,
    meta_group_num: int,
    eta: float = GE_ETA,
    residual_tol: float = 1e-6
) -> Dict[str, float]:
    """Decompose provider-level GE into between- and within-group components.

    Smooth the full provider exposure vector once. Use that same vector for
    the overall index, group means, and within-group indices.

    Args:
        provider_exposures: One exposure value per provider.
        pid2mgid: Mapping from integer provider IDs to integer meta-group IDs.
        alpha: Generalized entropy sensitivity parameter.
        meta_group_num: Number of provider meta-groups.
        eta: Dimensionless coefficient for scale-relative smoothing.
        residual_tol: Absolute tolerance for the decomposition residual.

    Returns:
        A dictionary containing the overall index (ge_total), between-group
        component (ge_between), weighted within-group component (ge_within),
        between-group share (between_ratio), absolute decomposition residual
        (decomposition_residual), and exposure shares (exp_ratio_mg_{g}).
    """
    provider_exposures = _smooth_ge_exposures(provider_exposures, eta)
    provider_num = len(provider_exposures)
    if (isinstance(meta_group_num, bool) or not isinstance(meta_group_num, (int, np.integer))
            or meta_group_num < 1 or not np.isscalar(alpha) or not np.isfinite(alpha)):
        raise ValueError('GE requires finite alpha and a positive integer meta_group_num')

    results = {
        'ge_total': 0.0,
        'ge_between': 0.0,
        'ge_within': 0.0,
        'between_ratio': 0.0,
        'decomposition_residual': 0.0,
    }
    for g in range(meta_group_num):
        results[f'exp_ratio_mg_{g}'] = 0.0

    if provider_num == 0:
        return results

    # Use the same smoothed exposure vector throughout the decomposition.

    N_total = provider_num
    mu_total = provider_exposures.mean()
    total_exposure_sum = provider_exposures.sum()

    if mu_total <= 0:
        return results

    # Compute overall inequality across providers.
    results['ge_total'] = _ge_alpha_raw(provider_exposures, alpha)

    # Collect provider exposures by meta-group.
    mg_exposures = {g: [] for g in range(meta_group_num)}
    missing_count = 0

    for pid in range(provider_num):
        mgid = pid2mgid.get(pid)
        if mgid is None or not (0 <= mgid < meta_group_num):
            missing_count += 1
            mgid = 0  # fallback
        mg_exposures[mgid].append(provider_exposures[pid])

    if missing_count > 0:
        warnings.warn(
            f"compute_GE_decomposition: {missing_count} providers "
            f"missing or invalid in pid2mgid, assigned to meta-group 0."
        )

    for g in range(meta_group_num):
        mg_exposures[g] = (np.array(mg_exposures[g])
                           if mg_exposures[g]
                           else np.array([], dtype=float))

    # Compute per-group summary statistics.
    mg_stats = {}

    for g in range(meta_group_num):
        exps = mg_exposures[g]

        if len(exps) == 0:
            mg_stats[g] = {
                'N_g': 0, 'mu_g': 0.0, 'sum_g': 0.0,
                'share': 0.0, 'ge_g': 0.0,
            }
        else:
            N_g = len(exps)
            mu_g = exps.mean()
            sum_g = exps.sum()
            share = N_g / N_total
            ge_g = _ge_alpha_raw(exps, alpha) if N_g > 1 else 0.0

            mg_stats[g] = {
                'N_g': N_g, 'mu_g': mu_g, 'sum_g': sum_g,
                'share': share, 'ge_g': ge_g,
            }

        # Store the exposure share of this group.
        results[f'exp_ratio_mg_{g}'] = float(
            mg_stats[g]['sum_g'] / total_exposure_sum
        ) if total_exposure_sum > 0 else 0.0

    # Compute between-group inequality.
    representative_values = []
    for g in range(meta_group_num):
        stats = mg_stats[g]
        if stats['N_g'] > 0:
            representative_values.extend([stats['mu_g']] * stats['N_g'])

    if len(representative_values) > 1:
        results['ge_between'] = _ge_alpha_raw(
            np.array(representative_values), alpha
        )

    # Sum the weighted within-group indices.
    ge_within = 0.0

    for g in range(meta_group_num):
        stats = mg_stats[g]
        if stats['N_g'] == 0 or stats['mu_g'] <= 0:
            continue

        share = stats['share']             # N_g / N
        ratio = stats['mu_g'] / mu_total   # μ_g / μ

        # Group weight: w_g = (N_g/N) * (μ_g/μ)^α.
        if abs(alpha) < 1e-9:
            weight = share                 # α=0
        elif abs(alpha - 1.0) < 1e-9:
            weight = share * ratio         # α=1
        else:
            weight = share * (ratio ** alpha)

        ge_within += weight * stats['ge_g']

    results['ge_within'] = float(ge_within)

    # Compute the decomposition residual.
    residual = abs(
        results['ge_total'] - results['ge_between'] - results['ge_within']
    )
    results['decomposition_residual'] = float(residual)

    if residual > residual_tol and results['ge_total'] > 1e-12:
        warnings.warn(
            f"GE decomposition residual = {residual:.2e} "
            f"(total={results['ge_total']:.6f}, "
            f"between={results['ge_between']:.6f}, "
            f"within={results['ge_within']:.6f})"
        )

    # Express the between-group component as a share of total inequality.
    if results['ge_total'] > 1e-12:
        results['between_ratio'] = results['ge_between'] / results['ge_total']
    else:
        results['between_ratio'] = 0.0

    return results

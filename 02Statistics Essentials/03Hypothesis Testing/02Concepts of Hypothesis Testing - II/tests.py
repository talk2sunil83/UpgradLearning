# %%

import re
from typing import Callable, Tuple, Union
import numpy as np
from enum import Enum, auto
from scipy.stats import norm, t, chi2
from scipy import special

# %%
decimal_limit = 4


class TestType(Enum):
    # Non-directional
    DOUBLE_TAILED = auto()
    # directional
    LOWER_TAILED = auto()
    UPPER_TAILED = auto()


# https://stackoverflow.com/questions/3496656/convert-z-score-z-value-standard-score-to-p-value-for-normal-distribution-in/3508321

# %%
def get_standard_error(σ: float, n: int) -> float:
    return round(σ / np.sqrt(n), decimal_limit)


def get_p_value_normal(z_score: float) -> float:
    """get p value for normal(Gaussian) distribution 

    Args:
        z_score (float): z score

    Returns:
        float: p value
    """
    return round(norm.sf(z_score), decimal_limit)


def get_p_value_t(z_score: float) -> float:
    """get p value for t distribution 

    Args:
        z_score (float): z score

    Returns:
        float: p value
    """
    return round(t.sf(z_score), decimal_limit)


def get_p_value_chi2(z_score: float) -> float:
    """get p value for chi2 distribution 

    Args:
        z_score (float): z score

    Returns:
        float: p value
    """
    return round(chi2.ppf(z_score, df), decimal_limit)


def prepare_output(LCV, UCV, critical_value, population_mean, z_critical):
    msg = (
        '"Fail" to reject the null hypothesis'
        if LCV <= round(population_mean, decimal_limit) <= UCV
        else "Reject the null hypothesis"
    )
    return f"LCV: {LCV}, UCV: {UCV}, Addition Value: {critical_value}, Zc: {z_critical}, Result: {msg}"


def get_critical_value(
    population_mean: float,
    population_std: float,
    sample_size: int,
    alpha: float,
    test_type: TestType,
    df=None,
    critical_value_calculator: Union[
        Callable[[float, int], float], Callable[[float], float]
    ] = get_z_critical_normal,
    standard_error_calculator: Callable[[float, int], float] = get_standard_error,
    sample_mean: float = None,
) -> Union[float, Tuple[float, float]]:

    if test_type == TestType.DOUBLE_TAILED:
        alpha = alpha / 2

    se = standard_error_calculator(population_std, sample_size)
    if df is None:
        z_critical = critical_value_calculator(alpha)
    else:
        z_critical = critical_value_calculator(alpha, df)
    critical_value = round(z_critical * se, decimal_limit)
    LCV = population_mean - critical_value
    UCV = population_mean + critical_value
    if sample_mean is None:
        sample_mean = population_mean
    return (
        prepare_output(LCV, UCV, critical_value, sample_mean, z_critical),
        (LCV, UCV, critical_value, sample_mean, z_critical),
    )


def get_critical_value_with_zc(
    z_critical: float,
    population_mean: float,
    population_std: float,
    sample_size: int,
    sample_mean: float = None,
) -> float:
    se = get_standard_error(population_std, sample_size)
    critical_value = round(z_critical * se, decimal_limit)
    LCV = population_mean - critical_value
    UCV = population_mean + critical_value
    if sample_mean is None:
        sample_mean = population_mean
    return (
        prepare_output(LCV, UCV, critical_value, sample_mean, z_critical),
        (LCV, UCV, critical_value, sample_mean, z_critical),
    )


# %%
# se = get_standard_error(population_std, sample_size)
# sample_mean = 34.5
# z = (sample_mean - population_mean) / se
# z

# %% [markdown]
"""
Calculate the value of z-score for the sample mean point on the distribution
Calculate the p-value from the cumulative probability for the given z-score using the z-table

Make a decision on the basis of the p-value (multiply it by 2 for a two-tailed test) with respect to the given value of α (significance value).
"""

# %%
def get_z_score(
    population_mean: float, sample_mean: float, sample_std: float, sample_size: int
) -> float:
    return round(
        (sample_mean - population_mean) / get_standard_error(sample_std, sample_size),
        decimal_limit,
    )


def test_hypothesis_with_p_method(
    sample_mean: float,
    sample_std: float,
    sample_size: int,
    test_type: TestType,
    alpha: float,
    population_mean: float,
    population_std: float = None,
):
    z_score = get_z_score(
        population_mean,
        sample_mean,
        population_std if population_std is not None else sample_std,
        sample_size,
    )
    p_value = get_p_value_normal(abs(z_score))
    p_value = p_value * 2 if test_type == TestType.DOUBLE_TAILED else p_value

    decision = (
        f"Reject the null hypothesis as p_value({p_value}) is < significance level({alpha})"
        if p_value < alpha
        else '"Fail" to reject the null hypothesis as p_value({p_value}) is >= significance level({alpha})'
    )
    return z_score, p_value, decision


# %%

test_hypothesis_with_p_method(
    sample_mean=34.5,
    sample_std=4,
    sample_size=49,
    test_type=TestType.DOUBLE_TAILED,
    alpha=0.03,
    population_mean=36,
)


# %%
test_hypothesis_with_p_method(
    sample_mean=510,
    sample_std=110,
    sample_size=900,
    test_type=TestType.DOUBLE_TAILED,
    alpha=0.05,
    population_mean=500,
)

# %%
test_hypothesis_with_p_method(
    sample_mean=510,
    sample_std=110,
    sample_size=90000,
    test_type=TestType.DOUBLE_TAILED,
    alpha=0.05,
    population_mean=500,
)
# %%

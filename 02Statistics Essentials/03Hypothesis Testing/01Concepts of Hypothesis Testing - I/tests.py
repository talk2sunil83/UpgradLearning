# %%
from logging import critical
from typing import Callable, Tuple, Union
import numpy as np
from enum import Enum, auto
from scipy.stats import norm, t, chi2

# %%
decimal_limit = 2


class TestType(Enum):
    # Non-directional
    DOUBLE_TAILED = auto()
    # directional
    LOWER_TAILED = auto()
    UPPER_TAILED = auto()


# https://machinelearningmastery.com/critical-values-for-statistical-hypothesis-testing/
# https://dfrieds.com/math/z-tests.html#:~:text=In%20this%20instance%2C%20the%20z,and%20standard%20deviation%20of%201).
# https://reneshbedre.github.io/blog/anova.html
# https://statisticsbyjim.com/hypothesis-testing/one-tailed-two-tailed-hypothesis-tests/
# %%


def get_standard_error(σ: float, n: int) -> float:
    return round(σ / np.sqrt(n), decimal_limit)


# def get_standard_error(μ:float, σ:float, n:int)->float:


def get_z_critical_normal(alpha: float) -> float:
    # Calculate Zc
    probability = 1 - alpha
    return round(norm.ppf(probability), decimal_limit)


def get_z_critical_t(alpha: float, df: int) -> float:
    # Calculate Zc
    probability = 1 - alpha
    return round(t.ppf(probability, df), decimal_limit)


def get_z_critical_chi2(alpha: float, df: int) -> float:
    # Calculate Zc
    probability = 1 - alpha
    return round(chi2.ppf(probability, df), decimal_limit)


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
    return prepare_output(LCV, UCV, critical_value, sample_mean, z_critical)


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
    return prepare_output(LCV, UCV, critical_value, sample_mean, z_critical)


# %%
get_critical_value_with_zc(2.17, 36, 4, 49)
# %%
get_critical_value_with_zc(2.17, 36, 4, 49, sample_mean=34.6)
# %%
alpha = 0.03
test_type = TestType.DOUBLE_TAILED
population_mean = 36
population_std = 4
sample_size = 49
get_critical_value(population_mean, population_std, sample_size, alpha, test_type)

# %%
# se = get_standard_error(population_std, sample_size)
# sample_mean = 34.5
# z = (sample_mean - population_mean) / se
# z

# %%
alpha = 0.05
test_type = TestType.LOWER_TAILED
population_mean = 350
print(get_z_critical_normal(alpha))
population_std = 90
sample_size = 36
sample_mean = 34.6
x = 370.16
get_critical_value(population_mean, population_std, sample_size, alpha, test_type)

# %%
alpha = 0.03
test_type = TestType.LOWER_TAILED
population_mean = 2.5
print(get_z_critical_normal(alpha))
population_std = 0.6
sample_size = 100
sample_mean = 2.6
get_critical_value(
    population_mean,
    population_std,
    sample_size,
    alpha,
    test_type,
    sample_mean=sample_mean,
)

# %%
alpha = 0.03
test_type = TestType.LOWER_TAILED
population_mean = 2.5
population_std = 0.6
sample_size = 1000
sample_mean = 2.6
get_critical_value(
    population_mean,
    population_std,
    sample_size,
    alpha,
    test_type,
    sample_mean=sample_mean,
)

# %%
alpha = 0.02
test_type = TestType.DOUBLE_TAILED
population_mean = 60
population_std = 10.7
sample_size = 100
sample_mean = 62.6
get_critical_value(
    population_mean,
    population_std,
    sample_size,
    alpha,
    test_type,
    sample_mean=sample_mean,
)

# %%

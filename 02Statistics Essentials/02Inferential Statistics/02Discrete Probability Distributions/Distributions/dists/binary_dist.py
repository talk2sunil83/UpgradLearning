# %%
from dists.utils import Utils as u
import math
# %%
# %%


class BinaryDist:
    def __init__(self, n: int, r: int, p: float) -> None:
        if not (u.check_n_r(n, r) and (0. <= p <= 1.)):
            raise ValueError(
                f"Check values of n,r, and p. Hint: {u.n_r_error_message} and (0.<=p<=1.)")
        else:
            self.n = n
            self.r = r
            self.p = p
            self.__pdf__ = lambda n, r, p: (
                u.nCr(n, r) * math.pow(p, r) * math.pow((1-p), (n-r)))

    def pdf(self, in_percent=False) -> float:
        res = self.__pdf__(self.n, self.r, self.p)
        if in_percent:
            res = round(res*100, 2)
        return res

    def cdf(self, in_percent=False) -> float:

        res = sum([self.__pdf__(self.n, rt, self.p)
                   for rt in range(1, (self.r+1))])
        if in_percent:
            res = round(res*100, 2)
        return res

    def expected_value(self) -> float:

        res = sum([rt * self.__pdf__(self.n, rt, self.p)
                   for rt in range(1, (self.r+1))])
        return res

# %%

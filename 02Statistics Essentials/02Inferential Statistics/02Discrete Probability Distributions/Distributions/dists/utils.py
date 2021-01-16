import functools
import operator


class Utils:
    n_r_error_message = "n >= 0 and r >= 0 0 and n >= r"

    @staticmethod
    def check_n_r(n: int, r: int) -> bool:
        return n >= 0 and r >= 0 and n >= r

    @staticmethod
    def fact(n: int) -> int:
        if n == 0:
            return 1
        elif n == 1:
            return 1
        elif n >= 2:
            return functools.reduce(operator.mul, range(2, n+1))
        else:
            raise ValueError("n must be zero or positive integer")

    @staticmethod
    def nCr(n: int, r: int) -> int:
        if Utils.check_n_r(n, r):
            return int(Utils.fact(n)/(Utils.fact(r)*(Utils.fact(n-r))))
        else:
            raise ValueError(
                f"Check values of n and r and definition of nCr. Hint: {Utils.n_r_error_message}")

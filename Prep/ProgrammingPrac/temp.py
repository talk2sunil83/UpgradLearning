# %%

from typing import Sequence
from numba import njit
# %%


def fib(n: int):
    if(n <= 2):
        return 1
    return fib(n-1) + fib(n-2)


# print(fib(3))
# print(fib(5))
# print(fib(8))

def fib_memo(n: int, memo={1: 1, 2: 1}):
    if n >= 1:
        if (n in memo):
            return memo[n]
        memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
        return memo[n]

    else:
        print("Provide Positive int")

# %%

# print(fib_memo(100))


# @njit(fastmath=True, cache=True)
def grid_traveller(x: int, y: int, memo={}) -> int:
    if (x, y) in memo.keys():
        return memo[(x, y)]
    if x*y == 0:
        return 0
    if x*y == 1:
        return 1

    res = grid_traveller(x-1, y, memo) + grid_traveller(x, y-1, memo)
    memo[(x, y)] = res
    return res


print(grid_traveller(1, 1))
print(grid_traveller(3, 3))
print(grid_traveller(10, 10))

# %%


def can_sum(target: int, arr: Sequence[int]) -> bool:
    pass
# %%


def how_sum(target: int, arr: Sequence[int]) -> Sequence[int]:
    pass
# %%


def can_sum(target: int, arr: Sequence[int]):
    pass

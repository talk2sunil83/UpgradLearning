# %%
from typing import Sequence
# %%
# %%


def seq_shift(arr: Sequence[int], rotate_size: int) -> Sequence[int]:
    array_len = len(arr)
    for _ in range(rotate_size):
        current = arr[0]
        for i in range(array_len-1):
            arr[i] = arr[i+1]
        arr[array_len-1] = current
    return arr


def slicing(arr: Sequence[int], rotate_size: int) -> Sequence[int]:
    return arr[rotate_size:] + arr[:rotate_size]


def gcd(a: int, b: int) -> int:
    pass


def rotate(arr: Sequence[int], rotate_size: int) -> Sequence[int]:
    if len(arr) <= rotate_size:
        raise ValueError("Rotate size must be less than array size")

    # return slicing(arr, rotate_size)
    # return seq_shift(arr, rotate_size)
    pass


arr = [1, 2, 3, 4, 5, 6, 7]

rotate(arr, 2)

# %%

# %%

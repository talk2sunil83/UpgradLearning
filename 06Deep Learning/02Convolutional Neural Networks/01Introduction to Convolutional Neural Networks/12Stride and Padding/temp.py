# %%
from typing import List


def get_op_size(image_size: int, padding_size: int, filter_size: int, stride_size: int) -> int:
    """Get output image size after convolution

    Args:
        image_size (int): input image size
        padding_size (int): padding size
        filter_size (int): filter/kernel size
        stride_size (int): stride size

    Returns:
        int: output image size
    """
    return (((image_size+2*padding_size - filter_size)/stride_size) + 1)


def get_possible_stride_sizes(image_size: int, padding_size: int, filter_size: int) -> List[int]:
    """Get possible stride sizes as list of Integers

    Args:
        image_size (int): input image size
        padding_size (int): padding size
        filter_size (int): filter/kernel size

    Returns:
        int: all possible stride sizes
    """
    resultant_srides = []
    total_value = (image_size+2*padding_size - filter_size)
    print(f"{total_value=}")
    for i in range(1, total_value):
        if total_value % i == 0:
            resultant_srides.append(i)
    return resultant_srides


def get_possible_padding_sizes(n: int, k: int, s: int, max_padding_size: int = 10) -> List[int]:
    """Get possible padding sizes as list of Integers

    Args:
        n (int): input image size
        k (int): filter size
        s (int): stride size
        max_stride_size (int): stride size
    Returns:
        int: all possible padding sizes
    """
    resultant_srides = []
    for p in range(1, max_padding_size):
        total_value = (n+2*p-k)
        if total_value % s == 0:
            resultant_srides.append(p)
    if not resultant_srides:
        print("No possible padding")
    return resultant_srides


# %%
print(get_op_size(224, 3, 5, 1))
print(get_op_size(224, 3, 5, 2))
print(get_op_size(224, 3, 5, 3))
print(get_op_size(224, 3, 5, 4))

# %%
get_possible_stride_sizes(224, 3, 5)

# %%
get_possible_padding_sizes(224, 5, 2)

# %%
print(get_op_size(image_size=8, filter_size=3, stride_size=1, padding_size=0))
print(get_op_size(image_size=7, filter_size=3, stride_size=2, padding_size=0))
print(get_op_size(image_size=8, filter_size=3, stride_size=1, padding_size=1))

# %%
get_op_size(128, 1, 3, 1)

# %%
2*27*3-1

# %%


def get_total_calculations(n, p, k, c=1):
    multiplications_for_one_convolution = (k**2)*c
    additions_for_one_convolution = multiplications_for_one_convolution - 1

    total_movements_h = (n+2*p-k+1)
    total_movements = total_movements_h**2
    total_calculations = (multiplications_for_one_convolution+additions_for_one_convolution) * total_movements

    print(f"{total_movements_h=}")
    print(f"{total_movements=}")
    print(f"{multiplications_for_one_convolution=}")
    print(f"{additions_for_one_convolution=}")

    print(f"{total_calculations=}")
    # return total_movements


get_total_calculations(3, 1, 3, 3)

# %%
get_total_calculations(3, 0, 3)

# %%
get_op_size(224, 0, 2, 2)
# %%
get_op_size(4, 1, 3, 1)

# %%

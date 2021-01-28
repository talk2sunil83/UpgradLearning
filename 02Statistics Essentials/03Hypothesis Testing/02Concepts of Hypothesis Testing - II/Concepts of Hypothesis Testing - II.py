# %%
import re

from pandas.core.algorithms import value_counts


def checkmail(email):
    # complete the function
    # the function should return the strings "invalid" or "valid" based on the email ID entered
    # https://www.w3schools.com/python/python_regex.asp
    email_re = "^[0-9a-zA-Z]+[@]+[a-z]+[\._]?[a-z]{2,3}$"
    return "valid" if re.search(email_re, email) else "invalid"


email = "a#43@gmail.com"  # "prerna@upgrad.com"  # input()
print(checkmail(email))

# %%

input_list = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
op = []
for lst in input_list:
    op.extend(lst)

op
# %%
n = 9


def weirdsum(n, dig_count=4):
    return sum([int(x * str(n)) for x in range(1, (dig_count + 1))])


print(weirdsum(9))
# %%
from collections import Counter

value = "ddddaacccb"
sorted([x[0] for x in Counter(value).most_common(3)])


# %%
x = 3
y = 4
import numpy as np

nested_lst = [[(i + j) / 2 for j in range(y)] for i in range(x)]
print(np.matrix(nested_lst))


# %% [markdown]
"""
SELECT FORMAT(SUM(ABS(dist_other_from_origin - emp_dist_from_origin)) / COUNT(dist_other_from_origin), 2) AS average
FROM (
      SELECT CAST(address AS SIGNED) AS dist_other_from_origin
      FROM employee
      WHERE ssn != '123456789'
     ) other_employees
CROSS JOIN (
      SELECT CAST(address AS SIGNED) AS emp_dist_from_origin
      FROM employee
      WHERE ssn = '123456789'
     ) employee
"""

# %% [markdown]
"""
SELECT 
    student_id
FROM (
    SELECT 
        student_id, 
        (SUBSTRING(marks,1,2) + SUBSTRING(marks,4,2) + SUBSTRING(marks,7,2))/3 as avg_marks 
    FROM 
        upgrad.marks 
    ORDER BY 
        avg_marks 
    DESC limit 1) avg_pcm_student
"""

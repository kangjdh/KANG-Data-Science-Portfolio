# ### JPMorgan Quatitative Research Forage Task 4

# This script aims to generate buckets that best summarizes the given FICO data, with the focus on 
# finding the best boundaries, to determine the probability of default for each borrower
# This is rating map that maps the FICO score of borrowers where a lower rating symbolizes a better credit score

import pandas as pd
import numpy as np
import os

# import data
df = pd.read_csv('Task 3 and 4_Loan_Data.csv')

# define x and y
X = df['default'].tolist()
y = df['fico_score'].tolist()

n = len(X)

# set score range where max is 850 and min is 300
score_range = 850 - 300 + 1

defaults = [0] * score_range
totals = [0] * score_range
squared_errors = [0] * score_range

for i in range(n):
    score = int(y[i]) - 300
    defaults[score] += X[i]
    totals[score] += 1

cum_default = np.cumsum(defaults)
cum_total = np.cumsum(totals)

def mse(start, end):
    total = cum_total[end] - (cum_total[start - 1] if start > 0 else 0)
    if total == 0:
        return 0
    default_sum = cum_default[end] - (cum_default[start - 1] if start > 0 else 0)
    mean = default_sum / total
    error = 0
    for i in range(start, end + 1):
        for _ in range(totals[i]):
            error += (defaults[i]/totals[i] - mean) ** 2
    return error

r = 10
dp = [[float('inf')] * score_range for _ in range(r + 1)]
backtrack = [[-1] * score_range for _ in range(r + 1)]

for j in range(score_range):
    dp[1][j] = mse(0, j)

for i in range(2, r + 1):
    for j in range(score_range):
        for k in range(i - 2, j):
            cost = dp[i - 1][k] + mse(k + 1, j)
            if cost < dp[i][j]:
                dp[i][j] = cost
                backtrack[i][j] = k

boundaries = []
k = score_range - 1
b = r
while b > 1:
    k = backtrack[b][k]
    boundaries.append(k + 300)
    b -= 1

boundaries.append(300)
boundaries.append(850)
boundaries = sorted(boundaries)

print("FICO score boundaries:")
print(boundaries)
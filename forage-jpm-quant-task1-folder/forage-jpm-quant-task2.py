### JPMorgan Quatitative Research Forage Task 2

# this script aims to create a function that takes multipe factors into account and returns a price for the contract
# factors: injection dates, withdrawal dates, prices at which commodity can be purchased/sold for
# on certain dates, rate at which gas can be withdrawn/injected, the maximum volume that can be stored,
# and storage costs

import datetime
import math

# intialize injection and withdrawl date lists
injection_dates = []
withdrawal_dates = []

injection_prices = []
withdrawal_prices = []

# initialize variables
volume = 0 # amount of gas in storage
buy_cost = 0 # cost of buying and injecting gas
cash_in = 0 # revenue from selling gas minus the withdrawal cost

inj_dp = {}
with_dp = {}

# create while loops that allow users to input dates of choice for injection and withdrawal
# after the loop, allow users to input prices for each date inputed
while True:
    user_inj_dates = input(f"Enter injection date(s) or enter 'stop' to stop: ")
    if user_inj_dates == 'stop':
        break
    try:
        date_obj = datetime.datetime.strptime(user_inj_dates, "%Y-%m-%d").date()
        injection_dates.append(date_obj)
    except ValueError:
        print(f'Please input dates in YYYY-MM-DD format!')

for date in injection_dates: 
    user_inj_prices = float(input(f"Enter injection price for {date}: "))
    injection_prices.append(user_inj_prices)
    inj_dp[date] = user_inj_prices

# this is the same date and price input loop as above except for withdrawals now
while True:
    user_with_dates = input(f"Enter withdrawal date(s) or enter 'stop' to stop: ")
    if user_with_dates == 'stop':
        break
    try:
        date_obj = datetime.datetime.strptime(user_with_dates, "%Y-%m-%d").date()
        withdrawal_dates.append(date_obj)
    except ValueError:
        print(f'Please input dates in YYYY-MM-DD format!')

for date in withdrawal_dates:
    user_with_prices = float(input(f"Enter withdrawal price for {date}: "))
    withdrawal_prices.append(user_with_prices)
    with_dp[date] = user_with_prices

assert len(injection_dates) == len(injection_prices)
assert len(withdrawal_dates) == len(withdrawal_prices)

# set fixed parameters and prices
rate = 2500 # the rate at which the gas can be injected/withdrawn
max_volume = 100000 # maximum volume that can be stored
storage_costs = 50 # storage costs for each month utilized
withdrawal_cost_rate = .0001 # cost per cubic feet withdrawn

# sort all days in chronological order
all_dates = sorted(set(injection_dates + withdrawal_dates))

# create a loop that goes through each inputed date
for date in all_dates:
# check if date is in injection dates
# if so, inject at rate until storage capacity cannot take in any more
# calculate buying and injection cost
    if date in inj_dp:
        if volume + rate <= max_volume:
            volume += rate
            price = inj_dp[date]
            cost = rate * price
            injection_cost = rate * withdrawal_cost_rate
            buy_cost += cost + injection_cost
# if not enough storage to inject, write message
        else:
            print(f'Not enough storage capacity to inject')

# check if date is in withdrawal dates
# if so, withdraw at rate until not enough storage capacity to withdraw
# calculate cost of withdrawal and revenue gained from selling at date-appropriate price
    if date in with_dp:
        if volume >= rate:
            volume -= rate
            price = with_dp[date]
            revenue = rate * price
            withdrawal_cost = rate * withdrawal_cost_rate
            cash_in += revenue - withdrawal_cost
# if not enough storage capacity to withdraw, write message
        else:
            print(f'Not enough storage capacity to withdraw')

# calculate total time of storage and associated storage cost
total_storage_days = (max(withdrawal_dates) - min(injection_dates)).days
total_storage_months = math.ceil(total_storage_days / 30) # ciel will round up to full months
total_storage_cost = total_storage_months * storage_costs

# calculate the total profit earned from selling gas
profit = cash_in - buy_cost - total_storage_cost
print(f'The price of this contract is {profit}.')

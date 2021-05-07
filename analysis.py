import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from scipy.special import expit

# Many variables in this script (such as the race distributions) are four-dimensional vectors.
# For these variables:
#  - the first value corresponds to data for white subjects
#  - the second value corresponds to data for black subjects
#  - the third value corresponds to data for Hispanic subjects
#  - the fourth value corresponds to data for Asian subjects

# Import data
total_population = 0
population = {}
race_distribution = {}
vacc_distribution = {}
with open('filled_in_data.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        state = row[0]
        population[state] = int(row[1])
        total_population += int(row[1])
        race_distribution[state] = [float(val) for val in row[2:6]]
        vacc_distribution[state] = [float(val) for val in row[6:10]]

vacc_total = {}
with open('num_vaccines.csv', 'r') as file:
    reader = csv.reader(file)
    r = 0
    for row in reader:
        if r > 2 and r < 54:
            vacc_total[row[0]] = int(row[4])
        r += 1

urbanization = {}
with open('state_urbanization.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        if row[0] in population.keys():
            urbanization[row[0]] = float(row[1])

# Normalize data
for state in race_distribution.keys():
    dist = race_distribution[state]
    adjust = 1 / sum(dist)
    race_distribution[state] = [p * adjust for p in dist]

    dist = vacc_distribution[state]
    adjust = 1 / sum(dist)
    vacc_distribution[state] = [p * adjust for p in dist]

# Convert distributions to number of people/vaccines
num_people = {}
num_vaccines = {}
for state in population.keys():
    total = population[state]
    num_people[state] = [total * p for p in race_distribution[state]]
    total = vacc_total[state]
    num_vaccines[state] = [total * p for p in vacc_distribution[state]]

# Use # vaccines administered to calculate probability of vaccination
probabilities = {}
for state in population.keys():
    num_p = num_people[state]
    num_v = num_vaccines[state]
    state_prob = [v / p for v, p in zip(num_v, num_p)]
    probabilities[state] = state_prob

if False: # Optional outputs
    print("          P(Vacc | A=White)   P(Vacc | A=Black)   P(Vacc | A=Hispanic)   P(Vacc | A=Asian)")
    for s in population:
       print(f"{s}: {probabilities[s]}")
    print('')
    for s in population:
        for r in probabilities[s]:
            if r >= 1:
                print(f"{s}: {probabilities[s]}")

# Certain fully-censored states need to be accounted for through the data of similar states.
# To do this, we weigh uncensored states' data by the inverse of their chance of being uncensored,
# with the chance of being uncensored defined by a logistic regression model fit to the censorship data.
# Calculating the logistic regression parameters:

censored = ['Montana', 'New Hampshire', 'North Dakota', 'Wyoming']
# Partially censored - placed on a separate line for easy disabling
#censored.extend(['Hawaii', 'South Carolina', 'Vermont', 'West Virginia', 'Kentucky', 'Louisiana'])
U = []
Y = []
for state in population:
    U.append(urbanization[state])
    Y.append(0 if state in censored else 1)
X = [[u] for u in U]
lr = linear_model.LogisticRegression()
lr.fit(X, Y)
print(f"k = {lr.coef_}")
print(f"x0 = {lr.intercept_}")
print('')

# Plotting
plt.figure(1, figsize=(8, 4))
plt.style.use('seaborn')
plt.scatter(U, Y, color='red', s=5, zorder=20)
X_test = np.linspace(0, 15, 300)
loss = expit(X_test * lr.coef_ + lr.intercept_).ravel()
plt.plot(X_test, loss, color='blue', linewidth=3)
plt.title("Logistic Regression on State Censoring")
plt.xlabel('Urbanization Index')
plt.ylabel('Censored')
plt.show()

# Applying this to only the fully censored states, we get
# a logistic coefficient of 1.357 and
# a logistic bias of -10.908.

def log_curve_probability(x, k, x0):
    return 1 / (1 + math.exp(-((k * x) + x0)))

# Applying the logistic curve to the uncensored
total_pseudopop = 0
psuedopop = {}
for state in population:
    if state not in censored:
        factor = 1 / log_curve_probability(urbanization[state], lr.coef_, lr.intercept_)
        print(f"{state}: {factor}")
        psuedopop[state] = population[state] * factor
        total_pseudopop += population[state] * factor

# Stratification across the censorship-adjusted pseudopopulation
overall_prob = [0.0] * 4
for state in psuedopop:
    # Stratify the data from this state
    factor = psuedopop[state] / total_pseudopop
    for i in range(4):
        overall_prob[i] += factor * probabilities[state][i]

print('')
print("STRATIFICATION: COUNTRY-WIDE RESULTS")
print("(Y = 1 represents vaccination)")
print('')
print(f" P(Y^(A = white) = 1):      {overall_prob[0]}")
print(f" P(Y^(A = black) = 1):      {overall_prob[1]}")
print(f" P(Y^(A = hispanic) = 1):   {overall_prob[2]}")
print(f" P(Y^(A = asian) = 1):      {overall_prob[3]}")

# IP Weighting
vacc_subjects = [0.0] * 4
total_subjects = [0.0] * 4
for state in psuedopop:
    for r in range(4):
        weight = 1 / race_distribution[state][r]
        num_subjects = race_distribution[state][r] * psuedopop[state]
        vacc_subjects[r] += weight * num_subjects * probabilities[state][r]
        total_subjects[r] += weight * num_subjects

overall_ip = [v/t for v, t in zip(vacc_subjects, total_subjects)]

print('')
print("IP WEIGHTING: COUNTRY-WIDE RESULTS")
print("(Y = 1 represents vaccination)")
print('')
print(f" P(Y^(A = white) = 1):      {overall_ip[0]}")
print(f" P(Y^(A = black) = 1):      {overall_ip[1]}")
print(f" P(Y^(A = hispanic) = 1):   {overall_ip[2]}")
print(f" P(Y^(A = asian) = 1):      {overall_ip[3]}")

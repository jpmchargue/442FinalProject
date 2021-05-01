import csv
import math

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
# with the chance of being uncensored defined as (1 - exp(7 - u)), where u represents the state's urbanization index.
censored = ['Montana', 'New Hampshire', 'North Dakota', 'Wyoming']
total_pseudopop = 0
psuedopop = {}
for state in population:
    if state not in censored:
        factor = 1 / (1 - math.exp(7 - urbanization[state]))
        #print(f"{state}: {factor}")
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

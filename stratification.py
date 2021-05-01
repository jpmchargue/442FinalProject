import csv

# Import data
population = {}
race_distribution = {}
vacc_distribution = {}
with open('filled_in_data.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        state = row[0]
        population[state] = int(row[1])
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

#print(len(population))
#print(len(race_distribution))
#print(len(vacc_distribution))
#print(len(vacc_total))

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

print("          P(Vacc | A=White)   P(Vacc | A=Black)   P(Vacc | A=Hispanic)   P(Vacc | A=Asian)")

for s in population:
   print(f"{s}: {probabilities[s]}")

print('')

for s in population:
    for r in probabilities[s]:
        if r >= 1:
            print(f"{s}: {probabilities[s]}")

#for s in population:
#    print(f"{s}: {vacc_total[s]/population[s]}")

#for s in num_people:
#    print(f"{s}: {num_people[s]}")

#print('')

#for s in num_vaccines:
#    print(f"{s}: {num_vaccines[s]}")

#censored = ['Montana', 'New Hampshire', 'North Dakota', 'Wyoming']



# Standardize data across all states

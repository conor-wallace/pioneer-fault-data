import GA
import numpy as np
from ann import NeuralNetwork

sol_per_pop = 8
num_parents_mating = 4
crossover_location = 5

# Defining the population size.
pop_size = (sol_per_pop) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
print(pop_size)
new_population = []

for i in range(sol_per_pop):
    new_network = NeuralNetwork()
    weights = []
    #Input Layer
    input_weights=np.random.rand(4, 6) #weight
    input_biases=np.random.rand(6) #biases
    weights.append(input_weights)
    weights.append(input_biases)
    #Hidden Layers
    hidden_weights=np.random.rand(6, 6) #weight
    hidden_biases=np.random.rand(6) #biases
    weights.append(hidden_weights)
    weights.append(hidden_biases)
    #Output Layer
    output_weights=np.random.rand(6, 3) #weight
    output_biases=np.random.rand(3) #biases
    weights.append(output_weights)
    weights.append(output_biases)

    new_network.create_model(np.asarray(weights))

    new_population.append(new_network)

best_outputs = []
num_generations = 25
for generation in range(num_generations):
    print("Generation : ", generation)
    #print(new_population)
    # Measuring the fitness of each chromosome in the population.
    fitness = GA.cal_pop_fitness(new_population)
    print("Fitness")
    print(fitness)

    #best_outputs.append(np.max(np.sum(new_population, axis=1)))
    # The best result in the current iteration.
    #print("Best result : ", np.max(np.sum(new_population, axis=1)))

    # Selecting the best parents in the population for mating.
    parents = GA.select_mating_pool(np.asarray(new_population), fitness, num_parents_mating)
    print("Parents")
    print(parents)
    #print(len(parents))

    # Generating next generation using crossover.
    offspring_crossover = GA.crossover(np.asarray(parents))
    print("Crossover")
    print(offspring_crossover)

    # Adding some variations to the offspring using mutation.
    offspring_mutation = GA.mutation(offspring_crossover)
    print("Mutation")
    print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    new_population = np.reshape(new_population, [8, 3, 2])
    #print(new_population.shape)
    new_population[:4, :, :] = parents
    new_population[4:, :, :] = offspring_mutation
    print(new_population)


'''
best_outputs = []
num_generations = 25
for generation in range(num_generations):
    print("Generation : ", generation)
    #print(new_population)
    # Measuring the fitness of each chromosome in the population.
    fitness = GA.cal_pop_fitness(np.asarray(new_population), generation)
    print("Fitness")
    print(fitness)

    #best_outputs.append(np.max(np.sum(new_population, axis=1)))
    # The best result in the current iteration.
    #print("Best result : ", np.max(np.sum(new_population, axis=1)))

    # Selecting the best parents in the population for mating.
    parents = GA.select_mating_pool(np.asarray(new_population), fitness, num_parents_mating)
    print("Parents")
    print(parents)
    #print(len(parents))

    # Generating next generation using crossover.
    offspring_crossover = GA.crossover(np.asarray(parents))
    print("Crossover")
    print(offspring_crossover)

    # Adding some variations to the offspring using mutation.
    offspring_mutation = GA.mutation(offspring_crossover)
    print("Mutation")
    print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    new_population = np.reshape(new_population, [8, 3, 2])
    #print(new_population.shape)
    new_population[:4, :, :] = parents
    new_population[4:, :, :] = offspring_mutation
    print(new_population)

# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = GA.cal_pop_fitness(new_population, 10)
# Then return the index of that solution corresponding to the best fitness.
best_match = 0
for i in range(len(fitness)):
    if fitness[i] > fitness[best_match]:
        best_match = i
new_population = np.reshape(new_population, [8, 3, 2])

print("Best solution : ", new_population[best_match, :, :])
print("Best solution fitness : ", fitness[best_match])
controller1 = ControlLaw("agent1")
trust1 = controller1.fuzzyControl(0, new_population[best_match, 0, :], new_population[best_match, 1, :], new_population[best_match, 2, :], True)
'''

'''

new_network = NeuralNetwork()

training_file = '../config/training_data.csv'
data_set = new_network.read_data_sets(training_file)
print("Loaded training data...")

features, labels_norm = new_network.generate_train_test_data_sets(data_set)
print(features.shape)
labels = []
# convert label data to one_hot arrays
for k in range(0, len(labels_norm)):
    one_hot_label = np.zeros([3], dtype=float)
    one_hot_label[int(float(labels_norm[k]))] = 1.0
    labels = np.append(labels, one_hot_label)
labels = np.reshape(labels, [-1, new_network.n_labels])

print(labels.shape)

weights = []
#Input Layer
input_weights=np.random.rand(4, 6) #weight
input_biases=np.random.rand(6) #biases
weights.append(input_weights)
weights.append(input_biases)
#Hidden Layers
hidden_weights=np.random.rand(6, 6) #weight
hidden_biases=np.random.rand(6) #biases
weights.append(hidden_weights)
weights.append(hidden_biases)
#Output Layer
output_weights=np.random.rand(6, 3) #weight
output_biases=np.random.rand(3) #biases
weights.append(output_weights)
weights.append(output_biases)
print("weights shape")
print(np.asarray(weights).shape)

new_network.create_model(np.asarray(weights))
new_network.predict(features)
'''

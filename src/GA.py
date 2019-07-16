import numpy

def cal_pop_fitness(population):
    # Calculating the fitness value of each solution in the current population.
    fitness = []
    for i in range(8):
        training_file = '../config/training_data.csv'
        data_set = population[i].read_data_sets(training_file)
        print("Loaded training data...")

        features, labels_norm = population[i].generate_train_test_data_sets(data_set)
        print(features.shape)
        labels = []
        # convert label data to one_hot arrays
        for k in range(0, len(labels_norm)):
            one_hot_label = numpy.zeros([3], dtype=float)
            one_hot_label[int(float(labels_norm[k]))] = 1.0
            labels = numpy.append(labels, one_hot_label)
        labels = numpy.reshape(labels, [-1, population[i].n_labels])

        print("Labels length")
        print(len(labels))
        print("Features length")
        print(len(features))

        score = 0

        for j in range(0, len(features)):
            prediction = population[i].predict(features[j])
            print("prediction")
            print(prediction[0, 0])
            print(prediction[0, 1])
            print(prediction[0, 2])
            rounded_prediction = []
            rounded_prediction.append(int(round(prediction[0, 0])))
            rounded_prediction.append(int(round(prediction[0, 1])))
            rounded_prediction.append(int(round(prediction[0, 2])))
            rounded_prediction = numpy.reshape(rounded_prediction, [1, 3])
            print("prediction rounded")
            print(rounded_prediction[0, 0])
            print(rounded_prediction[0, 1])
            print(rounded_prediction[0, 2])
            print("label")
            print(labels[j, 0])
            print(labels[j, 1])
            print(labels[j, 2])

            correct = True
            for n in range(len(labels[j])):
                if labels[j, n] != rounded_prediction[0, n]:
                    correct = False

            if correct == True:
                score += 1

        print("score: %s%%" % str(100 * (float(score) / float(len(labels)))))
        fitness.append(100 * (float(score) / float(len(labels))))

    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = numpy.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover

# Implement the algorithm here
import matplotlib.pyplot as plt
import statistics
import numpy as np
import csv
import time
import random
import pandas as pd
from itertools import permutations
import math


def make_dest_arr(reader):
    dist = np.zeros((24, 24))
    cities = []
    i = 0

    for row in reader:
        if (i == 0):
            for j in range(24):  # forste linje i filen er byer

                cities.append(row[j])
        else:
            for j in range(len(dist)):  # each row is a list
                dist[i - 1, j] = float(row[j])
        i += 1
    return dist, cities


def get_length(dest, cities):
    how_long = 0
    for i in range(len(cities)):
        if (i + 1 < len(cities)):
            by = cities[i]
            neste_by = cities[i + 1]

            how_long += dest[by][neste_by]
        else:
            by = cities[i]
            neste_by = cities[0]
            how_long += dest[by][neste_by]
    return how_long


def pmx(p1, p2, start, stop):  # crossover, pmx fra ukesoppgaver
    child = np.zeros((p2.size), int)
    child.fill(-1)
    # 1. Choose random segment and copy it from P1
    child[start:stop] = p1[start:stop]
    # 2. Starting from the first crossover point look for elements in that segment of P2 that have not been copied
    for i in range(start, stop):
        if not (p2[i] in p1[start:stop]):  # hvis tallet ikke er kopiert til barn
            hvor = (np.where(p2 == p1[i]))
            while (start <= hvor[0][0] < stop):
                hvor = (np.where(p2 == p1[hvor[0]]))
            child[hvor[0]] = p2[i]
            # kopiere over resten av arrayet
    for i in range(p1.size):
        if child[i] == -1:
            child[i] = p2[i]
    return child


def swap_permut(genotype):  # mutation
    tilf = np.random.choice(genotype, 2, replace=False)
    arr = (np.zeros(2, int))
    for i in range(len(genotype)):
        if tilf[0] == genotype[i]:
            arr[0] = i
        elif tilf[1] == genotype[i]:
            arr[1] = i
    genotype[arr[0]], genotype[arr[1]] = genotype[arr[1]], genotype[arr[0]]
    return genotype


def make_population(population_size, nbr_cities):
    population = []
    for i in range((population_size)):
        random_tour = random.sample(range(0, nbr_cities), nbr_cities)
        population.append(random_tour)
    # print(population)
    return population


def tournament_selection(population, dist):  # parent selection
    all_times = np.zeros(len(population))
    best_tours = []
    for i in range(len(population)):
        all_times[i] = get_length(dist, population[i])
    temp = np.argpartition(all_times, (len(population) // 2))
    result_args = temp[:(len(population) // 2)]

    for i in result_args:
        best_tours.append(population[i])

    return best_tours  # beste turer, basert på lengde


def child_selection(population, dist, size):
    all_times = np.zeros(len(population))
    best_tours = []
    for i in range(len(population)):
        all_times[i] = get_length(dist, population[i])
    temp = np.argpartition(all_times, (len(population) // 2))
    result_args = temp[:size]

    for i in result_args:
        best_tours.append(population[i])

    return best_tours  # size beste turer, basert på tid


def mutate(population, pr):
    for tour in population:
        if random.random() < pr:

            tour = swap_permut(tour)
    return population


def get_best_ind(dest, neighbours):
    best_length = get_length(dest, neighbours[0])
    best_n = neighbours[0]
    for n in neighbours:
        temp_length = get_length(dest, n)
        if temp_length < best_length:
            best_length = temp_length
            best_n = n
    return best_length


def genetic(dist, population_size, nbr_cities, nbr_generations):
    start = 2
    stop = 6
    pr = 0.1  # Probability of mutation
    best_fit = np.zeros(nbr_generations)  # best tid tilpasset hver generasjon
    population = make_population(population_size, nbr_cities)

    children = []
    g = 0
    while g < nbr_generations:

        best_tours = tournament_selection(population, dist)

        for i in range(0, len(best_tours) - 1, 2):
            p1 = np.array(best_tours[i])
            p2 = np.array(best_tours[i + 1])
            c1 = pmx(p1, p2, start, stop)  # new children
            c2 = pmx(p2, p1, start, stop)
            children.append(list(c1))
            children.append(list(c2))

        population = child_selection(children, dist, population_size)
        population = mutate(population, pr)  # mutate

        best_fit[g] = get_best_ind(dist, population)
        g += 1

    #   population = child_selection(children, dist, population_size//2) #De 50% beste individene av siste generasjon
    population = child_selection(children, dist, 10)  # De 20 beste individene av siste generasjon

    best_length = np.zeros(len(population))
    for i in range(len(population)):
        best_length[i] = get_length(dist, population[i])

    return population, best_length, best_fit


def main_gen():
    file = open('european_cities.csv', 'r+')
    re = csv.reader(file, delimiter=";")
    dist, cities = make_dest_arr(re)

    population_size = [50, 100, 150]
    nbr_cities = [10, 24]
    nbr_generations = 80
    nbr_of_runs = 20
    time_of_last = np.zeros(2)

    pop_avg = np.zeros((2, 3, nbr_generations))
    best_average_fit = np.zeros((nbr_of_runs, nbr_generations))

    for k in range(len(nbr_cities)):
        for j in range(3):
            all_times = []
            all_pop = []
            for i in range(nbr_of_runs):
                start_time = time.time()

                population, best_length, best_fit = genetic(dist, population_size[j], nbr_cities[k], nbr_generations)
                all_times.append(best_length)
                all_pop.append(population)
                end_time = time.time() - start_time
                time_of_last[k] = end_time

                best_average_fit[i] = best_fit

            mean = np.mean(all_times)
            std = np.std(all_times)

            print()
            # report best, worst, mean and standard deviation of tour length out of 20 runs of the algorithm 
            print("population_size: ", population_size[j])
            print("nbr_cities: ", nbr_cities[k])
            print("nbr_generations: ", nbr_generations)
            print("Mean: ", mean)
            print("Std: ", std)
            print("Time of last run", time_of_last[k])

            best_time = np.amin(all_times)  # find best
            worst_time = np.amax(all_times)
            hvor_b = np.where(all_times == best_time)
            hvor_w = np.where(all_times == worst_time)

            best_tour = all_pop[hvor_b[0][0]]
            worst_tour = all_pop[hvor_w[0][0]]
            print("Best length: ", best_time)
            print("Worst length:", worst_time, "\n")

            aveage_all_gen = np.zeros(nbr_generations)
            for i in range(nbr_generations):
                average = np.mean(best_average_fit[:, i])
                #    aveage_all_gen[i] = average

                pop_avg[k][j][i] = average



            # Also, find and plot the average fitness of the best fit individual in each generation
            # (average across runs), and include a figure with all three curves in the same plot in the report.
            # Conclude which is best in terms of tour length and number of generations of evolution time.

    # plot the results
    plt.plot(pop_avg[0][0], label='Size of population is 50')
    plt.plot(pop_avg[0][1], label='Size of population is 100')
    plt.plot(pop_avg[0][2], label='Size of population is 150')
    plt.legend(loc='best')
    plt.xlabel('Generations')
    plt.ylabel('Route length')
    plt.savefig('plot24.png')
    plt.show()


main_gen()

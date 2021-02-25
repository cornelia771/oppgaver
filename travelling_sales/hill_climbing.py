# Implement the algorithm here
# from statistics import mean
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


def to_cities(cities, beste_reise):
    c = []
    for i in beste_reise:
        c.append(cities[i])
    return c


def hill_climber(dest, cities):
    shortest = get_length(dest, cities)
    all_neighbours = get_all_neighbours(cities)
    best_n, best_length = get_best(dest, all_neighbours)

    while best_length < shortest:
        shortest = best_length
        cities = best_n
        all_neighbours = get_all_neighbours(cities)
        best_n, best_length = get_best(dest, all_neighbours)
    return cities, shortest


def get_all_neighbours(cities):
    neighbours = []
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            temp_cities = cities.copy()
            temp_cities[i] = cities[j]
            temp_cities[j] = cities[i]
            neighbours.append(temp_cities)
    return neighbours


def get_best(dest, neighbours):
    best_length = get_length(dest, neighbours[0])
    best_n = neighbours[0]
    for n in neighbours:
        temp_length = get_length(dest, n)
        if temp_length < best_length:
            best_length = temp_length
            best_n = n
    return best_n, best_length


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


def get_random_path(nbr):
    randomlist = random.sample(range(0, nbr), nbr)
    return randomlist


def main_hill():
    file = open('european_cities.csv', 'r+')
    re = csv.reader(file, delimiter=";")
    nbr_of_runs = 20
    nbr_cities = [10, 24]

    # Report the length of the tour of the best, worst and mean of 20 runs (with random starting tours),
    # as well as the standard deviation of the runs, both with the 10 first cities
    all_times = np.zeros(nbr_of_runs)
    all_tours = []
    dest, c = make_dest_arr(re)
    for j in range(len(nbr_cities)):
        for i in range(nbr_of_runs):
            start_time = time.time()

            cities = get_random_path(nbr_cities[j])
            cities, shortest = hill_climber(dest, cities)
            all_times[i] = shortest
            all_tours.append(cities)

            end_time = time.time() - start_time

        mean = np.mean(all_times)
        std = np.std(all_times)

        best_time = np.amin(all_times)  # find best
        worst_time = np.amax(all_times)
        hvor_b = np.where(all_times == best_time)
        hvor_w = np.where(all_times == worst_time)

        best_tour = all_tours[hvor_b[0][0]]
        worst_tour = all_tours[hvor_w[0][0]]

        city_name_best = to_cities(c, best_tour)
        city_name_worst = to_cities(c, worst_tour)

        print("Nbr of runs: ", nbr_of_runs)
        print("Nbr of cities: ", nbr_cities[j])
        print("Mean: ", mean)
        print("Std: ", std, "\n")
        print("Best tour: ", city_name_best)
        print("Best length:", best_time, "\n")
        print("Worst tour: ", city_name_worst)
        print("Worst length:", worst_time)
        print("Time of last run", end_time)
        print("-------------------")


main_hill()

# Implement the algorithm here
import numpy as np
import csv
import time
import random
import pandas as pd
from itertools import permutations
import math


# onsker a lage en array med alle avstander
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


# onsker a forst finne alle kombinasjoner av byer man kan besoke
def find_best(dist, city_nbr):
    all_comb = list(permutations(city_nbr))
    shortes = 10000000
    dist_best = 1000000000
    for i in range(math.factorial(len(city_nbr))):
        how_long = 0
        for j in range(len(city_nbr)):
            if (j + 1 < len(city_nbr)):
                by = all_comb[i][j]
                neste_by = all_comb[i][j + 1]
                how_long += dist[by][neste_by]

            else:  # ma reise tilbake til startby
                by = all_comb[i][j]
                neste_by = all_comb[i][0]
                how_long += dist[by][neste_by]

        if how_long < dist_best:
            dist_best = how_long
            beste_reise = all_comb[i]
    return beste_reise, dist_best


def to_cities(cities, beste_reise):
    c = []
    for i in beste_reise:
        c.append(cities[i])
    return c


def how_long(nbr):
    file = open('european_cities.csv', 'r+')
    re = csv.reader(file, delimiter=";")

    start_time = time.time()

    city_nbr = range(nbr)
    dest_array, cities = make_dest_arr(re)
    beste_reise, dist_best = find_best(dest_array, city_nbr)
    b = to_cities(cities, beste_reise)

    end_time = time.time() - start_time
    print("Nbr of cities: ", nbr)
    print("Shorest route: ", b)
    print("Distance: ", dist_best)
    print("Time: ", end_time, "seconds", "\n")

    return end_time


def main_exhaustive():
    for i in range(6, 11):
        how_long(i)


main_exhaustive()

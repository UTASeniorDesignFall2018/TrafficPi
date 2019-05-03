import random
import numpy as np 
import time

def strTimeProp(start, end, format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formated in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, format))
    etime = time.mktime(time.strptime(end, format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(format, time.localtime(ptime))


def randomDate(start, end, prop):
    return strTimeProp(start, end, '%m/%d/%Y %H:%M', prop)

speed_limit = 30

avg_nonspeeder = 28
std_nonspeeder = 3

speeder_chance = 0.05

avg_speeder = 40
std_speeder = 4

avg_cars = np.array([16, 12, 9, 8, 6, 8, 14, 20, 25, 35, 37, 35, 33, 32, 31, 31, 30, 29, 27, 27, 24, 20, 18, 16])

num_total_cars = 1273

avg_cars = avg_cars / sum(avg_cars)
avg_cars = np.round(avg_cars * num_total_cars).astype(int)

data = []

for t in range(len(avg_cars)):

    for i in range(avg_cars[t]):
        if random.random() <= speeder_chance:
            car_speed = np.random.normal(avg_speeder, std_speeder)
        else:
            car_speed = np.random.normal(avg_nonspeeder, std_nonspeeder)

        if t < 9:
            data.append([randomDate("1/1/2001 0{}:00".format(t), "1/1/2001 0{}:00".format(t+1), random.random()), car_speed])
        elif t == 9:
            data.append([randomDate("1/1/2001 0{}:00".format(t), "1/1/2001 {}:00".format(t+1), random.random()), car_speed])
        elif t == 23:
            data.append([randomDate("1/1/2001 23:00".format(t), "1/1/2001 23:59".format(t+1), random.random()), car_speed])
        else:
            data.append([randomDate("1/1/2001 {}:00".format(t), "1/1/2001 {}:00".format(t+1), random.random()), car_speed])

data.sort(key=lambda x: x[0])

with open("data.log", 'w') as f:
    for line in data:
        f.write("{} {}\n".format(str(line[0]), str(line[1])))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime
from matplotlib import dates

def main():
    times = ["12:00 AM", "2:00 AM", "4:00 AM", "6:00 AM", "8:00 AM", "10:00 AM", "12:00 PM", "2:00 PM", "4:00 PM", "6:00 PM", "8:00 PM", "10:00 PM"]
    with open("data.log", 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        spl = line.split()
        data.append([datetime.datetime.strptime("{} {}".format(spl[0], spl[1]), '%m/%d/%Y %H:%M'), float(spl[2])])

    data = np.array(data)
    
    amount = []
    for i in range(24):
        if i == 23:
            amount.append([datetime.datetime(2001, 1, 1, i), np.sum((data[:,0] >= datetime.datetime(2001, 1, 1, i)))])
        else:
            amount.append([datetime.datetime(2001, 1, 1, i), np.sum((data[:,0] >= datetime.datetime(2001, 1, 1, i)) & (data[:,0] < datetime.datetime(2001, 1, 1, i+1)))])

    amount = np.array(amount)
    
    # plot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.scatter(data[:, 0], data[:, 1], s=2, c='red', label='Speed of Vehicles')
    ax2.plot(amount[:,0], amount[:,1], label='Number of Vehicles', linewidth=5)

    plt.xlim((data[0,0], data[0,0] + datetime.timedelta(days=1)))
    ax1.set_ylim((0, 60))
    ax2.set_ylim((0, 100))

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Speed")
    ax2.set_ylabel("Number")
    
    ax1.xaxis.set_major_locator(dates.HourLocator(interval=2))   #to get a tick every 15 minutes
    ax1.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))     #optional formatting 

    ax1.grid(linestyle='--')
    plt.title("Traffic Pi Visualizer", fontsize=24)
    fig.legend(prop={'size': 16})
    fig.autofmt_xdate()
    plt.show()

main()
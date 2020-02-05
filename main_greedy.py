import random
import os
import time
import pandas as pd
import csv

from tqdm import tqdm
from bandits import Bandits
from agents import Greedy
#------------------------------------------

parameters = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]


def train(epsilon,arms=10, iterations=1000, tests=2000):
    # create bandit and agent.
    bandit = Bandits(arms)
    agent = Greedy(epsilon,[i for i in range(arms)])
    # collect data
    data = []

    for i in tqdm(range(tests)):
        # clearing the data from previous run.
        bandit.reset()
        agent.reset()

        for j in range(iterations):
            chosen_arm = agent.action()
            reward = bandit.pull(chosen_arm)
            agent.update({chosen_arm:reward})

        data.extend(bandit.history())

    return data

#--------------------------------------------

def save_data(data,path='./data/'):
    # process the data. Computing mean across different trials.
    data1 = [0 for _ in range(len(data[0]))]
    data2 = [0 for _ in range(len(data[0]))]

    for i in range(len(data)):
        if(i%2==0):
            # storing action
            for j in range(len(data[i])):
                data1[j] += 2*data[i][j]/len(data)

        else:
            # storing reward
            for j in range(len(data[i])):
                data2[j] += 2*data[i][j]/len(data)

    # save data for later.
    os.makedirs("./data",exist_ok=True)
    print("saving data..")
    num = random.choice([i for i in range(1000)])
    with open(path+str(num)+"_greedy.csv","w+") as f:
        writer = csv.writer(f)
        writer.writerows(zip(data1,data2))
    print(str(num)+" numbered file saved.")

#---------------------------------------------


if __name__ == '__main__':

    # train and collect statistics.
    print("Start training..")
    for x in parameters :
        # trying various parameters.
        print("epsilon value:"+str(x))
        start_time = time.time()
        reward_data = train(x)
        time_taken = time.time() - start_time
        print("Done. Time taken:"+str(time_taken)+" seconds")
        save_data(reward_data)

    print("All done!!")

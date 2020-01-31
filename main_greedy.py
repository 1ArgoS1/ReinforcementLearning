import random
import os
import time
import csv
import matplotlib.pyplot as plt

from tqdm import tqdm
from bandits import Bandits
from agents import Greedy
#------------------------------------------


def train(arms=10, iterations=1000, tests=200):
    # create bandit and agent.
    bandit = Bandits(arms)
    agent = Greedy(0,[i for i in range(arms)])
    data = []

    for i in tqdm(range(tests)):
        # clearing the data from previous run.
        bandit.reset()
        agent.reset()

        for j in range(iterations):
            chosen_arm = agent.action()
            #chosen_arm = random.randrange(arms)
            reward = bandit.pull(chosen_arm)
            agent.update({chosen_arm:reward})

        # Getting the optimal action.
        optimal_action = bandit.distribution.index(max(bandit.distribution))
        data.append([bandit.history(),optimal_action])

    return data
#--------------------------------------------

def process(data):
    # process the raw data sent by the agent and environment.
    pass
#--------------------------------------------

def save_csv(data,path='./data/greedy.csv'):
    # saves data to a csv file for plotting later.
    os.makedirs("./Data",exist_ok=True)
    print("saving data..")
    with open(path, 'w+') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerows(data)
#---------------------------------------------


if __name__ == '__main__':

    # train and collect statistics.
    print("Start training..")
    start_time = time.time()
    reward_data = train()
    time_taken = time.time() - start_time
    print("Done. Time taken:"+str(time_taken)+" seconds")
    save_csv((reward_data))
    print("All done!!")

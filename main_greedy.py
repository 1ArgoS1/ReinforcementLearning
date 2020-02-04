import random
import os
import time
import pandas as pd
import numpy

from tqdm import tqdm
from bandits import Bandits
from agents import Greedy
#------------------------------------------


def train(arms=10, iterations=1000, tests=2000):
    # create bandit and agent.
    bandit = Bandits(arms)
    agent = Greedy(0,[i for i in range(arms)])
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

def save_data(data,path='./data/greedy.csv'):
    # process the data for plotting.
    df = pd.DataFrame(data)
    # ram
    print(df)
    print(df.shape)







    # save image for later.
    os.makedirs("./data",exist_ok=True)
    print("saving data..")


#---------------------------------------------


if __name__ == '__main__':

    # train and collect statistics.
    print("Start training..")
    start_time = time.time()
    reward_data = train()
    time_taken = time.time() - start_time
    print("Done. Time taken:"+str(time_taken)+" seconds")
    save_data(reward_data)
    print("All done!!")

import random
import scipy.stats as stats

class Bandits():

    ''' Creates the multiarm bandits environment. Stores history.
        Methods:
            pull(i) - pulls the i lever and returns the reward r(i).
            history() - returns the history of the actions taken in a list.
            reset()- returns the bandit to initial state.

    '''

    def __init__(self, arms):
        # number of arms
        self.arms = arms
        # initialise arm's optimal reward(Q value).
        self.distribution = [random.gauss(0, 1) for _ in range(self.arms)]
        #self.distribution = stats.norm.rvs(size=self.arms,loc=0,scale=1)
        # collect history of the actions.
        self.data = []

    def pull(self, x):
        reward = random.gauss(self.distribution[x], 1)
        self.data.append([x, reward])
        return reward

    def history(self):
        # store reward per step
        reward_per_step = 0
        data = []
        for i,reward in enumerate(self.data):
            reward_per_step += reward[1]/(i+1)
            data.append(reward_per_step)
        return data

    def reset(self):
        self.data = []
        self.distribution = [random.gauss(0,1) for _ in range(self.arms)]





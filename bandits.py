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
        # initialise arm's optimal reward.
        self.distribution = [random.gauss(0, 1) for _ in range(self.arms)]
        #self.distribution = stats.norm.rvs(size=self.arms,loc=0,scale=1)

        # Optimal arm
        self.optim_arm = self.distribution.index(max(self.distribution))
        # collect history of the actions.
        self.data = [[], []]

    def pull(self, x):
        reward = random.gauss(self.distribution[x], 1)
        self.data[0].append(x)
        self.data[1].append(reward)
        return reward

    def history(self):
        # processing data for plotting.
        reward_count = 0
        optimal_action_count = 0
        data = [[], []]

        for i, action in enumerate(self.data[0]):
            # count number of times optimal action taken.
            if action == self.optim_arm:
                optimal_action_count += 1
            data[0].append(action_count)

        for i, reward in enumerate(self.data[1]):
            # calculate reward per step.
            reward_count += reward
            data[1].append(reward_count)

        return data



    def reset(self):
        self.data = [[], []]
        self.distribution = [random.gauss(0, 1) for _ in range(self.arms)]



import random
import math
import scipy.stats as stats


class Agent():
    ''' Implements different RL algorithms for learning about environment.
        Takes an action based on the algorithm and previous rewards.
        Available algorithms:
            (1) Epsilon-Greedy
            (2) Softmax
            (3) Upper Confidence Bound(UCB1)
            (4) Median Elimination Algorithm(MEA)
        Attributes:
            (1) actions - list of all possible action
            (2) count[i] - number of times an action i is taken.
            (3) estimate[i] - holds the expected reward value.
        Methods:
            (1) action - choose the action based on policy.
            (2) reset - flushes the buffer for each action.
            (3) update - places the reward from the environment into the action buffer
    '''

    def __init__(self, actions):
        # action set.
        self.actions = actions
        # holds the current estimate reward for the action.
        # elements of action should be unique else key clash will happen.
        self.estimate = {self.actions[i]: 0 for i in range(len(actions))}
        # count of actions
        self.count = {self.actions[i]: 0 for i in range(len(actions))}

    def action(self):
        raise NotImplementedError("action not initialised !")

    def update(self, x={}):
        # update all the estimates with reward obtained.
        for action in x.keys():
            # new_estimate = old_estimate + step_size*(reward - old_estimate)
            self.count[action] += 1
            self.estimate[action] += (x[action]-self.estimate[action])/(self.count[action])


    def reset(self, new_actions=None):
        self.estimate = {self.actions[i]: 0 for i in range(len(self.actions))}
        if(new_actions is not None):
            # changing action set at runtime
            self.actions = new_actions
            self.count = {self.actions[i]: 0 for i in range(len(self.actions))}
            self.estimate = {self.actions[i]: 0 for i in range(len(self.actions))}


class Greedy(Agent):
    ''' Implementation of epsilon greedy algorithm. Takes epsilon as a
        parameter and generates the optimal action 1-epsilon times and
        random action epsilon times. 0 <= epsilon < 1.
    '''

    def __init__(self, epsilon=0.2, *args):
        super(Greedy, self).__init__(*args)
        self.epsilon = epsilon
        # returns 1 (epsilon) times and 0 (1-epsilon) times
        self.distribution = stats.bernoulli(self.epsilon)

    def action(self):
        # sample from the distribution.
        explore = self.distribution.rvs()
        if(explore):
            # explorative action
            choice = random.choice(self.actions)
            return choice

        else:
            # exploitative action
            choice = max(self.estimate, key=self.estimate.get)
            return choice


class Softmax(Agent):
    ''' The action is given by sampling action from the
        Boltzmann distribution calculated over the Q values
        after the update. Parameters:
            (1) T - float denoting the temperature parameter.

    '''

    def __init__(self, temperature=2.15, *args):
        super(Softmax, self).__init__(*args)
        self.T = temperature


    def action(self):
        # initialise the initial estimate for calculating probablities.
        pdf = [math.exp(i/self.T) for i in self.estimate.values()]
        sum_pdf = [pdf[i]/sum(pdf) for i in range(len(self.estimate))]
        softmax = stats.rv_discrete(values=(self.actions,sum_pdf))
        # sampling
        choice = softmax.rvs()
        return choice




class UCB1(Agent):
    ''' Upper Confidence Bound algorithm. Estimates


    '''
    def __init__(self, c=2, *args):
        super(UCB1, self).__init__(*args)
        self.c = c
        self.time = 0

    def action(self):
        # calculate ucb
        self.time += 1
        self.ucb = {i: self.estimate[i] +
                    self.c*math.sqrt((math.log(self.time))/(self.count[i]+1))
                    for i in self.estimate.keys()}
        choice = max(self.ucb, key=self.ucb.get)
        return choice

class MEA(Agent):
    '''


    '''
    def __init__(self, epsilon, delta, *args):
        super(MEA, self).__init__(*args)
        self.epsilon = epsilon
        self.delta = delta

    def action(self):
        pass



















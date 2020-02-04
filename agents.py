import random
import math
import scipy.stats as stats
import statistics

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

    def update(self, x):
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

    def __init__(self, temperature=0.5, *args):
        super(Softmax, self).__init__(*args)
        self.T = temperature


    def action(self):
        if(all(self.count.values())):
           # calculation of softmax probablities.
           pdf = [math.exp(i/self.T) for i in self.estimate.values()]
           sum_pdf = [pdf[i]/sum(pdf) for i in range(len(self.estimate))]
           softmax = stats.rv_discrete(values=(self.actions,sum_pdf))
           # sampling
           choice = softmax.rvs()
           return choice

        else:
           # initialise the estimates.
           initial_set = [x for i, x in enumerate(self.actions) if self.count[i]==0]
           choice = random.choice(initial_set)
           return choice



class UCB1(Agent):
    ''' Upper Confidence Bound algorithm.
        Generates the action from estimate computed using Q values
        and UCB term. takes C as a parameter as a scaling factor for
        UCB term.
    '''
    def __init__(self, constant=2, *args):
        super(UCB1, self).__init__(*args)
        # multiplicative factor in UCB algorithm.
        self.c = constant


    def action(self):
        if(all(self.count.values())):
           # calculate ucb
           self.ucb = {i:(self.estimate[i] + self.c*math.sqrt((math.log(sum(self.count)))/(self.count[i]))) for i in self.estimate.keys()}
           choice = max(self.ucb, key=self.ucb.get)
           index = self.actions.index(choice)
           return choice

        else:
           # initialise the estimates.
           initial_set = [x for i, x in enumerate(self.actions) if self.count[i]==0]
           choice = random.choice(initial_set)
           return choice






class MEA(Agent):
    ''' Median Elimination Algorithm. The idea is to throw the worst half of
        the arms at each iteration. Takes epsilon and delta as parameters.

    '''
    def __init__(self, epsilon, delta, *args):
        super(MEA, self).__init__(*args)
        self.epsilon = epsilon
        self.delta = delta
        self.iteration = 0
        self.optimal_states = self.actions.copy()
        self.data = [self.epsilon,self.delta]





    def action(self):
            # main loop.
            self.data[0] *= 0.75
            self.data[1] *= 0.5
            self.iteration +=1
















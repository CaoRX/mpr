import time
from lib.dqn_model import DQN
import numpy as np
import torch
from lib.wrappers import make_env
import random
import copy
from multiprocessing import Pool
import multiprocessing as mp
import itertools
import json
import os

ENV_NAME = "PongNoFrameskip-v4"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

STEP_UPPERBOUND = 99995

deviceName = 'cpu'
# deviceName = 'cuda'

device = torch.device(deviceName)

class AgentBase:

    totalGameN = None
    finishedGameN = 0

    def __init__(self, verbose = True, env = None):
        # self.env = gym.make(ENV_NAME)
        self.env = env if (env is not None) else make_env(ENV_NAME)
        self.verbose = verbose

    def getAction(self, state):
        action = self.env.action_space.sample()
        return action 

    def playEpisode(self):
        state = self.env.reset()

        if (self.verbose):
            startTime = time.time()

        step = 0

        totReward = 0.0
        while (step <= STEP_UPPERBOUND):
            action = self.getAction(state)
            newState, reward, done, _ = self.env.step(action)
            totReward += reward
            state = newState
            step += 1
            if done:
                break
        
        AgentBase.finishedGameN += 1
        if (self.verbose):
            # noStr = '{}'.format(self.No) if self.No is not None else ''
            noStr = '{}/{}'.format(AgentBase.finishedGameN, AgentBase.totalGameN) if AgentBase.totalGameN is not None else '{}'.format(AgentBase.finishedGameN)
            print('game {} end at step {}, reward = {}'.format(noStr, step, totReward))
            # endTime = time.time()
            # print('total time = {} seconds'.format(endTime - startTime))
        
        if step > STEP_UPPERBOUND:
            totReward = -21.0
            step = 0
        return (totReward, -step * totReward) # same reward: more steps are better if reward is minus, smaller is better if plus

class NNAgent(AgentBase):

    def __init__(self, verbose = True):
        super(NNAgent, self).__init__(verbose)
        env = self.env
        # print('env shape = {}'.format(env.observation_space.shape))
        self.net = DQN(env.observation_space.shape, env.action_space.n).to(device)

    def getAction(self, state):
        # print('state shape = {}'.format(state.shape))
        # action = self.env.action_space.sample()

        # state_a = np.array([self.state], copy=False)
        # state_v = torch.tensor(state_a).to(device)
        # q_vals_v = net(state_v)
        # _, act_v = torch.max(q_vals_v, dim=1)

        stateArray = np.array([state], copy = False)
        stateTensor = torch.tensor(stateArray).to(device)
        qValues = self.net(stateTensor)
        _, action = torch.max(qValues, dim = 1)
        # qVals = self.net(stateTensor).data.numpy()[0]
        # action = np.argmax(qVals)
        # print('get action {} from NNAgent'.format(action))
        return action

DEFAULT_M = 8
DEFAULT_N = 800
DEFAULT_SIGMA = 0.004
DEFAULT_CORE = 12

def splitList(data, core):
    n = len(data)
    if (core == 1):
        return [data]
    
    size = n // core
    rem = n % core
    curr = 0

    res = []
    for _ in range(core):
        if (rem > 0):
            res.append(data[curr : (curr + size + 1)])
            rem -= 1
            curr += (size + 1)
        else:
            res.append(data[curr : (curr + size)])
            curr += size
    
    return res

def mapPlayEpisode(agentList):
    # print(agentList)
    # return map(lambda x: x.playEpisode(), agentList)
    return [agent.playEpisode() for agent in agentList]

def splitMap(agents, core):
    with Pool(core) as p:
        agentSets = splitList(agents, core)
        # print(agentSets)
        # res = [p.apply_async(mapPlayEpisode, args = (agentSet,)) for agentSet in agentSets]
        # res = [process.get() for process in res]
        res = p.map(mapPlayEpisode, agentSets)
        # print('res = {}'.format(res))
        return list(itertools.chain.from_iterable(res))

def argsort(data, descending = False):
    n = len(data)
    dataWithIndices = [(i, data[i]) for i in range(n)]
    dataWithIndices = sorted(dataWithIndices, key = lambda x: x[1], reverse = descending)
    # print(dataWithIndices)

    return [x[0] for x in dataWithIndices]


class GeneticAgent:

    def __init__(self, agentGenerator, **kwargs):
        self.n = kwargs.get('n', DEFAULT_N)
        self.m = kwargs.get('m', DEFAULT_M)
        self.sigma = kwargs.get('sigma', DEFAULT_SIGMA)
        self.core = kwargs.get('core', DEFAULT_CORE)
        self.loadTimestamp = kwargs.get('load-timestamp', None)

        self.agents = [agentGenerator() for _ in range(self.n)]

        self.step = 0

        strTime = time.strftime("%Y%m%d-%H%M%S")
        os.mkdir('runs/run{}'.format(strTime))
        self.weightPath = 'runs/run{}/{}-best-weight'.format(strTime, strTime)

        self.outputPath = 'runs/run{}/{}-output.dat'.format(strTime, strTime)

        self.outputFile = open(self.outputPath, 'w')

        if (self.loadTimestamp is not None):
            self.load()

    def load(self):
        print('begin loading weights')
        if (self.loadTimestamp is None):
            return
        
        i = 0
        weightPath = 'runs/run{}/{}-best-weight'.format(self.loadTimestamp, self.loadTimestamp)
        while (True):
            nextFilename = weightPath + '-{}-{}.pt'.format(i + 1, self.m - 1)
            print('detecting {}'.format(nextFilename))
            if not os.path.isfile(nextFilename):
                break
            i += 1

        if i == 0:
            return
        
        for j in range(self.m):
            # net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
            filename = weightPath + '-{}-{}.pt'.format(i, j)
            print('loading weights from file {}'.format(filename))
            self.agents[j].net.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

        print('loading finished!')

    def getModifiedAgent(self):
        agent = copy.deepcopy(self.agents[random.randint(0, self.m - 1)])

        with torch.no_grad():
            for param in agent.net.parameters():
                param.add_(torch.randn(param.size()).to(device) * self.sigma)
        return agent

    def iterate(self, sigmaRate = 1.0):
        print('iteration step {}'.format(self.step))
        # for i in range(s]elf.n):
        #     self.agents[i].No = i
        startTime = time.time()
        self.sigma *= sigmaRate

        AgentBase.finishedGameN = 0
        AgentBase.totalGameN = self.n
        self.step += 1

        if (deviceName == 'cpu'):
        # if (False):
            fitness = splitMap(self.agents, self.core)
        else:
            fitness = [agent.playEpisode() for agent in self.agents]
        # fitness = splitMap(self.agents, self.core)
        indices = argsort(fitness, descending = True)
        # print(indices)

        print('best reward = {}'.format(fitness[indices[0]][0]))

        agents = [self.agents[x] for x in indices]

        for i in range(self.m, self.n):
            agent = agents[i]
            agents[i] = None
            del agent
        agents = agents[:self.m]

        self.agents = agents

        self.agents = [agents[0]] + [self.getModifiedAgent() for _ in range(self.n - 1)]

        endTime = time.time()
        print('step time: {} seconds.'.format(endTime - startTime))

        rewards = [x[0] for x in fitness]
        average = np.average(rewards)
        std = np.std(rewards)
        maxReward = rewards[indices[0]]

        for i in range(self.m):
            model = agents[i].net
            weightPath = self.weightPath + '-{}'.format(self.step) + '-{}.pt'.format(i)
            torch.save(model.state_dict(), weightPath)
            print('weight is saved to {}'.format(weightPath))

        print('max reward = {}, average reward = {}, reward std = {}'.format(maxReward, average, std))
        print('current sigma = {}'.format(self.sigma))

        res = {'step': self.step, 'average reward': average, 'std': std, 'max reward': maxReward}

        self.outputFile.write(json.dumps(res) + '\n')
        self.outputFile.flush()

        return res

def testFunc(x):
    return x * x

stepN = 3000

# rateMap = {
#     10: 0.5, 
#     20: 0.5, 
#     40: 0.2, 
#     80: 0.2,
# }

rateMap = {
    10: 0.5,
    40: 0.2
}

def generateSigmaRates(stepN):
    res = np.ones(stepN)
    for step in rateMap:
        if (stepN > step):
            res[step] = rateMap[step]
    # if (stepN > 10):
    #     res[10] = 0.5
    # if (stepN > 20):
    #     res[20] = 0.5
    # if (stepN > 40):
    #     res[40] = 0.2
    # if (stepN > 80):
    #     res[80] = 0.2

    return res

if __name__ == '__main__':
    # agent = NNAgent()
    # agent.playEpisode()

    # agent = NNAgent()
    # # print(agent.net.parameters())
    # for param in agent.net.parameters():
    #     print(param.data)

    # device = torch.device("cuda" if args.cuda else "cpu")
    # device = torch.device("cuda")

    mp.set_start_method('spawn')

    options = {
        'n': 200,
        'm': 8, 
        'sigma': 0.04,
        'core': 2,
        'load-timestamp': '20220113-115751'
    }

    # with Pool(3) as p:
    #     res = p.map(testFunc, [1, 2, 3])
    #     print(res)

    GA = GeneticAgent(NNAgent, **options)
    outputs = []
    sigmaRates = generateSigmaRates(stepN)
    for i in range(stepN):
        outputs.append(GA.iterate(sigmaRate = sigmaRates[i]))

    json.dump(outputs, 'finish.json')

    

        


        


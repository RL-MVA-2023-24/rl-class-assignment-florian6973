from gymnasium.wrappers import TimeLimit
from sklearn.ensemble import ExtraTreesRegressor
from env_hiv import HIVPatient
import numpy as np
from tqdm import tqdm
import random
import os
import torch
import pickle
from evaluate import evaluate_HIV
import torch.nn as nn
from copy import deepcopy

import gc

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.device = device
        self.data = []
        self.index = 0 # index of the next cell to be filled
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def build_buffer(self, env, nb_samples=2e6, replay_buffer_size=1e6):
        nb_samples = int(nb_samples)
        replay_buffer_size = int(replay_buffer_size)
        print("Testing insertion of", nb_samples, "samples in the replay buffer of size", replay_buffer_size)
        memory = ReplayBuffer(replay_buffer_size, self.device)
        state, _ = env.reset()
        for _ in tqdm(range(nb_samples)):
            action = env.action_space.sample()
            next_state, reward, done, trunc, _ = env.step(action)
            memory.append(state, action, reward, next_state, done)
            if done:
                state, _ = env.reset()
            else:
                state = next_state
        return memory
    
    def build_network(self, env, nb_neurons=50):        
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n 

        DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                                nn.ReLU(),
                                nn.Linear(nb_neurons, nb_neurons),
                                nn.ReLU(), 
                                nn.Linear(nb_neurons, n_action)).to(self.device)
        
        return DQN
            
    def collect_samples(self, env, horizon, random_rate=0, disable_tqdm=False, print_reset_states=False):
        s, _ = env.reset()
        #dataset = []
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in tqdm(range(horizon), disable=disable_tqdm):
            # a = env.action_space.sample() # not impacted by SEED CAREFUL
            if np.random.rand() < random_rate:
                a = np.random.randint(0, env.action_space.n)
            else:
                a = self.act(s)

            s2, r, done, trunc, _ = env.step(a)
            #dataset.append((s,a,r,s2,done,trunc))
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                s, _ = env.reset()
                if print_reset_states:
                    print("Resetting state to", s)
            else:
                s = s2
        S = np.array(S)
        A = np.array(A).reshape((-1,1))
        R = np.array(R)
        S2= np.array(S2)
        D = np.array(D)
        return S, A, R, S2, D
    
    def greedy_action(self, network, state):
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()
        
    def init_training(self, config, model):            
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,self.device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.target_model = deepcopy(self.model).to(self.device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0
    

    def MC_eval(self, env, nb_trials):   # NEW NEW NEW
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = self.greedy_action(self.model, x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)
    
    def V_initial_state(self, env, nb_trials):   # NEW NEW NEW
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(self.device)).max().item())
        return np.mean(val)
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

    def train_network(self, env, max_episode):
        episode_return = []
        MC_avg_total_reward = []   # NEW NEW NEW
        MC_avg_discounted_reward = []   # NEW NEW NEW
        V_init_state = []   # NEW NEW NEW
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict + (1-tau)*target_state_dict
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                # Monitoring
                if self.monitoring_nb_trials>0:
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)    # NEW NEW NEW
                    V0 = self.V_initial_state(env, self.monitoring_nb_trials)   # NEW NEW NEW
                    MC_avg_total_reward.append(MC_tr)   # NEW NEW NEW
                    MC_avg_discounted_reward.append(MC_dr)   # NEW NEW NEW
                    V_init_state.append(V0)   # NEW NEW NEW
                    episode_return.append(episode_cum_reward)   # NEW NEW NEW
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          ", MC tot ", '{:6.2f}'.format(MC_tr),
                          ", MC disc ", '{:6.2f}'.format(MC_dr),
                          ", V0 ", '{:6.2f}'.format(V0),
                          sep='')
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          sep='')

                
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state

    def train(self, S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
        nb_samples = S.shape[0]
        SA = np.append(S,A,axis=1)
        # a  = two boolean instead of a number?
        for iter in tqdm(range(iterations), disable=disable_tqdm):
            if iter==0:
                value=R.copy()
            else:
                Q2 = np.zeros((nb_samples,nb_actions))
                for a2 in range(nb_actions):
                    A2 = a2*np.ones((S.shape[0],1))
                    S2A2 = np.append(S2,A2,axis=1)
                    Q2[:,a2] = Q.predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = R + gamma*(1-D)*max_Q2 # d is one is the state is terminal
            Q = ExtraTreesRegressor(n_estimators=50, n_jobs=-1)
            Q.fit(SA,value)
        return Q

    # finite number of actions but continuous observation space
    # observations: T1, T1star, T2, T2star, V, E
    # actions: 0, 1, 2, 3 (no drug, first, second, both)
    def act(self, observation, use_random=False):
        # print(observation)
        if use_random:
            return np.random.choice(4)
        # return self.greedy_action(self.model, observation)
        Qsa = []
        nb_actions = 4
        for a in range(nb_actions):
            sa = np.append(observation,a).reshape(1, -1)
            Qsa.append(self.Q.predict(sa))
        policy = np.argmax(Qsa)
        # print(policy)
        return policy
        # return 0

    def save(self, path):
        if path is None:
            path = 'Q.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.Q, f)
        # if path is None:
        #     path = 'weights.pkl'
        # torch.save(self.model.state_dict(), path)

    def load(self):
        path = 'Q.pkl'
        if not os.path.exists(path):
            print("No model to load")
            return
        with open(path, 'rb') as f:
            self.Q = pickle.load(f)
        # self.model.load_state_dict(torch.load('weights.pkl'))

if __name__ == "__main__":
    seed_everything(seed=42)

    # find why not deterministic
    
    env = TimeLimit(
        env=HIVPatient(domain_randomization=False), max_episode_steps=200 # 200
    )  # The time wrapper limits the number of steps in an episode at 200.
    # Now is the floor is yours to implement the agent and train it.


    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
    agent = ProjectAgent()
    agent.load()

    # # training the agent
    # 400 times, 60 000 samples

    nb_iter_fitting = 400
    nb_sample_per_it = 6000
    for iteration in range(10):
        random_rate = 1 if iteration == 0 else 0.15 
        S, A, R, S2, D = agent.collect_samples(env, nb_sample_per_it, random_rate=random_rate, print_reset_states=True) # reset after 200 iterations?
        # import pickle
        with open('data.pkl', 'wb') as f:
            pickle.dump((S, A, R, S2, D), f)
        with open('data.pkl', 'rb') as f:
            S, A, R, S2, D = pickle.load(f)
        Q = agent.train(S, A, R, S2, D, nb_iter_fitting, 4, 1) # mean to compute reward
        agent.Q = Q
        agent.save(None)
        agent.load()

        score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
        print(score_agent)

    # # memory = agent.build_buffer(env, nb_samples=2e2, replay_buffer_size=1e2)
    # print("Building network...")
    # network = agent.build_network(env, nb_neurons=50)
    # print("Initializing training...")
    # agent.init_training(config={'nb_actions': env.action_space.n,
    #                             'buffer_size': 1000000,
    #                             }, model=network)
    # print("Training network...")
    # ep_length, disc_rewards, tot_rewards, V0 = agent.train_network(env, max_episode=2)

    # print("Evaluating agent...")
    # agent.save(None)
    # import matplotlib.pyplot as plt
    # plt.plot(ep_length, label="training episode length")
    # plt.plot(tot_rewards, label="MC eval of total reward")
    # plt.legend()
    # plt.show()
    # plt.figure()
    # plt.plot(disc_rewards, label="MC eval of discounted reward")
    # plt.plot(V0, label="average $max_a Q(s_0)$")
    # plt.legend()
    # plt.show()

    # s0,_ = env.reset()
    # Vs0 = np.zeros(nb_iter)
    # for i in range(len(Qfunctions)):
    #     Qs0a = []
    #     for a in range(env.action_space.n):
    #         s0a = np.append(s0,a).reshape(1, -1)
    #         Qs0a.append(Qfunctions[i].predict(s0a))
    #     Vs0[i] = np.max(Qs0a)
    # plt.plot(Vs0)
    # plt.show()

    

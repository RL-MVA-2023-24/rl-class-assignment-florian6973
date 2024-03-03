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
            
    def collect_samples(self, env, horizon, random_rate=0, disable_tqdm=False, print_reset_states=False):
        s, _ = env.reset()
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
    
    def train(self, S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
        nb_samples = S.shape[0]
        SA = np.append(S,A,axis=1)

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

    def act(self, observation, use_random=False):
        # print(observation)
        if use_random:
            return np.random.choice(4)
        Qsa = []
        nb_actions = 4
        for a in range(nb_actions):
            sa = np.append(observation,a).reshape(1, -1)
            Qsa.append(self.Q.predict(sa))
        policy = np.argmax(Qsa)
        return policy

    def save(self, path):
        if path is None:
            path = 'Q.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.Q, f)

    def load(self):
        path = 'Q.pkl'
        if not os.path.exists(path):
            print("No model to load")
            return
        with open(path, 'rb') as f:
            self.Q = pickle.load(f)

if __name__ == "__main__":
    seed_everything(seed=42)

    env = TimeLimit(
        env=HIVPatient(domain_randomization=False), max_episode_steps=200 # 200
    )  # The time wrapper limits the number of steps in an episode at 200.
    # Now is the floor is yours to implement the agent and train it.


    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
    agent = ProjectAgent()
    agent.load()

    nb_iter_fitting = 400
    nb_sample_per_it = 6000
    for iteration in range(10):
        random_rate = 1 if iteration == 0 else 0.15 
        S, A, R, S2, D = agent.collect_samples(env, nb_sample_per_it, random_rate=random_rate, print_reset_states=True) # reset after 200 iterations?

        Q = agent.train(S, A, R, S2, D, nb_iter_fitting, 4, 1) # mean to compute reward
        agent.Q = Q
        agent.save(None)
        agent.load()

        score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
        print(score_agent)
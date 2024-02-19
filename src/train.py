from gymnasium.wrappers import TimeLimit
from sklearn.ensemble import RandomForestRegressor
from env_hiv import HIVPatient
import numpy as np
from tqdm import tqdm
import random
import os
import torch
import pickle
from evaluate import evaluate_HIV

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
    def collect_samples(self, env, horizon, disable_tqdm=False, print_done_states=False):
        s, _ = env.reset()
        #dataset = []
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in tqdm(range(horizon), disable=disable_tqdm):
            a = env.action_space.sample()
            s2, r, done, trunc, _ = env.step(a)
            #dataset.append((s,a,r,s2,done,trunc))
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                s, _ = env.reset()
                if done and print_done_states:
                    print("done!")
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
        Qfunctions = []
        SA = np.append(S,A,axis=1)
        for iter in tqdm(range(iterations), disable=disable_tqdm):
            if iter==0:
                value=R.copy()
            else:
                Q2 = np.zeros((nb_samples,nb_actions))
                for a2 in range(nb_actions):
                    A2 = a2*np.ones((S.shape[0],1))
                    S2A2 = np.append(S2,A2,axis=1)
                    Q2[:,a2] = Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = R + gamma*(1-D)*max_Q2 # d is one is the state is terminal
            Q = RandomForestRegressor()
            Q.fit(SA,value)
            Qfunctions.append(Q)
        return Qfunctions

    # finite number of actions but continuous observation space
    # observations: T1, T1star, T2, T2star, V, E
    # actions: 0, 1, 2, 3 (no drug, first, second, both)
    def act(self, observation, use_random=False):
        # print(observation)
        Qsa = []
        nb_actions = 4
        for a in range(nb_actions):
            sa = np.append(observation,a).reshape(1, -1)
            Qsa.append(self.Q.predict(sa))
        policy = np.argmax(Qsa)
        print(policy)
        return policy
        # return 0

    def save(self, path):
        if path is None:
            path = 'Q.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.Q, f)

    def load(self):
        with open('Q.pkl', 'rb') as f:
            self.Q = pickle.load(f)

if __name__ == "__main__":
    seed_everything(seed=42)
    
    env = TimeLimit(
        env=HIVPatient(domain_randomization=False), max_episode_steps=200
    )  # The time wrapper limits the number of steps in an episode at 200.
    # Now is the floor is yours to implement the agent and train it.


    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
    agent = ProjectAgent()

    # training the agent
    S, A, R, S2, D = agent.collect_samples(env, 200)
    Qfunctions = agent.train(S, A, R, S2, D, 10, 4, 0.9)
    agent.Q = Qfunctions[-1]

    agent.save('Q.pkl')

    score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
    print(score_agent)

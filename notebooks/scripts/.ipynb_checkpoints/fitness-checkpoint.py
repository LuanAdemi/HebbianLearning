import torch
import numpy as np

import gym
import pybullet_envs

from gym import wrappers as w
from gym.spaces import Discrete, Box

from typing import List, Any

def fitness_hebb(environment : str, init_weights:str, evolved_parameters: np.ndarray) -> float:
    """
    Evaluate the policy network using some evolved parameters and environment.
    """
    def weights_init(m):
        if isinstance(m, torch.nn.Linear):
            if init_weights == 'xa_uni':  
                torch.nn.init.xavier_uniform(m.weight.data, 0.3)
            elif init_weights == 'sparse':  
                torch.nn.init.sparse_(m.weight.data, 0.8)
            elif init_weights == 'uni':  
                torch.nn.init.uniform_(m.weight.data, -0.1, 0.1)
            elif init_weights == 'normal':  
                torch.nn.init.normal_(m.weight.data, 0, 0.024)
            elif init_weights == 'ka_uni':  
                torch.nn.init.kaiming_uniform_(m.weight.data, 3)
            elif init_weights == 'uni_big':
                torch.nn.init.uniform_(m.weight.data, -1, 1)
            elif init_weights == 'xa_uni_big':
                torch.nn.init.xavier_uniform(m.weight.data)
            elif init_weights == 'ones':
                torch.nn.init.ones_(m.weight.data)
            elif init_weights == 'zeros':
                torch.nn.init.zeros_(m.weight.data)
            elif init_weights == 'default':
                pass
            
    # Unpack evolved parameters
    hebb_coeffs = evolved_parameters

    
    # disable the autograd system
    with torch.no_grad():
                    
        # Load environment
        try:
            env = gym.make(environment, verbose = 0)
        except:
            env = gym.make(environment)
            
        # env.render()  # render bullet envs

        # get the input dimensions of the environment
        input_dim = env.observation_space.shape[0]
            
        # Determine action space dimension
        action_dim = env.action_space.shape[0]
        
        # Initialize policy network
        p = HebbianNetwork(input_dim, action_dim)          
          
        # Randomly sample initial weights from chosen distribution
        p.apply(weights_init)
        p = p.float()
        
        # Unpack network's weights
        weights1_2, weights2_3, weights3_4 = list(p.parameters())
            
        # JIT
        weights1_2 = weights1_2.detach().numpy()
        weights2_3 = weights2_3.detach().numpy()
        weights3_4 = weights3_4.detach().numpy()
        
        # reset the environment
        observation = env.reset() 

        # Burnout phase for the bullet ant so it starts off from the floor
        if environment == 'AntBulletEnv-v0':
            action = np.zeros(8)
            for _ in range(40):
                __ = env.step(action)        
        
        # Normalize weights flag for non-bullet envs
        normalised_weights = False if environment[-12:-6] == 'Bullet' else True


        # Main loop
        neg_count = 0 # count the amount of times we receive a negative reward
        rew_ep = 0 # cummulative reward over an episode
        t = 0 # timestep
        
        while True:
            
            # For obaservation ∈ gym.spaces.Discrete, we one-hot encode the observation
            if isinstance(env.observation_space, Discrete): 
                observation = (observation == torch.arange(env.observation_space.n)).float()
            
            o0, o1, o2, o3 = p([observation])
            
            # JIT
            o0 = o0.numpy()
            o1 = o1.numpy()
            o2 = o2.numpy()
            
            # preprocess the observation
            if environment[-12:-6] == 'Bullet':
                o3 = torch.tanh(o3).numpy()
                action = o3
            else: 
                if isinstance(env.action_space, Box):
                    action = o3.numpy()                        
                    action = np.clip(action, env.action_space.low, env.action_space.high)  
                elif isinstance(env.action_space, Discrete):
                    action = np.argmax(o3).numpy()
                o3 = o3.numpy()

            
            # Environment simulation step
            observation, reward, done, info = env.step(action)  
            if environment == 'AntBulletEnv-v0': reward = env.unwrapped.rewards[1] # Distance walked
            rew_ep += reward
            
            # env.render('human') # Gym envs
                                       
            # Early stopping conditions
            if environment[-12:-6] == 'Bullet':
                ## Special stopping condition for bullet envs
                # always play 200 episodes
                if t > 200:
                    # after 200 episodes: count the amount of negative
                    # reward we receive in a row
                    neg_count = neg_count+1 if reward < 0.0 else 0
                    
                    # if we receive a negative reward 30 times in row, stop
                    if (done or neg_count > 30):
                        break
            else:
                if done:
                    break
            
            t += 1
            
            #### Episodic/Intra-life hebbian update of the weights
            weights1_2, weights2_3, weights3_4 = hebbian_update_rule(hebb_coeffs, weights1_2, weights2_3, weights3_4, o0, o1, o2, o3)
            

            # Normalise weights per layer
            if normalised_weights == True:
                (a, b, c) = (0, 1, 2) if not pixel_env else (2, 3, 4)
                list(p.parameters())[a].data /= list(p.parameters())[a].__abs__().max()
                list(p.parameters())[b].data /= list(p.parameters())[b].__abs__().max()
                list(p.parameters())[c].data /= list(p.parameters())[c].__abs__().max()
        
        # close the environment
        env.close()

    return rew_ep

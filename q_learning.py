#!/usr/bin/env python
# coding: utf-8

# # Q-Learning on Viral Load Data for Optimal Treatment Strategies

# ## Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import itertools


# In[2]:


data = pd.read_excel('/Users/hamiddastgir/Desktop/mgddll6.xlsx')


# In[3]:


data = data.drop(columns='Unnamed: 0')


# In[4]:


data.head()


# ## Defining State and Action Spaces

# In[5]:


state_columns = ['HGB_LC', 'MCV_LC', 'PLATLC', 'WBC_LC', 'HSRAT']
states = data[state_columns].values


# In[6]:


action_columns = ['NRTI', 'NNRTI', 'PI', 'OTHER']
action = data[action_columns].values


# In[7]:


scaler = MinMaxScaler()
normalized_states = scaler.fit_transform(states)


# In[8]:


action_combinations = list(itertools.product([0, 1], repeat=len(action_columns)))


# In[9]:


action_mapping = {combo: idx for idx, combo in enumerate(action_combinations)}
inverse_action_mapping = {idx: combo for combo, idx in action_mapping.items()}


# In[10]:


action_tuples = [tuple(a) for a in action]
action_indices = [action_mapping[a] for a in action_tuples]


# In[11]:


num_actions = len(action_combinations)


# ## Discretizing the State Space

# In[12]:


num_bins = 10
state_bins = [np.linspace(0, 1, num_bins + 1) for _ in range(normalized_states.shape[1])]


# In[13]:


def discretize_state(state):
    discretized_state = []
    for i in range(len(state)):
        discretized_value = np.digitize(state[i], state_bins[i]) - 1
        discretized_state.append(discretized_value)
    return tuple(discretized_state)


# ## Initializing the Q-Table

# In[14]:


state_space = set()
for state in normalized_states:
    state_space.add(discretize_state(state))
state_mapping = {state: idx for idx, state in enumerate(state_space)}
inverse_state_mapping = {idx: state for state, idx in state_mapping.items()}

num_states = len(state_mapping)
q_table = np.zeros((num_states, num_actions))


# ## Setting Q-Learning Parameters

# In[15]:


learning_rate_alpha = 0.1
discount_factor_gamma = 0.5
exploration_rate_epsilon = 0.1
# optional: exploration_rate_epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)


# In[16]:


VLOAD = data['VLOAD'].values


# ## Implementing the Q-Learning Algorithm

# In[17]:


for episode in range(1000):
    for i in range(len(normalized_states) - 1):
        state = normalized_states[i]
        discretized_state = discretize_state(state)
        state_index = state_mapping[discretized_state]
        
        if np.random.uniform(0, 1) < exploration_rate_epsilon:
            action_index = np.random.choice(num_actions)
        else:
            action_index = np.argmax(q_table[state_index, :])
        
        actual_action_index = action_indices[i]
        
        agent_action_vector = np.array(action_combinations[action_index])
        actual_action_vector = np.array(action_combinations[actual_action_index])

        action_similarity = np.sum(agent_action_vector == actual_action_vector) / len(action_columns)

        reward = VLOAD[i] - VLOAD[i + 1]
        
        adjusted_reward = reward * action_similarity
        
        next_state = normalized_states[i + 1]
        discretized_next_state = discretize_state(next_state)
        next_state_index = state_mapping[discretized_next_state]
        next_max = np.max(q_table[next_state_index, :])
        
        q_table[state_index, action_index] = (1 - learning_rate_alpha) * q_table[state_index, action_index] + \
                                             learning_rate_alpha * (adjusted_reward + discount_factor_gamma * next_max)


# ## Creating DataFrames and Saving Results

# In[18]:


q_table_df = pd.DataFrame(q_table, columns=[str(combo) for combo in action_combinations])
q_table_df


# In[20]:


state_index_to_normalized_state = {}
for state in normalized_states:
    discretized_state = discretize_state(state)
    state_index = state_mapping[discretized_state]
    state_index_to_normalized_state[state_index] = state

normalized_states_list = [state_index_to_normalized_state[idx] for idx in state_indices]

normalized_state_df = pd.DataFrame(normalized_states_list, columns=[col + '_norm' for col in state_columns])

full_df = pd.concat([state_df, normalized_state_df, q_table_df], axis=1)

full_df.to_csv('q_table_with_og_states.csv', index=False)


# In[ ]:





# In[ ]:





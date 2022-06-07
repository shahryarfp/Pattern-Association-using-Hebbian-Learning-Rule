#!/usr/bin/env python
# coding: utf-8

# # Q1
# 

# In[1]:


import numpy as np

num_of_inputs = 3
input_1 = np.array([
    -1,-1,-1,-1,-1,-1,1,
    -1,-1,-1,-1,-1,-1,1,
    1,1,1,1,1,1,1,
    1,-1,-1,1,-1,-1,-1,
    1,-1,-1,1,-1,-1,-1,
    -1,-1,-1,1,-1,-1,-1,
    -1,-1,-1,1,-1,-1,-1,
    -1,-1,-1,1,-1,-1,-1,
    -1,-1,-1,1,-1,-1,-1,
])

input_2 = np.array([
    1,-1,-1,-1,-1,-1,1,
    1,-1,-1,-1,-1,-1,1,
    1,-1,-1,-1,-1,-1,1,
    1,-1,-1,-1,-1,-1,1,
    1,-1,-1,-1,-1,-1,1,
    1,-1,-1,-1,-1,-1,1,
    1,1,1,1,1,1,1,
    -1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,1,-1,-1,-1,
])


input_3 = np.array([
    1,-1,-1,-1,-1,-1,1,
    1,-1,-1,-1,-1,-1,1,
    1,-1,-1,-1,-1,-1,1,
    1,-1,-1,-1,-1,-1,1,
    1,-1,-1,1,-1,-1,1,
    1,-1,-1,-1,-1,-1,1,
    1,-1,-1,-1,-1,-1,1,
    1,-1,-1,-1,-1,-1,1,
    1,1,1,1,1,1,1,
])


output_1 = np.array([
    -1,-1,1,
    1,1,1,
    1,1,-1,
    -1,1,-1,
    -1,1,-1,
])

output_2 = np.array([
    1,-1,1,
    1,-1,1,
    1,1,1,
    -1,-1,-1,
    -1,1,-1
])

output_3 = np.array([
    1,-1,1,
    1,-1,1,
    1,1,1,
    1,-1,1,
    1,1,1
])


# ## Part 2
# ### Initializing weight vector

# In[2]:


weight_vector = np.zeros((len(input_1),len(output_1)))
weight_vector.shape


# ### There are 3 training samples, so there will be 3 iterations

# In[3]:


def update_weight(input_, output_):
    for i in range(len(output_)):
        multiplied_list = [element * output_[i] for element in input_]
        weight_vector[:,i] += multiplied_list
        
update_weight(input_1, output_1)
update_weight(input_2, output_2)
update_weight(input_3, output_3)

print(weight_vector.shape)
print(weight_vector)


# ## Part 3

# In[4]:


import copy
# import plotly.express as px
def plot_matrix(output_, dimension):
    temp = copy.deepcopy(output_)
    temp = np.reshape(temp, dimension)
    print(temp)


# In[5]:


def my_sign(x):
    if x>=0:
        return 1
    else:
        return -1
def predict(input_):
    input__ = copy.deepcopy(input_)
    ans = np.matmul(np.transpose(input__), weight_vector)
    for i in range(len(ans)):
        ans[i] = my_sign(ans[i])
    return ans 

ans = predict(input_1)
print('1:')
print('predicted:', ans)
print('real ans:', output_1)
if list(ans) == list(output_1):
    print('Predicted Correctly')
else:
    print('Predicted Wrong')
plot_matrix(ans, (5,3))
print(' ')

ans = predict(input_2)
print('2:')
print('predicted:', ans)
print('real ans:', output_2)
if list(ans) == list(output_2):
    print('Predicted Correctly')
else:
    print('Predicted Wrong')
plot_matrix(ans, (5,3))
print(' ')

ans = predict(input_3)
print('3:')
print('predicted:', ans)
print('real ans:', output_3)
if list(ans) == list(output_3):
    print('Predicted Correctly')
else:
    print('Predicted Wrong')
plot_matrix(ans, (5,3))


# ## Part 4

# In[6]:


#Changing Outputs
dim = (1,2)

output_1 = np.array([
    -1,-1
])

output_2 = np.array([
    1,-1
])

output_3 = np.array([
    -1,1
])


# In[7]:


weight_vector = np.zeros((len(input_1),len(output_1)))
update_weight(input_1, output_1)
update_weight(input_2, output_2)
update_weight(input_3, output_3)

ans = predict(input_1)
print('1:')
print('predicted:', ans)
print('real ans:', output_1)
if list(ans) == list(output_1):
    print('Predicted Correctly')
else:
    print('Predicted Wrong')
plot_matrix(ans, dim)
print(' ')

ans = predict(input_2)
print('2:')
print('predicted:', ans)
print('real ans:', output_2)
if list(ans) == list(output_2):
    print('Predicted Correctly')
else:
    print('Predicted Wrong')
plot_matrix(ans, dim)
print(' ')

ans = predict(input_3)
print('3:')
print('predicted:', ans)
print('real ans:', output_3)
if list(ans) == list(output_3):
    print('Predicted Correctly')
else:
    print('Predicted Wrong')
plot_matrix(ans, dim)
print(' ')


# ## Part 5

# ### Adding noise to inputs
# #### Part 2 output Dimension

# In[8]:


def reset_inputs(input_1, input_2, input_3):
    input_1 = np.array([
        -1,-1,-1,-1,-1,-1,1,
        -1,-1,-1,-1,-1,-1,1,
        1,1,1,1,1,1,1,
        1,-1,-1,1,-1,-1,-1,
        1,-1,-1,1,-1,-1,-1,
        -1,-1,-1,1,-1,-1,-1,
        -1,-1,-1,1,-1,-1,-1,
        -1,-1,-1,1,-1,-1,-1,
        -1,-1,-1,1,-1,-1,-1,
    ])

    input_2 = np.array([
        1,-1,-1,-1,-1,-1,1,
        1,-1,-1,-1,-1,-1,1,
        1,-1,-1,-1,-1,-1,1,
        1,-1,-1,-1,-1,-1,1,
        1,-1,-1,-1,-1,-1,1,
        1,-1,-1,-1,-1,-1,1,
        1,1,1,1,1,1,1,
        -1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,1,-1,-1,-1,
    ])


    input_3 = np.array([
        1,-1,-1,-1,-1,-1,1,
        1,-1,-1,-1,-1,-1,1,
        1,-1,-1,-1,-1,-1,1,
        1,-1,-1,-1,-1,-1,1,
        1,-1,-1,1,-1,-1,1,
        1,-1,-1,-1,-1,-1,1,
        1,-1,-1,-1,-1,-1,1,
        1,-1,-1,-1,-1,-1,1,
        1,1,1,1,1,1,1,
    ])
    
    return input_1, input_2, input_3
    
output_1 = np.array([
    -1,-1,1,
    1,1,1,
    1,1,-1,
    -1,1,-1,
    -1,1,-1,
])

output_2 = np.array([
    1,-1,1,
    1,-1,1,
    1,1,1,
    -1,-1,-1,
    -1,1,-1
])

output_3 = np.array([
    1,-1,1,
    1,-1,1,
    1,1,1,
    1,-1,1,
    1,1,1
])

input_1, input_2, input_3 = reset_inputs(input_1, input_2, input_3)
weight_vector = np.zeros((len(input_1),len(output_1)))
update_weight(input_1, output_1)
update_weight(input_2, output_2)
update_weight(input_3, output_3)


import random
def add_noise(input_, noise):
    for i in range(len(input_)):
        rand = random.random()
        if rand < noise:
            input_[i] *= -1

noises = [0.2, 0.6]
for noise in noises:
    winning_num = 0
    for _ in range(100):
        input_1, input_2, input_3 = reset_inputs(input_1, input_2, input_3)
        add_noise(input_1, noise)
        add_noise(input_2, noise)
        add_noise(input_3, noise)
        
        ans1 = predict(input_1)
        ans2 = predict(input_2)
        ans3 = predict(input_3)
        if list(ans1) == list(output_1):
            winning_num += 1
        if list(ans2) == list(output_2):
            winning_num += 1
        if list(ans3) == list(output_3):
            winning_num += 1
    print('Noise:', noise*100, '%')
    print('Accuracy:', winning_num/(num_of_inputs*100)*100,'%')
    print('')


# #### Part 4 output Dimension

# In[38]:


#Changing Outputs
output_1 = np.array([
    -1,-1
])

output_2 = np.array([
    1,-1
])

output_3 = np.array([
    -1,1
])

input_1, input_2, input_3 = reset_inputs(input_1, input_2, input_3)
weight_vector = np.zeros((len(input_1),len(output_1)))
update_weight(input_1, output_1)
update_weight(input_2, output_2)
update_weight(input_3, output_3)

noises = [0.2, 0.6]
for noise in noises:
    winning_num = 0
    for _ in range(100):
        input_1, input_2, input_3 = reset_inputs(input_1, input_2, input_3)
        add_noise(input_1, noise)
        add_noise(input_2, noise)
        add_noise(input_3, noise)
        
        ans1 = predict(input_1)
        ans2 = predict(input_2)
        ans3 = predict(input_3)
        if list(ans1) == list(output_1):
            winning_num += 1
        if list(ans2) == list(output_2):
            winning_num += 1
        if list(ans3) == list(output_3):
            winning_num += 1
    print('Noise:', noise*100, '%')
    print('Accuracy:', winning_num/(num_of_inputs*100)*100,'%')
    print('')


# ## Part 6

# ### Removing datas
# #### Part 2 output Dimension

# In[36]:


output_1 = np.array([
    -1,-1,1,
    1,1,1,
    1,1,-1,
    -1,1,-1,
    -1,1,-1,
])

output_2 = np.array([
    1,-1,1,
    1,-1,1,
    1,1,1,
    -1,-1,-1,
    -1,1,-1
])

output_3 = np.array([
    1,-1,1,
    1,-1,1,
    1,1,1,
    1,-1,1,
    1,1,1
])

input_1, input_2, input_3 = reset_inputs(input_1, input_2, input_3)
weight_vector = np.zeros((len(input_1), len(output_1)))
update_weight(input_1, output_1)
update_weight(input_2, output_2)
update_weight(input_3, output_3)

def remove_noise(input_, noise):
    for i in range(len(input_)):
        rand = random.random()
        if rand < noise:
            input_[i] = 0

noises = [0.2, 0.6]
for noise in noises:
    winning_num = 0
    for _ in range(100):
        input_1, input_2, input_3 = reset_inputs(input_1, input_2, input_3)
        remove_noise(input_1, noise)
        remove_noise(input_2, noise)
        remove_noise(input_3, noise)
        
        ans1 = predict(input_1)
        ans2 = predict(input_2)
        ans3 = predict(input_3)
        if list(ans1) == list(output_1):
            winning_num += 1
        if list(ans2) == list(output_2):
            winning_num += 1
        if list(ans3) == list(output_3):
            winning_num += 1
    print('Noise:', noise*100, '%')
    print('Accuracy:', winning_num/(num_of_inputs*100)*100,'%')
    print('')


# #### Part 4 output Dimension

# In[26]:


#Changing Outputs
output_1 = np.array([
    -1,-1
])

output_2 = np.array([
    1,-1
])

output_3 = np.array([
    -1,1
])

input_1, input_2, input_3 = reset_inputs(input_1, input_2, input_3)
weight_vector = np.zeros((len(input_1), len(output_1)))
update_weight(input_1, output_1)
update_weight(input_2, output_2)
update_weight(input_3, output_3)

noises = [0.2, 0.6]
for noise in noises:
    winning_num = 0
    for _ in range(100):
        input_1, input_2, input_3 = reset_inputs(input_1, input_2, input_3)
        remove_noise(input_1, noise)
        remove_noise(input_2, noise)
        remove_noise(input_3, noise)
        
        ans1 = predict(input_1)
        ans2 = predict(input_2)
        ans3 = predict(input_3)
        if list(ans1) == list(output_1):
            winning_num += 1
        if list(ans2) == list(output_2):
            winning_num += 1
        if list(ans3) == list(output_3):
            winning_num += 1
    print('Noise:', noise*100, '%')
    print('Accuracy:', winning_num/(num_of_inputs*100)*100,'%')
    print('')


# In[ ]:





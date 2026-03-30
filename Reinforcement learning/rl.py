import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

maze = np.array([
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 0]
])

start = (0, 0)
goal = (9, 9)

num_episodes = 5000
alpha = 0.1
gamma = 0.9
epsilon = 0.5

reward_fire = -10
reward_goal = 50
reward_step = -1

actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

Q = np.zeros(maze.shape + (len(actions),))

def is_valid(pos):
    r, c = pos
    if r < 0 or r >= maze.shape[0]:
        return False
    if c < 0 or c >= maze.shape[1]:
        return False
    if maze[r, c] == 1:
        return False
    return True


def choose_action(state):
    if np.random.random() < epsilon:
        return np.random.randint(len(actions))
    else:
        return np.argmax(Q[state])
    
def step(state, action):
    r, c = state
    dr, dc = actions[action]
    new_state = (r + dr, c + dc)
    
    if not is_valid(new_state):
        return state, reward_fire
    
    if new_state == goal:
        return new_state, reward_goal
    
    return new_state, reward_step   

for episode in range(num_episodes):
    state = start
    while state != goal:
        action = choose_action(state)
        new_state, reward = step(state, action)
        
        best_next_q = np.max(Q[new_state])
        Q[state + (action,)] += alpha * (reward + gamma * best_next_q - Q[state + (action,)])
        
        state = new_state       

cmap = ListedColormap(['white', 'black', 'red', 'green'])
maze_visual = np.copy(maze)     


state = start
while state != goal:
    action = np.argmax(Q[state])
    r, c = state
    dr, dc = actions[action]
    new_state = (r + dr, c + dc)
    
    if not is_valid(new_state):
        break
    
    maze_visual[new_state] = 2  
    state = new_state

maze_visual[goal] = 3   
plt.figure(figsize=(6, 6))
plt.imshow(maze_visual, cmap=cmap)  

plt.xticks([])
plt.yticks([])  
plt.title('Learned Path from Start to Goal')
plt.show()


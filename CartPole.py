import numpy as np 
import time
import gym
import cv2
import pickle

from sklearn.preprocessing import KBinsDiscretizer
import math, random
from typing import Tuple

# Create the CartPole environment
env = gym.make('CartPole-v1', render_mode='rgb_array')

# Define a simple policy (always choose action 1)
policy = lambda *obs: 1

# Define video writer for recording the environment
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for the video writer
out = cv2.VideoWriter('cartpole_video.mp4', fourcc, 20.0, (600, 400))  # Set the output video parameters

# Reset the environment and record the initial observation
obs = env.reset()

# Main loop for rendering and recording the environment
while True:
    actions = policy(*obs)  # Choose action based on the current observation
    obs, reward, done, info = env.step(actions)[:4]  # Take a step in the environment
    frame = env.render()  # Render the current frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert color format for compatibility
    out.write(frame)  # Write the frame to the video
    cv2.imshow('CartPole', frame)  # Display the frame
    if cv2.waitKey(1) & 0xFF == ord('q') or done:  # Exit if 'q' is pressed or episode ends
        break

# Release the video writer and close the environment
out.release()
cv2.destroyAllWindows()
env.close()

# Define parameters for discretization
n_bins = (6, 12)
lower_bounds = [env.observation_space.low[2], -math.radians(60)]
upper_bounds = [env.observation_space.high[2], math.radians(60)]

def discretizer(_, __, angle, pole_velocity) -> Tuple[int,...]:
    """Convert continuous state into a discrete state"""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample=None)
    est.fit([lower_bounds, upper_bounds])
    return tuple(map(int, est.transform([[angle, pole_velocity]])[0]))

# Initialize the Q-table with zeros
Q_table = np.zeros(n_bins + (env.action_space.n,))
Q_table.shape

def policy(state: tuple):
    """Choose action based on epsilon-greedy policy"""
    return np.argmax(Q_table[state])

def new_Q_value(reward: float, new_state: tuple, discount_factor=1) -> float:
    """Temporal difference for updating Q-value of state-action pair"""
    future_optimal_value = np.max(Q_table[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value

def learning_rate(n: int, min_rate=0.01) -> float:
    """Decaying learning rate"""
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))

def exploration_rate(n: int, min_rate=0.1) -> float:
    """Decaying exploration rate"""
    return max(min_rate, min(1, 1.0 - math.log10((n + 1) / 25)))

# Training
n_episodes = 10000  # Number of training episodes

for e in range(n_episodes):
    current_state, done = discretizer(*env.reset(), 0, 0), False  # Reset the environment and discretize the initial state
    
    while not done:
        action = policy(current_state)  # Choose action based on the current state
        
        if np.random.random() < exploration_rate(e):  # Epsilon-greedy exploration
            action = env.action_space.sample()  # Randomly select an action
            
        obs, reward, done, _ = env.step(action)[:4]  # Take a step in the environment and get the next observation
        new_state = discretizer(*obs)  # Discretize the new state
        
        lr = learning_rate(e)  # Decay learning rate
        learnt_value = new_Q_value(reward, new_state)  # Calculate the new Q-value
        old_value = Q_table[current_state][action]  # Get the old Q-value
        Q_table[current_state][action] = (1 - lr) * old_value + lr * learnt_value  # Update Q-value using the Bellman equation
        
        current_state = new_state  # Update the current state for the next iteration
    
        frame = env.render()  # Render the environment
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR format for compatibility with OpenCV
        cv2.imshow('CartPole', frame)  # Display the rendered frame
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
            break
        time.sleep(0.02)  # Adjust sleep time for smoother rendering
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit if 'q' is pressed
        break

cv2.destroyAllWindows()  # Close OpenCV windows
env.close()  # Close the environment

# Print the final Q-table after training
print("Q-Table after training:")
print(Q_table)

# Save the Q-table to a file after training
with open('Q_table.pkl', 'wb') as f:
    pickle.dump(Q_table, f)
print("Q-table saved successfully.")

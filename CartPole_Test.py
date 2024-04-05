import numpy as np  # Import NumPy for numerical operations
import cv2  # Import OpenCV for rendering
import gym  # Import Gym for reinforcement learning environments
import pickle  # Import Pickle for saving and loading Q-table
from sklearn.preprocessing import KBinsDiscretizer  # Import KBinsDiscretizer for state discretization
from typing import Tuple  # Import Tuple for type hinting
import math, random  # Import Math and Random for mathematical operations and random number generation
import time  # Import Time for timing the episodes

# Load Q-table from file
try:
    with open('Q_table.pkl', 'rb') as f:  # Open Q_table.pkl file for reading in binary mode
        Q_table = pickle.load(f)  # Load Q-table from file
    print("Q-table loaded successfully.")  # Print message indicating successful loading
except FileNotFoundError:  # Handle the case when Q_table.pkl file is not found
    print("No existing Q-table found. Exiting...")  # Print message indicating absence of Q-table
    exit()  # Exit the program

# Print Q_table (for verification)
print(Q_table)

env = gym.make('CartPole-v1', render_mode='rgb_array')  # Create the CartPole environment

n_testing_episodes = 20  # Number of episodes for testing

n_bins = (6, 12)  # Number of bins for discretization
lower_bounds = [env.observation_space.low[2], -math.radians(60)]  # Lower bounds for angle and pole velocity
upper_bounds = [env.observation_space.high[2], math.radians(60)]  # Upper bounds for angle and pole velocity

# Define evaluation metrics
total_rewards = []
episode_durations = []

def discretizer(_, __, angle, pole_velocity) -> Tuple[int, ...]:
    """Convert continuous state into a discrete state"""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample=None)  # Initialize discretizer
    est.fit([lower_bounds, upper_bounds])  # Fit the discretizer to the given bounds
    return tuple(map(int, est.transform([[angle, pole_velocity]])[0]))  # Discretize the input state and return

def policy(state: tuple):
    """Choosing action based on epsilon-greedy policy"""
    return np.argmax(Q_table[state])  # Choose action with the highest Q-value for the given state

for e in range(n_testing_episodes):  # Iterate over testing episodes
    current_state, done = discretizer(*env.reset(), 0, 0), False  # Reset the environment and discretize the initial state
    episode_reward = 0
    start_time = time.time()  # Record the start time of the episode
    while not done:  # Main loop for the episode
        action = policy(current_state)  # Choose action based on the current state
        obs, reward, done, _ = env.step(action)[:4]  # Take a step in the environment
        new_state = discretizer(*obs)  # Discretize the new state
        current_state = new_state  # Update the current state for the next iteration
        episode_reward += reward  # Accumulate episode reward

        frame = env.render()  # Render the environment
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert color format for compatibility
        cv2.imshow('CartPole Test', frame)  # Display the rendered frame
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Check for 'q' key press to exit
            break
    
    end_time = time.time()  # Record the end time of the episode
    episode_duration = end_time - start_time  # Calculate the duration of the episode
    print(f"Episode {e + 1} duration: {episode_duration:.2f} seconds, Total Reward: {episode_reward}")  # Print episode duration and total reward

    total_rewards.append(episode_reward)
    episode_durations.append(episode_duration)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Check for 'q' key press to exit
        break

# Print average total reward and average episode duration
print(f"Average Total Reward: {np.mean(total_rewards)}")
print(f"Average Episode Duration: {np.mean(episode_durations)}")

cv2.destroyAllWindows()  # Close OpenCV windows
env.close()  # Close the environment


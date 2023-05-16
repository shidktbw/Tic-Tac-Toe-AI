# Tic-Tac-Toe-AI

This repository contains a Python implementation of the classic game Tic-Tac-Toe, which uses Q-Learning and a simple neural network model implemented with TensorFlow for the AI player. 

# Features
* Play against an AI player train with Q-Learning
* Graphical User Interface with Pygame
* AI player uses a neural network model implemented with TensorFlow
* AI player can update its knowledge after each game and become better over time

# AI Training

The AI player uses Q-Learning to make decisions. The state of the game board is passed through a neural network to estimate the Q-values of each possible action. The AI player then selects the action with the highest Q-value.

The AI player updates its knowledge after each game - if it wins, it's more likely to repeat the successful actions, and if it loses, it's less likely to repeat the unsuccessful actions. Over time, the AI player becomes better at the game.

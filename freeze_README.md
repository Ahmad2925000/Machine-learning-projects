# Q-Learning Hyperparameter Tuning & Policy Evaluation in R

##  Project Objective
The primary goal of this project is to implement an automated **Grid Search and Evaluation Pipeline** for a Q-Learning agent in a grid-world environment. 

In Reinforcement Learning, simply training an agent isn't enough; we need to scientifically determine which combination of hyperparameters—specifically the **Discount Factor ($\gamma$)** and **Exploration Rate ($\epsilon$)**—yields the most reliable and high-performing policy. This script separates the **training phase** from the **frozen evaluation phase** to measure true deployment performance.

---

##  Methodology & Approach

1. **Hyperparameter Grid Search:** Evaluates 9 distinct pairs of `gamma` and `epsilon`.
2. **Training Phase (30,000 Episodes):** 
   - For each pair, the agent trains using an $\epsilon$-greedy policy, updating the Q-table to learn the environment dynamics.
3. **Freezing the Q-Table (`freeze = TRUE`):** 
   - Training is locked to simulate real-world deployment where the agent relies strictly on its acquired knowledge without further updates.
4. **Evaluation Phase (1,000 Episodes):** 
   - The frozen agent runs for 1,000 test episodes. 
   - Terminal states are tracked (+10 for success, -1 for penalty). 
   - The average reward across these episodes serves as the definitive performance metric.
5. **Optimal Selection:** 
   - The script compares all 9 averages and outputs the best-performing hyperparameter combination.

---

##  How to Run
1. Ensure R and the `ggplot2` package are installed.
2. Run the script to execute the grid search, evaluate performance, and visualize the optimal Q-table.

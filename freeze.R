library(ggplot2)
#added a comment
# The following code defines how each action is displayed and how it changes the agent’s position.
arrows <- c("^", ">", "v", "<")
action_deltas <- list(c(1,0),  # up
                      c(0,1),  # right
                      c(-1,0), # down
                      c(0,-1)) # left

vis_environment <- function(iterations=0, epsilon = 0.5, alpha = 0.1, gamma = 0.95, beta = 0){
  df <- expand.grid(x=1:H, y=1:W)
  foo <- mapply(function(x,y) ifelse(reward_map[x,y]== 0, q_table[x,y,1], NA), df$x, df$y)
  df$val1 <- as.vector(round(foo, 2))
  foo <- mapply(function(x,y) ifelse(reward_map[x,y]== 0, q_table[x,y,2], NA), df$x, df$y)
  df$val2 <- as.vector(round(foo, 2))
  foo <- mapply(function(x,y) ifelse(reward_map[x,y]== 0, q_table[x,y,3], NA), df$x, df$y)
  df$val3 <- as.vector(round(foo, 2))
  foo <- mapply(function(x,y) ifelse(reward_map[x,y]== 0, q_table[x,y,4], NA), df$x, df$y)
  df$val4 <- as.vector(round(foo, 2))
  foo <- mapply(function(x,y) 
    ifelse(reward_map[x,y]== 0, arrows[GreedyPolicy(x,y)], reward_map[x,y]), df$x, df$y)
  df$val5 <- as.vector(foo)
  foo <- mapply(function(x,y) ifelse(reward_map[x,y]== 0, max(q_table[x,y,]),
                                     ifelse(reward_map[x,y]<0, NA, reward_map[x,y])), df$x, df$y)
  df$val6 <- as.vector(foo)
  
  print(ggplot(df, aes(x = y, y = x)) +
          scale_fill_gradient(low = "white", high = "green", na.value = "red", name = "") +
          geom_tile(aes(fill=val6)) +
          geom_text(aes(label = val1), size = 4, nudge_y = .35, na.rm = TRUE) +
          geom_text(aes(label = val2), size = 4, nudge_x = .35, na.rm = TRUE) +
          geom_text(aes(label = val3), size = 4, nudge_y = -.35, na.rm = TRUE) +
          geom_text(aes(label = val4), size = 4, nudge_x = -.35, na.rm = TRUE) +
          geom_text(aes(label = val5), size = 10) +
          geom_tile(fill = 'transparent', colour = 'black') +
          ggtitle(paste("Q-table after ", iterations, " iterations\n",
                        "(epsilon = ", epsilon, ", alpha = ", alpha, ", gamma = ", gamma, ", beta = ", beta, ")")) +
          theme(plot.title = element_text(hjust = 0.5)) +
          scale_x_continuous(breaks = c(1:W), labels = c(1:W)) +
          scale_y_continuous(breaks = c(1:H), labels = c(1:H)))
}

GreedyPolicy <- function(x, y){
  scores <- q_table[x, y, ]
  best_actions <- which(scores == max(scores))
  return(best_actions[sample.int(length(best_actions), 1)])
}

EpsilonGreedyPolicy <- function(x, y, epsilon){
  if(runif(1) < epsilon){
    return(sample.int(4, 1))
  } else {
    return(GreedyPolicy(x, y))
  }
}

transition_model <- function(x, y, action, beta){
  delta <- sample(-1:1, size = 1, prob = c(0.5*beta, 1-beta, 0.5*beta))
  final_action <- ((action + delta + 3) %% 4) + 1
  foo <- c(x, y) + unlist(action_deltas[final_action])
  foo <- pmax(c(1,1), pmin(foo, c(H, W)))
  return(foo)
}

# Updated q_learning function with a 'freeze' parameter
q_learning <- function(start_state, epsilon = 0.5, alpha = 0.1, gamma = 0.95,
                       beta = 0, freeze = FALSE){
  x <- start_state[1]
  y <- start_state[2]
  episode_correction <- 0
  
  repeat{
    action <- EpsilonGreedyPolicy(x, y, epsilon)
    new_state <- transition_model(x, y, action, beta)
    reward <- reward_map[new_state[1], new_state[2]]
    
    if(reward != 0){
      target <- reward
    } else {
      target <- reward + gamma * max(q_table[new_state[1], new_state[2], ])
    }
    
    # Update Q-table ONLY if not frozen
    if(!freeze){
      correction <- target - q_table[x, y, action]
      q_table[x, y, action] <<- q_table[x, y, action] + alpha * correction
      episode_correction <- episode_correction + correction
    }
    
    if(reward != 0)
      return(c(reward, episode_correction))
    
    x <- new_state[1]
    y <- new_state[2]
  }
}

# Environment Setup
H <- 5
W <- 7
reward_map <- matrix(0, nrow = H, ncol = W)
reward_map[3,6] <- 10
reward_map[2:4,3] <- -1

# Hyperparameters grid
epsilons <- c(0.1, 0.2, 0.3)
gammas <- c(0.8, 0.9, 0.95)

# Create a data frame to store results for all 9 pairs
results_grid <- expand.grid(epsilon = epsilons, gamma = gammas)
results_grid$avg_reward <- NA

set.seed(12345)

best_avg_reward <- -Inf
best_pair <- NULL

# Grid search over 9 pairs
for(idx in 1:nrow(results_grid)){
  curr_eps <- results_grid$epsilon[idx]
  curr_gamma <- results_grid$gamma[idx]
  
  cat(sprintf("\n--- Training Pair %d: Epsilon = %.1f, Gamma = %.2f ---\n", idx, curr_eps, curr_gamma))
  
  # 1. Initialize fresh Q-table for each pair
  q_table <- array(0, dim = c(H, W, 4))
  
  # 2. Training Phase: 30,000 episodes (freeze = FALSE)
  for(i in 1:30000){
    q_learning(start_state = c(3, 1), epsilon = curr_eps, gamma = curr_gamma, freeze = FALSE)
  }
  
  # 3. Evaluation Phase: 1,000 episodes with FROZEN table (freeze = TRUE)
  # Using epsilon = 0 (pure greedy policy) or curr_eps during evaluation
  eval_rewards <- numeric(1000)
  for(j in 1:1000){
    res <- q_learning(start_state = c(3, 1), epsilon = 0, gamma = curr_gamma, freeze = TRUE)
    eval_rewards[j] <- res[1]
  }
  
  # 4. Calculate Average Reward
  avg_reward <- mean(eval_rewards)
  results_grid$avg_reward[idx] <- avg_reward
  cat(sprintf("Average Reward over 1000 evaluation episodes: %.4f\n", avg_reward))
  
  # Track the best pair
  if(avg_reward > best_avg_reward){
    best_avg_reward <- avg_reward
    best_pair <- c(epsilon = curr_eps, gamma = curr_gamma)
  }
}

# Final Results Summary
cat("\n========================================\n")
cat("          FINAL RESULTS SUMMARY         \n")
cat("========================================\n")
print(results_grid)

cat("\n BEST HYPERPARAMETER COMBINATION:\n")
cat("Epsilon:", best_pair["epsilon"], "\n")
cat("Gamma:", best_pair["gamma"], "\n")
cat("Highest Average Reward:", best_avg_reward, "\n")
#converting base hmm code to situation where the robot must stay atleast for a few timestates in a certain state. EXAM 23
# =====================================================================
# HMM LAB 2 - BASE CODE (mera lab code, jaisa hai waisa; plots hata diye)
# Exam mein isi ko copy karke modify karna hai.
# =====================================================================

# ---------- Q1: HMM BANANA (yahi sab se zyada badalta hai) ----------
library(HMM)

n_sectors <- 5   # number of sectors on the circle
max_error <- 1    # the device can be off by up to 2 sectors

by_sectors = [1,1,2,2,2,3,3,4,5]

#states  <- as.character(1:n_sectors)   # true sectors (hidden)
#i will have to hardcode states, because in this case the states hongi that aik state mai kitnay timestep rukna hai.
#for example agar aik state mai dou dafa rukna hai atleast toh mai iss state ky liye dou states banao ga.

states = as.character("st1c2, st1c1, st2c3, st2c2, st2c1, ...")
symbols <- as.character(1:n_sectors)   # device answers (observed) #i think yeh same hee rahy ga.

# Transition matrix A: row = sector now, column = sector next
trans <- matrix(c(),nrow = 10, ncol = 10, byrow = TRUE)#mujhy isko bharnay se pehle paper pe banana hoga and then i will put it here.
#yahan pe nrow and ncol maine jou matrix banaya tha physically wahan ki hee dimensions hain, and by row true karna lazmi hai.


# Emission matrix B: row = true sector, column = what the device says
emit5 <- matrix(0, n_sectors, n_sectors)
for (i in 1:n_sectors) {
  for (delta in -max_error:max_error) {   # -2, -1, 0, 1, 2
    emit5[i, ((i - 1 + delta) %% n_sectors) + 1] <- 1 / (2 * max_error + 1)
  }
}

emit = emit5[by_sectors] #yeh basically emission matrix jou bana hai uss ko kahy ga that tumne jou pehli row banai hai
#woh mujhy dou 2 times, and jou second row banai hai woh dou 3 dafa and so on. as a result aik matrix banay gi. woh matrix humari
#emission matrix hogi.

start_probs <- rep(1 / n_sectors, n_sectors)   # same chance to start anywhere

hmm <- initHMM(States = states, Symbols = symbols, startProbs = start_probs,
               transProbs = trans, emissionProbs = emit)

print(hmm)
rowSums(trans)   # every row should add up to 1
rowSums(emit)    # every row should add up to 1

# ---------- Q2: SIMULATE ----------
set.seed(12345)                  # fixed seed, so we always get the same sample
sim <- simHMM(hmm, 100)

true_states  <- sim$states       # where the robot really was (hidden, used only for checking)
observations <- sim$observation  # what the device said (the only thing we are allowed to use)

print(true_states)
print(observations)

# How often did the device say exactly the right sector?
mean(true_states == observations)

# ---------- Q3: FILTERING, SMOOTHING, VITERBI ----------
# Filtering: forward() gives alpha on the log scale.
# exp() undoes the log, and prop.table(..., 2) makes every column add up to 1.
alpha    <- exp(forward(hmm, observations))
filtered <- prop.table(alpha, 2)          # column t = p(Z^t | readings 1 to t)

# Smoothing: posterior() runs forward-backward and already makes columns add up to 1
smoothed <- posterior(hmm, observations)  # column t = p(Z^t | all 100 readings)

# Most probable path: Viterbi
viterbi_path <- viterbi(hmm, observations)

# Check: every column should add up to 1
range(colSums(filtered))
range(colSums(smoothed))

# Look at a few time steps (columns 1, 2, 50 and 100)
round(filtered[, c(1, 2, 50, 100)], 3)
round(smoothed[, c(1, 2, 50, 100)], 3)
print(viterbi_path)

# ---------- Q4: ACCURACY ----------
# The guess at each step is the most likely sector.
# which.max gives a row number (1-10); states[...] turns it back into a label like "3".
filtered_guess <- states[apply(filtered, 2, which.max)]
smoothed_guess <- states[apply(smoothed, 2, which.max)]

accuracy <- c(filtered = mean(filtered_guess == true_states),
              smoothed = mean(smoothed_guess == true_states),
              viterbi  = mean(viterbi_path   == true_states))
print(accuracy)

# ---------- Q7: PREDICT STEP 101 ----------
pred_101 <- filtered[, 100] %*% trans   # our belief at step 100, pushed one step forward
round(pred_101, 3)
sum(pred_101)                           # should be 1
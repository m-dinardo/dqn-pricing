source('src/process.R')

# Aggregate raw data 
boscun_agg <- boscun_init()

# States
states <- boscun_agg %>% select(-c(next_min_price, daily_return))

# PCA on 48 state variables (ended up not using this, didn't perform as well)
states_pc <- states %>% select(-c(trip, received_date))
states_pc <- prcomp(states_pc, center = TRUE, scale.= TRUE)
cumvar <- cumsum(states_pc$sdev^2 / sum(states_pc$sdev^2))
pc.index<-min(which(cumvar>0.99)) # Number of PCs for 95% cumulative variance
plot(cumvar, type='l')

# Scaled State representations
states_scaled <- data.frame(scale(states %>% select(-c(trip, received_date))))
states_scaled <- cbind(states[,1:2], states_scaled)
#states_scaled$day_of_week <- weekdays(as.Date(states_scaled$received_date))
#states_scaled <- cbind(states_scaled, data.frame(model.matrix(~day_of_week - 1, data=states_scaled)))
#states_scaled$day_of_week <- NULL
# PC- reduced state representations
states_r <- data.frame(states_pc$x[,1:pc.index]) # 18 PCs for 95% cumu. variance
states_r <- cbind(states[,1:2], states_r)

# Compute reward table, daily return on HOLD, add indicator for terminal reward (end of episode)
rewards <- boscun_agg %>% select(trip, received_date, daily_return)
rewards$terminal <- boscun_agg$next_min_price == 0 # Terminal State indicator

# Divide into train/test set for states and rewards
set.seed(123)
train <- sample(unique(rewards$trip), 0.8*length(unique(rewards$trip)),replace=FALSE)
states_train <- states_scaled %>% filter(trip %in% train)
states_test <- states_scaled %>% filter(!(trip %in% train))
rewards_train <- rewards %>% filter(trip %in% train)
rewards_test <- rewards %>% filter(!(trip %in% train))

write.csv(states_train, 'output/states_train.csv', row.names=FALSE)
write.csv(states_test, 'output/states_test.csv', row.names=FALSE)
write.csv(rewards_train, 'output/rewards_train.csv', row.names=FALSE)
write.csv(rewards_test, 'output/rewards_test.csv', row.names=FALSE)

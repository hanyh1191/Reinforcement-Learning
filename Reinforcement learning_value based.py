'''value-based reinforcement learning'''
'''warning:
#          incomplete code
#
'''


'''<Q-learning>'''
#<1>action choose policy for observed state
if np.random.uniform() < self.epsilon:
    # choose best action
    state_action = self.q_table[observation, :]
    # some actions may have the same value, randomly choose on in these actions
    action = np.random.choice(state_action[state_action == np.max(state_action)].index)
else:
    # choose random action
    action = np.random.choice(self.actions)
#<2>update Q value
if s_ != 'terminal':
    q_target = r + self.gamma * self.q_table[s_, :].max()  # next state is not terminal
else:
    q_target = r  # next state is terminal
self.q_table[s, a] += self.lr * (q_target - q_predict)  # update


'''<sarsa>'''
#<1>Choose action for next-state using policy
# RL take action and get next observation and reward
observation_, reward = env.step(observation, action)
# RL choose action based on next observation
action_ = RL.choose_action(str(observation_))
# swap observation and action
observation = observation_
action = action_
#<2>update Q value
if s_ != 'terminal':
    q_target = r + self.gamma * self.q_table.[s_, a_]  # next state is not terminal
else:
    q_target = r  # next state is terminal
self.q_table.[s, a] += self.lr * (q_target - q_predict)  # update


'''<sarsa-lambda>'''
#<1>Update E(s,a) at current step (s,a)   
# method-1:
self.eligibility_trace[s, a] += 1
# method-2:
self.eligibility_trace[s, :] *= 0
self.eligibility_trace[s, a] = 1
#<2>Update Q value and E value for all (s,a) by table-E
if s_ != 'terminal':
    q_target = r + self.gamma * self.q_table[s_, a_]  # next state is not terminal
else:
    q_target = r  # next state is terminal
error = q_target - q_predict
#Update table-Q
self.q_table += self.lr * error * self.eligibility_trace
#Update table-E
#the <lambda_> is decay rate of E value
self.eligibility_trace *= self.gamma*self.lambda_


'''<DQN>'''
#<1>Build network
#   Through network we can get action by observed state, 
#   and then from environment get next state and reward,
#   this is [s, a, r, s_].
#   besides, we should use predict network to get the Q(s,a) value,
#   and use target network to get the Q-target value.
#   and then perform a gradient descent.
#<2>choose action from network or random,use greedy  
if np.random.uniform() < self.epsilon:
    actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
    action = np.argmax(actions_value)
else:
    action = np.random.randint(0, self.n_actions) 
#<3>put [s, a, r, s_] into memory
transition = np.hstack((s, [a, r], s_))
index = self.memory_counter % self.memory_size
self.memory[index, :] = transition 
self.memory_counter += 1
#<4>get a batch from memory and use gradient descent 
#   to update predict-network parameters
if self.memory_counter > self.memory_size:
    sample_index = np.random.choice(self.memory_size, size=self.batch_size)
else:
    sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
batch_memory = self.memory[sample_index, :]
#   get q_next (from target_net) and q_eval(from predict_net)
q_next, q_eval = self.sess.run(
    [self.q_next, self.q_eval],
    feed_dict={
        self.s_: batch_memory[:, -self.n_features:],
        self.s: batch_memory[:, :self.n_features]
    })
q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
with tf.variable_scope('loss'): # error
    self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
with tf.variable_scope('train'):    # gradient descent
    self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

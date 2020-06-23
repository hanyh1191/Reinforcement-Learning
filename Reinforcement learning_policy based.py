'''policy-based reinforcement learning'''
'''warning:
#          incomplete code
#
'''


'''<Policy Gradient>'''
#<1>Build network
#   build a policy network used to calculate the probability of each action
#   and use the episode (s,a,r) to update network parameters by gradient descent
#<2>choose action
def choose_action(self, observation):
    prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
    action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
    return action
#<3>store (s,a,r)
def store_transition(self, s, a, r):
    self.ep_obs.append(s)
    self.ep_as.append(a)
    self.ep_rs.append(r)
#<4>update policy network by gradient descent
with tf.name_scope('loss'):
    # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
    # or in this way:
    #neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   
    # this is negative log of chosen action
    neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
    loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss
with tf.name_scope('train'):
    self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)


'''<Actor-Critic>'''
#<1>Build network
#   Build actor network like PG, and then build critic network
#   actor network is used to choose action, critic network is used to critic network
#<2>critic action and update critic network parameters
with tf.variable_scope('squared_TD_error'):
    self.td_error = self.r + GAMMA * self.v_ - self.v
    self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
with tf.variable_scope('train'):
    self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)


'''<DDPG>'''
#<1>critic network update method
with tf.variable_scope('Critic'):                                                                         
    # assign self.a = a in memory when calculating q for td_error,                                        
    # otherwise the self.a is from Actor when updating Actor                                              
    q = self._build_c(self.S, self.a, scope='eval', trainable=True)                                       
    q_ = self._build_c(self.S_, a_, scope='target', trainable=False) 
q_target = self.R + GAMMA * q_                                                                            
# in the feed_dic for the td_error, the self.a should change to actions in memory                         
td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)                                   
self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)                    
#<2>actor network update method
with tf.variable_scope('Actor'):                                                                          
    self.a = self._build_a(self.S, scope='eval', trainable=True)                                          
    a_ = self._build_a(self.S_, scope='target', trainable=False) 
a_loss = - tf.reduce_mean(q)    # maximize the q                                                          
self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)         
#<3>target network update method
# target net replacement                                                                                  
self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)                                                
                    for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]                
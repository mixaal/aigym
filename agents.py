import numpy as np
import random
import time


class RandomAgent(object):
    def __init__(self):
        pass

    def learn_how_to_play(self, env):
        pass

    def get_state(self, env):
        return env.action_space.sample()


class MarkovDecision(object):
    def __init__(self):
        self.total_episodes = 2000
        self.learning_rate = 0.8
        self.max_steps = 2000
        self.gamma = 0.95

        self.epsilon = 1.0  # exploration rate
        self.max_epsilon = 1.0  # exploration probability at start
        self.min_epsilon = 0.01  # min exploration probability
        self.decay_rate = 0.005  # exponential decay rate for exploration probability
        self.initialized = False
        self.game_state = 0

    def initialize(self, env):
        n_actions = env.action_space.n
        #n_states = env.observation_space.n
        n_states = 1 + self.max_steps

        self._Q = np.zeros((n_states, n_actions))
        print("############# Q #############")
        print(self._Q)
        print("#############################")

        self.initialized = True

    def learn_how_to_play(self, env):
        self.initialize(env)
        rewards = []


        # 2 For life or until learning is stopped
        for episode in range(self.total_episodes):
            time.sleep(1) # do not overheat cpu ;-)
            print("Episode: {}\r".format(episode))
            # Reset the environment
            state_no = -1
            state = env.reset()
            step = 0
            done = False
            total_rewards = 0

            for step in range(self.max_steps):
                state_no += 1
                # 3. Choose an action a in the current world state (s)
                ## First we randomize a number
                exp_exp_tradeoff = random.uniform(0, 1)
                state = state_no
                new_state = state + 1

                ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
                if exp_exp_tradeoff > self.epsilon:
                    action = np.argmax(self._Q[state, :])

                # Else doing a random choice --> exploration
                else:
                    action = env.action_space.sample()

                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, done, info = env.step(action)
                state = state_no
                new_state = state + 1



                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                # qtable[new_state,:] : all the actions we can take from new state
                #print state
                #print action
                #print reward
                self._Q[state, action] = self._Q[state, action] + self.learning_rate * (
                            reward + self.gamma * np.max(self._Q[new_state, :]) - self._Q[state, action])

                total_rewards += reward

                # Our new state is state
                state = new_state

                # If done (if we're dead) : finish episode
                if done:
                    break

            # Reduce epsilon (because we need less and less exploration)
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)
            rewards.append(total_rewards)

        print ("Score over time: " + str(sum(rewards) / self.total_episodes))
        print(self._Q)
        self._Q.dump("riverraid.model")

    def get_state(self, env):
        action = np.argmax(self._Q[self.game_state, :])
        if self.game_state >= self.max_steps:
            print "Trained for max_steps exceeded"
        else:
            self.game_state += 1
        return action

        # new_state, reward, done, info = env.step(action)
        #
        # if done:
        #     # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
        #     env.render()
        #
        #     # We print the number of step it took.
        #     print("Number of steps", step)
        #     break
        # state = new_state


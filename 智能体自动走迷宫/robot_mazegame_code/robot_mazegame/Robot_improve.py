import random

class Robot(object):
#update_freq目标网络的更新频率
#高频率更新（较小的update_freq）：可以使得目标网络更快地跟上主网络学习到的新知识，适用于环境变化较快或者需要快速学习的情况。但过于频繁的更新可能会引入额外的波动，影响学习的稳定性。
#低频率更新（较大的update_freq）：可以提供更稳定的学习目标，减少训练过程中的波动，适合于解决较为复杂的任务或当环境相对稳定时。然而，更新太慢可能使学习过程变得迟缓，无法及时利用新信息。
    def __init__(self, maze, alpha=0.55, gamma=0.85, epsilon0=0.6,update_freq=70):

        self.maze = maze
        self.valid_actions = self.maze.valid_actions
        self.state = None
        self.action = None

        # Set Parameters of the Learning Robot
        self.alpha = alpha
        self.gamma = gamma
        # epsilon0是用来控制探索性行为的参数
        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.update_freq = update_freq
        self.t = 0
        # 存储Q值，Q 值是对每个状态-动作对的估计值，表示在特定状态下执行特定动作所能获得的长期奖励的预期值
        self.Qtable = {}
        #不直接使用主网络的Q值来计算目标Q值，因为这可能导致过度波动和不稳定的学习。使用一个独立的、较旧的Q值估计（即目标网络的Q值）来计算目标Q值
        self.target_Qtable = {}
        self.reset()

    def reset(self):
        """
        Reset the robot
        """
        self.state = self.sense_state()
        self.create_Qtable_line(self.state)
        self.update_target_Qtable()

    def update_target_Qtable(self):
        """偶尔更新目标Q表为当前Q表的副本，减少计算量同时避免振荡"""
        self.target_Qtable = self.Qtable.copy()

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        """
        if self.testing:
            # No random choice when testing
            self.epsilon = 0.0
        else:
            # Update parameters when learning
            self.t += 1
            self.epsilon = self.epsilon0 / self.t
        return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot.
        """
        return self.maze.sense_robot()

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        """
        self.Qtable.setdefault(state, {a: 0.0 for a in self.valid_actions})
        self.target_Qtable.setdefault(state, {a: 0.0 for a in self.valid_actions})

    def choose_action(self):
        """
        Return an action according to given rules
        """

        def is_random_exploration():
            return random.random() < self.epsilon

        if self.learning:
            if is_random_exploration():
                return random.choice(self.valid_actions)
            else:
                # Use main network to choose action but target network to evaluate
                best_action = max(self.Qtable[self.state], key=self.Qtable[self.state].get)
                target_qvalue = self.target_Qtable[self.state][best_action]
                return best_action
        elif self.testing:
            return max(self.Qtable[self.state], key=self.Qtable[self.state].get)
        else:
            return random.choice(self.valid_actions)

    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        """
        if self.learning:
            # Double DQN update
            best_next_action = max(self.Qtable[next_state], key=self.Qtable[next_state].get)
            q_next_max = self.target_Qtable[next_state][best_next_action]

            self.Qtable[self.state][action] += self.alpha * (r + self.gamma * q_next_max - self.Qtable[self.state][action])

    def update(self):
        """
        Describle the procedure what to do when update the robot.
        Called every time in every epoch in training or testing.
        Return current action and reward.
        """
        self.state = self.sense_state()  # Get the current state
        self.create_Qtable_line(self.state)  # For the state, create q table line

        action = self.choose_action()  # choose action for this state
        reward = self.maze.move_robot(action)  # move robot for given action

        next_state = self.sense_state()  # get next state
        self.create_Qtable_line(next_state)  # create q table line for next state

        if self.learning and not self.testing:
            self.update_Qtable(reward, action, next_state)  # update q table
            self.update_parameter()  # update parameters

            # Periodically update the target Q-table
            if self.t % self.update_freq == 0:
                self.update_target_Qtable()

        return action, reward
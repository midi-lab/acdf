# simple gridworld environment akin to
# figure 2 in Tom Schaul's UVFA paper.

import numpy as np




class GridWorld(object):
    def __init__(self, rows, cols, start=(1, 1), goal=(6, 6),
                 state_config='light'):

        self.rows = rows
        self.cols = cols
        self.size = self.rows # square grid of size (e.g rows x cols)
        self.model = np.ones((self.size, self.size), dtype='int32')
        self.action_shape = (4,)
        self.state_config = state_config
        if state_config == 'light':
            self.state_shape = (self.size, self.size, 1)
        else:
            self.state_shape = (self.size, self.size, 3)
        self.start = np.asarray(start)
        self.goal = np.asarray(goal)
        self.agent_position = self.start
        self.current_state = self._build_state(self.start)
        self._build()
        self.goal_reward = 5


        self.show()

    def get_admissible_positions(self):
        def _compute():
            for i in range(self.size):
                for j in range(self.size):
                    yield np.asarray((i, j))
        return [pos for pos in _compute()
                if self._is_pos_admissible(pos)]

    def _build_state(self, agent_position):
        current_sta = np.array(self.model)
        current_goa = np.array(self.model) * 0.
        current_pos = np.array(self.model) * 0.
        current_pos[agent_position[0], agent_position[1]] = 1
        current_goa[self.goal[0], self.goal[1]] = 1
        if self.state_config == 'light':
            state_ = np.expand_dims(current_pos, 2)
        else:
            state_ = np.stack([current_sta, current_goa, current_pos], axis=2)
        return state_

    def _build(self):
        # leave one pixel for the border
        self.model[0, :] = 0
        self.model[:, 0] = 0
        self.model[-1, :] = 0
        self.model[:, -1] = 0
        # one pixel for the interior separations
        middle = int(self.size / 2)
        quarter = int(self.size / 4)
        self.model[middle, :] = 0
        self.model[:, middle] = 0
        # doors
        self.model[middle, quarter] = 1
        self.model[quarter, middle] = 1
        self.model[middle, -quarter - 1] = 1
        self.model[-quarter - 1, middle] = 1

    def img(self):
        model_ = np.array(self.model)
        model_[self.agent_position[0], self.agent_position[1]] = 9
        model_[self.goal[0], self.goal[1]] = 8

        return model_

    def show(self):
        model_ = np.array(self.model)
        model_[self.agent_position[0], self.agent_position[1]] = 9
        model_[self.goal[0], self.goal[1]] = 8
        print(model_)

    def step(self, action, reverse=False):
        assert (self.agent_position is not None)
        next_pos = self._execute_action(action, self.agent_position,
                                        reverse=reverse)
        is_admissible = self._is_pos_admissible(next_pos)
        terminal = self._has_reached_goal(next_pos)
        reward = self.goal_reward if terminal else -0.1 * (1 - is_admissible)
        # update agent position if it is admissible
        if is_admissible:
            self.agent_position = next_pos
        return self._build_state(self.agent_position), reward, terminal


    def reached_terminal(self, next_pos):
        terminal = self._has_reached_goal(next_pos)
        return terminal        


    def get_action_mask(self):
        return None

    def state_to_position(self, state):
        assert state.ndim == 3
        pos_slice = state[:, :, -1]
        pos = np.where(pos_slice == 1)
        assert len(pos[0]) == 1
        assert len(pos[1]) == 1
        return np.asarray((pos[0][0], pos[1][0]))

    def _has_reached_goal(self, position):
        return np.all(position == self.goal)

    def _is_pos_admissible(self, position):
        return np.all(position < self.size) \
            and np.all(position > 0) \
            and self.model[position[0], position[1]]

    def _execute_action(self, action, position, reverse=False):
        # n, e, s, w
        if reverse:
            action = (action + 2) % 4
        if action == 0:
            new_pos = np.asarray((position[0] - 1, position[1]))
        elif action == 1:
            new_pos = np.asarray((position[0], position[1] + 1))
        elif action == 2:
            new_pos = np.asarray((position[0] + 1, position[1]))
        elif action == 3:
            new_pos = np.asarray((position[0], position[1] - 1))
        else:
            raise Exception()
        return new_pos

    def reset(self, start=None):
        if start is None:
            self.agent_position = self.start
        else:
            self.agent_position = start
        return self._build_state(self.agent_position)

if __name__ == '__main__':
    g = GridWorld(9)
    while True:
        action = input('Type action:')
        print(g.step(int(action)))
        g.show()



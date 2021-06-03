# %%

from typing import List, Tuple
import numpy as np
from itertools import product

# %%


class TicTacToe():

    def __init__(self):
        """initialise the board"""
        self.board_shape = (3, 3)
        self.objective_value = 15
        # initialise state as an array
        self.state = [np.nan for _ in range(self.board_shape[0]*self.board_shape[1])]  # initializes the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)]  # , can initialise to an array or matrix
        self.Win = 'Win'
        self.Tie = 'Tie'
        self.Resume = 'Resume'
        self.reset()

    def is_winning(self, current_state: List[int]) -> bool:
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        board_state = np.array(current_state).reshape(self.board_shape)
        # Calculate Column, Row, diagonal and antidiagonal  wise Sum and check if 15 is there
        # With OR condition it will be checked sequentially as if multiple "if-elif" condition
        return \
            self.objective_value in board_state.sum(axis=0) or \
            self.objective_value in board_state.sum(axis=1) or \
            self.objective_value == np.trace(board_state) or \
            self.objective_value == np.fliplr(board_state).trace()

    def is_terminal(self, current_state: List[int]) -> Tuple[bool, str]:
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(current_state) == True:
            return True, self.Win

        elif len(self.allowed_positions(current_state)) == 0:
            return True, self.Tie

        else:
            return False, self.Resume

    def allowed_positions(self, current_state: List[int]) -> List[int]:
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(current_state) if np.isnan(val)]

    def allowed_values(self, current_state: List[int]) -> Tuple[List[int], List[int]]:
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in current_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 != 0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 == 0]

        return (agent_values, env_values)

    def action_space(self, current_state: List[int]) -> Tuple[List[int], List[int]]:
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(current_state), self.allowed_values(current_state)[0])
        env_actions = product(self.allowed_positions(current_state), self.allowed_values(current_state)[1])
        return (agent_actions, env_actions)

    def state_transition(self, current_state: List[int], current_action: Tuple[int, int]) -> List[int]:
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        new_state = current_state.copy()
        new_state[current_action[0]] = current_action[1]
        return new_state

    def step(self, current_state: List[int], current_action: Tuple[int, int]) -> Tuple[List[int], int, bool]:
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        board_state = None
        reward = None
        #  Agent turn
        # Check if current action is valid for current  state
        assert current_action in self.action_space(current_state)[0]
        # Put a number on board
        board_state = self.state_transition(current_state=current_state, current_action=current_action)

        # Check if game is won
        is_terminal_state, game_state = self.is_terminal(board_state)
        if is_terminal_state:
            if game_state == self.Win:
                reward = 10
            elif game_state == self.Tie:
                reward = 0

        # Env turn
        else:
            # Select a random action by environment
            valid_env_actions = [a for a in self.action_space(board_state)[1]]
            random_action_index = np.random.choice(list(range(len(valid_env_actions))))
            agent_action = valid_env_actions[random_action_index]
            # Put a number on board
            board_state = self.state_transition(current_state=board_state, current_action=agent_action)

            # Check if game is won
            is_terminal_state, game_state = self.is_terminal(board_state)
            if is_terminal_state:
                if game_state == self.Win:
                    reward = -10
                elif game_state == self.Tie:
                    reward = 0
            else:
                reward = -1

        return board_state, reward, is_terminal_state

    def reset(self):
        return self.state

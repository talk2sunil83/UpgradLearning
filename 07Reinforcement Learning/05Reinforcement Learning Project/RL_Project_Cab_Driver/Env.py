'''
# RL Project (Cab-Driver)

# Import routines
'''
import numpy as np
import random
# from itertools import permutations
from dataclasses import dataclass
from typing import List, Tuple
from math import ceil
# Defining hyperparameters
m = 5  # number of cities, ranges from 1 ..... m
t = 24  # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5  # Per hour fuel and other costs
R = 9  # per hour revenue from a passenger

'''
# Helper classes for states and actions
'''


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class Action:
    """Action data class, with pickup and drop
    """
    pickup: int
    drop: int


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class StateA1:
    """Sate class without action could be used as vanils state object
    """
    '''
    def __init__(self, loc: int = 0, time: int = 0, day: int = 0) -> None:
        """Sate class without action could be used as vanils state object

        Args:
            loc (int, optional): location [0,5]. Defaults to 0.
            time (int, optional): hour of the day, zero based [0,23]. Defaults to 0.
            day (int, optional): day number of the week, zero based [0,6]. Defaults to 0.

        Raises:
            ValueError: location (loc) is None or out of range
            ValueError: hour of the day (time) is None or out of range
            ValueError: day of the week (day) is None or out of range
        """
        if loc is None or loc < 0 or loc > (m - 1):
            raise ValueError(f"location (loc) is blank or inappropriate value, allowed [0,{m-1}]")
        if time is None or time < 0 or time > (t - 1):
            raise ValueError(f"hour of the day (time)is blank or inappropriate value, allowed [0,{t-1}]")
        if day is None or loc < 0 or loc > (d - 1):
            raise ValueError(f"day of the week (day) is blank or inappropriate value, allowed [0,{d-1}]")
        self._loc: int = loc
        self._time: int = time
        self._day: int = day
    '''

    def __init__(self, state: List[int]) -> None:
        if state is None or len(state) < 3:
            raise ValueError("State List (state) is blank or less values, minimum 3")
        if state[0] is None or state[0] < 0 or state[0] > (m - 1):
            raise ValueError(f"location (state[0]) is blank or inappropriate value, allowed [0,{m-1}]")
        if state[1] is None or state[1] < 0 or state[1] > (t - 1):
            raise ValueError(f"hour of the day (state[1])is blank or inappropriate value, allowed [0,{t-1}]")
        if state[2] is None or state[2] < 0 or state[2] > (d - 1):
            raise ValueError(f"day of the week (state[2]) is blank or inappropriate value, allowed [0,{d-1}]")
        self._loc: int = state[0]
        self._time: int = state[1]
        self._day: int = state[2]

    def encoded(self) -> List[np.int16]:
        state_encoded = np.zeros(m + t + d, dtype=np.int16)
        state_encoded[self._loc] = 1
        state_encoded[m + self._time] = 1
        state_encoded[m + t + self._day] = 1
        return state_encoded


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class StateA2(StateA1):
    """State with Action

    Args:
        StateA1 ([StateA1]): StateA1 class for inheritance
    """

    def __init__(self, action: List[int], state: List[int]) -> None:
        """Initiate StateA2 object from Action and StateA1 list values

        Args:
            action (List[int]): action values as integers
            state (List[int]): state values as integers

        Raises:
            ValueError: State List (state) is None or less values
            ValueError: location (state[0]) is None or inappropriate value
            ValueError: hour of the day (state[1])is None or inappropriate value
            ValueError: day of the week (state[2]) is None or inappropriate value
            ValueError: action is None or inappropriate value
        """
        if state is None or len(state) < 3:
            raise ValueError(f"State List (state) is None or less values, minimum 3")
        if state[0] is None or state[0] < 0 or state[0] > (m - 1):
            raise ValueError(f"location (state[0]) is None or inappropriate value, allowed [0,{m-1}]")
        if state[1] is None or state[1] < 0 or state[1] > (t - 1):
            raise ValueError(f"hour of the day (state[1])is None or inappropriate value, allowed [0,{t-1}]")
        if state[2] is None or state[2] < 0 or state[2] > (d - 1):
            raise ValueError(f"day of the week (state[2]) is None or inappropriate value, allowed [0,{d-1}]")
        if action is None or action[0] < 0 or action[1] < 0 or action[0] > (m - 1) or action[1] > (m - 1):
            raise ValueError(f"action is None or inappropriate value, allowed [0,{m-1}] for pickup/action[0] and drop/action[1]")
        super(StateA2, self).__init__(state)
        # self._loc: int = state[0]
        # self._time: int = state[1]
        # self._day: int = state[2]
        self._action = Action(action[0], action[1])

    '''
        # def __init__(self, action: Action, loc: int = 0, time: int = 0, day: int = 0) -> None:
        #     """Initiate StateA2 object from Action and StateA1 Properties

        #     Args:
        #         action (Action): object of Action class
        #         loc (int, optional): location [0,5]. Defaults to 0.
        #         time (int, optional): hour of the day, zero based [0,23]. Defaults to 0.
        #         day (int, optional): day number of the week, zero based [0,6]. Defaults to 0.

        #     Raises:
        #         ValueError: if action object is None or inappropriate value
        #     """
        #     super(StateA2, self).__init__(loc, time, day)
        #     if action is None or action.pickup < 0 or action.drop < 0 or action.pickup > (m - 1) or action.drop > (m - 1):
        #         raise ValueError(f"action is None or inappropriate value, allowed [0,{m-1}] for pickup and drop")
        #     self._action = action
    '''

    @property
    def location(self):
        return self._loc

    @property
    def time(self):
        return self._time

    @property
    def day(self):
        return self._day

    @property
    def pickup(self):
        return self._action.pickup

    @property
    def drop(self):
        return self._action.drop

    def encoded(self) -> List[np.int16]:
        state_encoded = np.zeros(m + t + d + m + m, dtype=np.int16)
        state_encoded[self._loc] = 1
        state_encoded[m + self._time] = 1
        state_encoded[m + t + self._day] = 1

        if self._action is not None:
            if (self._action.pickup != 0):
                state_encoded[m + t + d + self._action.pickup] = 1
            if (self._action.drop != 0):
                state_encoded[m + t + d + m + self._action.drop] = 1
        return state_encoded


class CabDriver():

    def __init__(self) -> None:
        """initialise your state and define your action space and state space"""
        # the cities pairs to travel
        self.action_space = [(p, q) for p in range(m) for q in range(m) if p != q or p == 0]  # [(0, 0)] + list(permutations(range(m), 2))
        # 𝑠= 𝑋𝑖𝑇𝑗𝐷𝑘 𝑤ℎ𝑒𝑟𝑒 𝑖= 0 ... 𝑚−1; 𝑗= 0 ... . 𝑡− 1; 𝑘= 0 ... . . 𝑑− 1
        self.state_space = [[x, y, z] for x in range(m) for y in range(t) for z in range(d)]
        self.state_init = random.choice(self.state_space)
        # print(f"{self.action_space=}\n{self.state_space=}\n{self.state_init=}")
        # Start the first round
        self.reset()

    def reset(self):
        return self.action_space, self.state_space, self.state_init
    # Encoding state (or state-action) for NN input

    def state_encoded_arch1(self, state: List[int]):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        return StateA1(state).encoded()

    # Use this function if you are using architecture-2

    def state_encoded_arch2(self, state: List[int], action: List[int]):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
        return StateA2(action=action, state=state).encoded()

    # Getting number of requests

    def requests(self, state: List[int]):
        """Determining the number of requests basis the location.
        Use the table specified in the MDP and complete for rest of the locations"""
        location_poisson_distribution = [2, 12, 4, 7, 8]
        requests = np.random.poisson(location_poisson_distribution[state[0]])
        # print(f"requests-chosen requests1 : {requests}")
        while requests == 0:
            requests = np.random.poisson(location_poisson_distribution[state[0]])
            # print(f"requests-chosen requests2 : {requests}")

        if requests > 15:
            requests = 15

        possible_actions_index = random.sample(range(1, (m - 1) * m + 1), requests)   # (0,0) is not considered as customer request
        if len(possible_actions_index) <= 0:
            possible_actions_index = random.sample(range(1, (m - 1) * m + 1), requests)
        # print(f"requests-chosen possible_actions_index : {possible_actions_index}")
        actions = [self.action_space[i] for i in possible_actions_index]

        actions.append([0, 0])
        # if len(possible_actions_index) <= 0:
        #   print(f"requests-state : {state}\nrequests-requests: {requests}\nrequests-actions: {actions}")

        return possible_actions_index, actions

    def get_time_day(self, time: int, day: int, ride_duration: int) -> Tuple[int, int]:
        """
        Takes in the current state and time taken for driver's journey to return
        the state post that journey.
        """
        # if time is None or time < 0 or time > (t - 1):
        #     raise ValueError(f"hour of the day (time)is blank or inappropriate value, allowed [0,{t-1}]")
        # if day is None or day < 0 or day > (d - 1):
        #     raise ValueError(f"day of the week (day) is blank or inappropriate value, allowed [0,{d-1}]")
        #  time between 0 to t
        if (time + ride_duration) < t:
            time += ride_duration
        else:
            time = (time + ride_duration) % t
        # days between 0 to d
        num_days = (time + ride_duration) // t
        # week days between 0 to 6
        day = (day + num_days) % d

        return int(ceil(time)), int(ceil(day))

    def reward_func(self, wait_time: int, transit_time: int, ride_time: int):
        """Takes in state, action and Time-matrix and returns the reward"""
        idle_time = wait_time + transit_time
        reward = (R * ride_time) - (C * (ride_time + idle_time))

        return reward

    def next_state_func(self, state: List[int], action: List[int], time_matrix: np.ndarray):
        """Takes state and action as input and returns next state"""
        next_state = []
        # Initialize various times
        total_time, transit_time, wait_time, ride_time = 0, 0, 0, 0

        # Derive the current location, time, day and request locations
        state_obj = StateA2(action, state)
        current_loc = state_obj.location
        pickup_loc = state_obj.pickup
        drop_loc = state_obj.drop
        current_time = state_obj.time
        current_day = state_obj.day
        """
         3 Scenarios:
           1) Already at pick up point
           2) Need to travel to pickup point.
           3) Went off duty or At starting point in the begging of month
        """
        # debug(f"{current_loc=} {pickup_loc=} {drop_loc=} {current_time=} {current_day=}")
        # 1) Already at pick up point
        if pickup_loc == current_loc:
            ride_time = time_matrix[current_loc][drop_loc][current_time][current_day]
            next_loc = drop_loc
        # 2) Need to travel to pickup point.
        elif pickup_loc != current_loc:

            transit_time = time_matrix[current_loc][pickup_loc][current_time][current_day]
            time, day = self.get_time_day(current_time, current_day, transit_time)
            # debug(f"{time=} {day=}")
            ride_time = time_matrix[pickup_loc][drop_loc][time][day]
            next_loc = drop_loc
        # 3) Went off duty or At starting point in the begging of month
        else:  # pickup_loc == drop_loc == 0:
            wait_time = 1
            next_loc = current_loc

        total_time = (wait_time + transit_time + ride_time)
        # debug(f"{wait_time=} {transit_time=} {ride_time=} {total_time=}")
        time, day = self.get_time_day(current_time, current_day, total_time)
        # debug(f"{time=} {day=}")
        next_state = [next_loc, time, day]
        return next_state, wait_time, transit_time, ride_time

    def step(self, state, action, Time_matrix):
        """
        Take a trip as cabby to get rewards next step and total time spent
        """
        next_state, wait_time, transit_time, ride_time = self.next_state_func(state, action, Time_matrix)

        # Calculate the reward based on the different time durations
        rewards = self.reward_func(wait_time, transit_time, ride_time)
        total_time = wait_time + transit_time + ride_time

        return rewards, next_state, total_time

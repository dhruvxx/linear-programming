#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cvxpy as cp
import json

# PARAMETERS
STEPCOST = -5
PENALTY = -40


all_actions = ['UP', 'LEFT', 'DOWN', 'RIGHT', 'STAY',
               'SHOOT', 'HIT', 'CRAFT', 'GATHER', 'NONE']
mm_state = ['D', 'R']


def change_tuple(state, **kwargs):
    p, m, a, s, h = state
    p = kwargs.get('pos', p)
    m = kwargs.get('mat', m)
    a = kwargs.get('arrow', a)
    s = kwargs.get('mm', s)
    h = kwargs.get('health', h)
    return (p, m, a, s, h)


class Position:

    def __init__(self, name, acts, prob1, prob2, prob3=0):
        self.name = name
        self.move_prob = prob1  # up, down, left, right, stay
        self.spec_prob = prob2  # hit, craft, gather
        self.shoot_prob = prob3  # shoot
        self.actions = list(np.array(all_actions)[acts])

    def up(self):
        i = positions.index(self)
        return positions[i+1]

    def down(self):
        i = positions.index(self)
        return positions[i-1]

    def right(self):
        i = positions.index(self)
        return positions[i+2]

    def left(self):
        i = positions.index(self)
        return positions[i-2]

    def default(self):
        return positions[-1]

    def move(self, action=''):
        if action == 'STAY':
            return self
        func = {'UP': self.up, 'DOWN': self.down,
                'RIGHT': self.right, 'LEFT': self.left}
        act = func.get(action, self.default)
        return act()


center = Position('C', [0, 1, 2, 3, 4, 5, 6], 0.85, 0.1, 0.5)
north = Position('N', [2, 4, 7], 0.85, [0.5, 0.35, 0.15])
south = Position('S', [0, 4, 8], 0.85, 0.75)
east = Position('E', [1, 4, 5, 6], 1, 0.2, 0.9)
west = Position('W', [3, 4, 5], 1, 0, 0.25)

positions = [west, south, center, north, east]


class State:

    def __init__(self, pos, mat, arrow, mm, health):
        self.tuple = (pos, mat, arrow, mm, health)
        self.pos = pos
        self.mat = mat
        self.arrow = arrow
        self.mm = mm
        self.health = health
        self.name = '({},{},{},{},{})'.format(
            pos.name, mat, arrow, mm_state[mm], health)

    def set_index(self, idx):
        if idx >= 0 and idx < 600:
            self._index = idx
            return idx
        return -1

    def get_index(self):
        try:
            return self._index
        except:
            return -1

    def get_actions(self):
        if self.health == 0:
            return ['NONE']
        actions = [a for a in self.pos.actions]
        if 'SHOOT' in actions and self.arrow == 0:
            actions.remove('SHOOT')
        if 'CRAFT' in actions and self.mat == 0:
            actions.remove('CRAFT')
        return actions

    def get_tuple(self, **kwargs):
        pos = kwargs.get('pos', self.pos)
        mat = kwargs.get('mat', self.mat)
        arrow = kwargs.get('arrow', self.arrow)
        mm = kwargs.get('mm', self.mm)
        health = kwargs.get('health', self.health)
        return (pos, mat, arrow, mm, health)

    def next_states(self, action):
        tuples = []
        probs = []

        if action in ['UP', 'LEFT', 'DOWN', 'RIGHT', 'STAY']:
            p1 = self.pos.move_prob
            tuples = [self.get_tuple(pos=self.pos.move(
                action)), self.get_tuple(pos=self.pos.move())]
            probs = [p1, 1-p1]
        elif action == 'SHOOT':
            p3 = self.pos.shoot_prob
            tuples = [self.get_tuple(
                arrow=self.arrow - 1, health=self.health-25), self.get_tuple(arrow=self.arrow - 1)]
            probs = [p3, 1-p3]
        elif action == 'CRAFT':
            a2 = min(3, self.arrow + 2)
            a1 = min(3, self.arrow + 1)
            tuples = [self.get_tuple(mat=self.mat-1, arrow=a1), self.get_tuple(
                mat=self.mat-1, arrow=a2), self.get_tuple(mat=self.mat-1, arrow=3)]
            probs = self.pos.spec_prob
        elif action == 'HIT':
            p2 = self.pos.spec_prob
            fh = max(0, self.health-50)
            tuples = [self.get_tuple(health=fh), self.tuple]
            probs = [p2, 1-p2]
        elif action == 'GATHER':
            p2 = self.pos.spec_prob
            fm = min(2, self.mat + 1)
            tuples = [self.get_tuple(mat=fm), self.tuple]
            probs = [p2, 1-p2]
        else:  # 'NONE'
            return [], []

        if self.mm:  # ready
            if self.pos in [center, east]:
                attack = self.get_tuple(mm=0, arrow=0)
                newprobs = [0.5*p for p in probs] + [0.5]
                newtuples = [s for s in tuples] + [attack]
            else:
                attack = [change_tuple(s, mm=0) for s in tuples]
                newprobs = [0.5*p for p in probs] + [0.5*p for p in probs]
                newtuples = [s for s in tuples] + attack
        else:  # dormant
            newprobs = [0.8*p for p in probs] + [0.2*p for p in probs]
            newtuples = tuples + [change_tuple(s, mm=1) for s in tuples]

        newstates = [get_stateobj(s) for s in newtuples]
        newprobs = [round(p, 6) for p in newprobs]
        return newstates, newprobs

    def reward(self, finalState):
        if self.mm == 1 and finalState.mm == 0:
            if self.pos in [center, east]:
                return STEPCOST + PENALTY
        return STEPCOST


def initialise_states():
    states = []
    i = 0

    for pos in positions:
        for mat in range(0, 3):
            for arrow in range(0, 4):
                for mm in range(0, 2):
                    for health in range(0, 5):
                        newstate = State(pos, mat, arrow, mm, health*25)
                        states.append(newstate)
                        newstate.set_index(i)
                        i += 1

    return states


def get_stateobj(stuple):
    global states
    for s in states:
        if s.tuple == stuple:
            return s


def state_index(stuple):
    s = get_stateobj(stuple)
    return s.get_index()


def initialise_pairs():
    global states
    pairs = []
    for state in states:
        actions = state.get_actions()
        for act in actions:
            pairs.append([state, act])
    return pairs


def pair_index(state, action='ANY'):
    for i, c in enumerate(stateaction):
        if c[0] == state:
            if action == 'ANY':
                return i
            if c[1] == action:
                return i


def get_alpha(startState=None):
    if startState:
        alpha = [0.]*600
        start = state_index(startState)
        alpha[start] = 1.
    else:
        alpha = [1/600]*600
    return np.reshape(alpha, (600, 1))


def get_Amatrix(arr, pairs):

    a = np.zeros((len(arr), len(pairs)))

    for i, pair in enumerate(pairs):
        startState, action = pair
        newStates, probs = startState.next_states(action)
        x = startState.get_index()

        for j, state in enumerate(newStates):
            y = state.get_index()
            a[x][i] += probs[j]
            a[y][i] -= probs[j]

        if not newStates:
            a[x][i] = 1

    return a


def get_Rmatrix(pairs):

    r = np.zeros(len(pairs))

    for i, pair in enumerate(pairs):
        startState, action = pair
        newStates, probs = startState.next_states(action)
        expected_reward = 0
        for j, state in enumerate(newStates):
            expected_reward += probs[j]*startState.reward(state)
        r[i] = round(expected_reward, 6)

    return r


def get_policy(arr, pairs, X):
    policy = []
    for state in arr:
        i = pair_index(state)
        n = len(state.get_actions())
        sl = X.value[i:i+n].tolist()
        u_max = np.max(sl)
        idx = i + sl.index(u_max)
        action = pairs[idx][1]
        policy.append([state.name, action])
    return policy


if __name__ == '__main__':

    states = initialise_states()  # State(object)
    stateaction = initialise_pairs()  # [State(object), string]
    n = len(stateaction)

    alpha = get_alpha()
    A = get_Amatrix(states, stateaction)
    R = get_Rmatrix(stateaction)
    x = cp.Variable(shape=(n, 1), name="x")

    constraints = [cp.matmul(A, x) == alpha, x >= 0]
    objective = cp.Maximize(cp.matmul(R, x))
    problem = cp.Problem(objective, constraints)

    solution = problem.solve()

    policy = get_policy(states, stateaction, x)

    X_list = [list(x) for x in x.value]
    A_list = [list(a) for a in A]

    final_dict = {"a": A_list, "r": R.tolist(), "alpha": alpha.tolist(
    ), "x": X_list, "policy": policy, "objective": solution}

    with open('outputs/part_3_output.json', 'w') as fd:
        json.dump(final_dict, fd)

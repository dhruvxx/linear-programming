# Report: Assignment 2 Part 3

Team 59
| 2019115008 | Dhruvee Birla |
| -------------- | -------------------- |
| **2019113013** | **Pahulpreet Singh** |

## Making the A matrix

We are using the lists `states` and `stateaction` to make the A matrix. Here the list `states` contains all 600 states (all possible combinations of positions, materials, arrows, MM state, MM health) and the list `stateaction` contains all possible _(state, action)_ pairs.

- To generate `states`, we iterate through all possible values of the 5 variables, make a **State** object with each tuple, and append it to the list. We also assign an index to each **State** object so that it's easier to access them later from the list. It has a total of 600 elements.

- To generate `stateaction`, we go through every element in `states` and check all the valid actions possible at that state based on health, position, arrows and materials. And then we append the list `[state, action]` _(**State** object, string)_ to our final list. It has a total of 1936 elements.

Now that we have the lists `states` and `stateaction`, we can generate the `A` matrix with the help of these lists.

- We declare a numpy array with rows equal to the number of `states` and the columns equal to the number of `stateaction` pairs. The shape of `A` is hence (600, 1936). And it is initialised to all zeros.

- Now corresponding to each _(state, action)_ pairs, we add/subtract the probabilities of state transitions representing out-flow and in-flow. That is, if a state transition for a given action has probability **p** given that action, the we subtract **p** from `A[y][i]` and add it to `A[x][i]`. If the only action possible from the state is _NONE_, i.e. there is no final state, we just add 1 to `A[x][i]`. We repeat this for all state-action pairs.

Here `x` and `y` are the indices (in `states`) of the starting state and final state respectively, and `i` is the index (in `stateaction`) of the state-action pair. Thus the indexing convention used in the `A` matrix is equivalent to **[state_index][pair_index]**.

Thus for each state-action pair, the values corresponding to each of the terminal states is a negative number greater than -1. The value for the starting state is a positive number less than (in case there is a self loop) or equal to 1. All other values are 0, as these states are not related to the given state-action pair.

Here is the code for generating the `A` matrix:

```python
def get_Amatrix(states, stateaction):

      a = np.zeros((len(states), len(stateaction)))

      for i, pair in enumerate(stateaction):
            startState, action = pair
            newStates, probs = startState.next_states(action)
            x = startState.get_index()

            for j, state in enumerate(newStates):
                  y = state.get_index()
                  a[x][i] += probs[j]
                  a[y][i] -= probs[j]

            # when action is 'NONE' there are no final states
            if not newStates:
                  a[x][i] = 1

      return a
```

## Finding the Policy

- The policy represents a sequence of _(state, action)_ pairs which tells us which is the most optimal action to be followed from a given state.
- We need matrices `A`, `R`, `alpha` and a variable `x` to find the policy. `A` matrix has been explained in detail in the previous section. Other matrices are explained in brief in the following points.
- `R` matrix contains the expected reward for each valid _(state, action)_ pair.
- `alpha` matrix contains the probabilities of each state being the start state. If a start state is specified, the value corresponding to that state is 1 and all other values are 0. If no start state is specified, i.e. all states are equally possible to be the start state, then it is an array filled with 1/N (here N = 600). If any other constraints are specified, `alpha` can be set accordingly. For this assignment, the second case is applicable.
- `x` is a variable which is obtained by solving the LPP. After solving, it will represent the utilities of all valid _(state, action)_ pairs.

Now that we understand what these matrices are, we can calculate `x` and eventually find the policy.

- We obtain `x` by solving the following Linear Programming problem:

```
maximize    R*x
subject to  A*x = alpha
            x >= 0
```

here A\*B represent matrix multiplication of A, B.
We solve this problem by using the `cvxpy` library.

- Now that we have the utilities of all valid _(state, action)_ pairs, we can iterate through all of them and check the maximum utility for each state. The action that gives this maximum utility is the most optimal action for that state.
- The sequence of _(state, action)_ pairs with the most optimal action for each state will give us the policy.

## The Case of Multiple Policies

Depending on various parameters, the generated policy can be different. Since the question doesn't clarify what changes are allowed and what aren't, we'll be discussing and commenting on multiple cases:

- One way is to obviously change the set of rules that IJ and MM are following in the process. For example, changing the probabilities of outcomes by different actions, or the probabilities of ready-dormant state transitions. This will change `A` and possibly `R`. This will change `x` and we can have a new policy.
- Another way is to change the rewards/step-cost/penalty. of state transitions. This will not change `A` but it will change `R` so the solution to `x` will be different, and hence policy will change.

Note that the above two points change the "game" altogether. If we want to have a different policy without changing the gameplay, there are still two methods.

- We can specify a start state (or a set of possible starting states) instead of all the states being equally probable. This will change `alpha`, and since the constraints of the LPP change, it will give a different solution to `x`, and hence, a different policy, even though both `A` and `R` remain unchanged.

- If we want a new policy without any change in the logic whatsoever, we can change the order of the actions in the list they are specified in. This will give us a different action for the policy in case two or more actions are giving the same utility, since the order in the lists will change. Note that the chance of multiple actions having the maximum utility for a given state is low, but it is definitely possible. This will change the order of values in `R`, and the order of columns in `A`. Hence the order of values in `x` will also be different then before.

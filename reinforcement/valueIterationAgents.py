# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            states = self.mdp.getStates()
            counter = self.values.copy()
            for state in states:
                qValuelist = []
                for action in self.mdp.getPossibleActions(state):
                    Qvalue = self.computeQValueFromValues(state, action)
                    qValuelist.append(Qvalue)
                    counter[state] = max(qValuelist)
            self.values = counter

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        statesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        qvalue = 0
        for nextState, prob in statesAndProbs:
            q = prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])
            qvalue += q
        return qvalue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actionValue = util.Counter()
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            actionValue[action] = self.computeQValueFromValues(state, action)
        bestAction = actionValue.argMax()
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(self.iterations):
            state = states[i % len(states)]
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                qValuelist = []
                for action in actions:
                    qValuelist.append(self.computeQValueFromValues(state, action))
                bestValue = max(qValuelist)
                self.values[state] = bestValue


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def computePredecessors(self):
        predecessors = {}
        states = self.mdp.getStates()
        for state in states:
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                statesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
                for nextState, prob in statesAndProbs:
                    if nextState not in predecessors:
                        predecessors[nextState] = {state}
                    predecessors[nextState].add(state)
        return predecessors

    def runValueIteration(self):
        pQueue = util.PriorityQueue()
        states = self.mdp.getStates()
        predecessors = self.computePredecessors()
        for state in states:
            if not self.mdp.isTerminal(state):
                qValuelist = []
                for action in self.mdp.getPossibleActions(state):
                    qValue = self.computeQValueFromValues(state, action)
                    qValuelist.append(qValue)
                bestValue = max(qValuelist)
                diff = abs(self.values[state] - bestValue)
                pQueue.update(state, - diff)
        i = 0
        while i < self.iterations and (not pQueue.isEmpty()):
            i += 1
            temp = pQueue.pop()
            qValueslist = []
            actions = self.mdp.getPossibleActions(temp)
            for action in actions:
                qValue = self.computeQValueFromValues(temp, action)
                qValueslist.append(qValue)
            bestValue = max(qValueslist)
            self.values[temp] = bestValue

            for p in predecessors[temp]:
                qValuelist = []
                for action in self.mdp.getPossibleActions(p):
                    qValue = self.computeQValueFromValues(p, action)
                    qValuelist.append(qValue)
                bestValue = max(qValuelist)
                diff = abs(self.values[p] - bestValue)
                if diff > self.theta:
                    pQueue.update(p, -diff)

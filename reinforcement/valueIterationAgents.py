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
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        prob = self.mdp
        states = prob.getStates()

        for i in range(self.iterations):
            holderValues = util.Counter()
            for state in states:
                newVal = []
                actions = prob.getPossibleActions(state)
                if not actions:
                    holderValues[state] = self.values[state]
                else:
                    for action in actions:
                        newVal.append(self.computeQValueFromValues(state, action))

                    holderValues[state] = max(newVal)
            self.values = holderValues


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
        prob = self.mdp
        possibleStates = prob.getTransitionStatesAndProbs(state, action)
        qVal = 0
        for nextState in possibleStates:
            qVal = qVal + nextState[1]*(prob.getReward(state, action, nextState[0]) +
                                        self.discount*self.getValue(nextState[0]))
        return qVal

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestVal = float('-inf')
        chosenAction = None
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            newScore = self.computeQValueFromValues(state, action)
            if newScore > bestVal:
                chosenAction = action
                bestVal = newScore
        return chosenAction


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
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
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
        prob = self.mdp
        states = prob.getStates()
        divider = len(states)
        for i in range(self.iterations):
            state = states[i % divider]
            newVal = []
            actions = prob.getPossibleActions(state)
            if actions and (not prob.isTerminal(state)):
                for action in actions:
                    newVal.append(self.computeQValueFromValues(state, action))
                self.values[state] = max(newVal)





class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        prob = self.mdp
        predecessors = self.getPredecessors(prob)
        from util import PriorityQueue
        heap = PriorityQueue()
        states = prob.getStates()
        for state in states:
            if not prob.isTerminal(state):
                actions = prob.getPossibleActions(state)
                value = []

                for action in actions:

                    value.append(self.computeQValueFromValues(state, action))

                diff = abs(self.values[state] - max(value))

                heap.push(state, -1 * diff)
        for i in range(self.iterations):
            if heap.isEmpty():
                break
            s = heap.pop()
            if not prob.isTerminal(s):
                self.bellmanUpdate(s)
                if predecessors[s]:
                    for pred in predecessors[s]:
                        predactions = prob.getPossibleActions(pred)
                        value = []

                        for predaction in predactions:
                            value.append(self.computeQValueFromValues(pred, predaction))

                        preddiff = abs(self.values[pred] - max(value))

                        if preddiff > self.theta:
                            heap.push(pred, -1 * preddiff)


    def bellmanUpdate(self, state):
        actions = self.mdp.getPossibleActions(state)
        val = float('-inf')
        if not actions:
            return
        else:
            for action in actions:
                newval = self.computeQValueFromValues(state, action)
                if newval > val:
                    val = newval
            self.values[state] = val



    def getPredecessors(self, problem):
        dict = {}
        states = problem.getStates()
        for state in states:
            dict[state] = []
        for state in states:
            actions = problem.getPossibleActions(state)
            reachable = set()
            for action in actions:
                reachableStates = problem.getTransitionStatesAndProbs(state, action)
                for nextstate in reachableStates:
                    if nextstate[1] > 0:
                        reachable.add(nextstate[0])
            for newstate in states:
                if newstate in reachable:
                    dict[newstate].append(state)
        return dict

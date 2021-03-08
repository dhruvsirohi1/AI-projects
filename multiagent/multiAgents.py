# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        foodPos = newFood.asList()
        score = -10*len(foodPos)
        from util import manhattanDistance
        minDist = float('inf')
        # maxDist = float('inf')
        xcom = ycom = 0
        for x in foodPos:
            dist = manhattanDistance(x, newPos)
            xcom = xcom + x[0]
            ycom = ycom + x[1]
            if dist < minDist or minDist == -1:
                minDist = dist
        score = score + (1/(10*minDist))
        # foodcom = (xcom/(len(foodPos)+1), ycom/(len(foodPos) + 1))
        # score = score - 0.25*max(minDist,manhattanDistance(newPos, foodcom))
        for pos in newGhostStates:
            if pos.getPosition() == newPos:
                return float('-inf')

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()
        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.chosenAction = None

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        evaluation = self.minimax(gameState, 0, 0)
        return self.chosenAction


    def minimax(self, state, depth, index):
        if state.isWin() or depth == self.depth or state.isLose():
            return self.evaluationFunction(state)

        if index == 0:
            maxeval = float('-inf')
            for action in state.getLegalActions(0):
                eval = self.minimax(state.generateSuccessor(0, action), depth, index + 1)
                if eval > maxeval:
                    maxeval = eval
                    if depth == 0:
                        self.chosenAction = action
            return maxeval
        else:
            mineval = float('inf')
            numghosts = state.getNumAgents() - 1
            for action in state.getLegalActions(index):
                if index == numghosts:
                    eval = self.minimax(state.generateSuccessor(index, action), depth + 1, 0)
                else:
                    eval = self.minimax(state.generateSuccessor(index, action), depth, index + 1)
                mineval = min(mineval, eval)
            return mineval


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        pruning = self.alphabeta(gameState, 0, 0, float('-inf'), float('inf'))
        return self.chosenAction

    def alphabeta(self, state, depth, index, alpha, beta):
        if state.isWin() or depth == self.depth or state.isLose():
            return self.evaluationFunction(state)

        if index == 0:
            maxeval = float('-inf')
            for action in state.getLegalActions(0):
                eval = self.alphabeta(state.generateSuccessor(0, action), depth, index + 1, alpha, beta)
                if eval > maxeval:
                    maxeval = eval
                    if depth == 0:
                        self.chosenAction = action
                alpha = max(alpha, eval)
                if beta < alpha:
                    break
            return maxeval
        else:
            mineval = float('inf')
            numghosts = state.getNumAgents() - 1
            for action in state.getLegalActions(index):
                if index == numghosts:
                    eval = self.alphabeta(state.generateSuccessor(index, action), depth + 1, 0, alpha, beta)
                else:
                    eval = self.alphabeta(state.generateSuccessor(index, action), depth, index + 1, alpha, beta)
                mineval = min(mineval, eval)
                beta = min(eval, beta)
                if beta < alpha:
                    break
            return mineval

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        evaluation = self.expectimax(gameState, 0, 0)
        return self.chosenAction

    def expectimax(self, state, depth, index):
        if state.isWin() or depth == self.depth or state.isLose():
            return self.evaluationFunction(state)

        if index == 0:
            maxeval = float('-inf')
            for action in state.getLegalActions(0):
                eval = self.expectimax(state.generateSuccessor(0, action), depth, index + 1)
                if eval > maxeval:
                    maxeval = eval
                    if depth == 0:
                        self.chosenAction = action
            return maxeval
        else:
            count = 0
            minscore = 0
            mineval = float('inf')
            numghosts = state.getNumAgents() - 1
            for action in state.getLegalActions(index):
                count = count + 1
                if index == numghosts:
                    eval = self.expectimax(state.generateSuccessor(index, action), depth + 1, 0)
                else:
                    eval = self.expectimax(state.generateSuccessor(index, action), depth, index + 1)
                minscore = minscore + eval
            mineval = min(mineval, minscore / count)
            return mineval

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    ghostpos = successorGameState.getGhostPositions()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    foodPos = newFood.asList()
    pellets = successorGameState.getCapsules()
    score = -10 * (len(foodPos) + len(pellets))
    from util import manhattanDistance
    pim = [0]
    # if len(pellets) >= 1:
    #     for x in pellets:
    #         pim.append(manhattanDistance(newPos, x))
    # score = score - 50*max(pim)
    # if newPos in foodPos:
    #     score = score + 1000

    minDist = float('inf')
    # maxDist = float('inf')
    xcom = ycom = 0

    if sum(newScaredTimes) > 0:
        score = score + 50 * sum(newScaredTimes)
        poss = []
        for posn in ghostpos:
            poss.append(manhattanDistance(newPos, posn))
        score = score - 50*max(poss)
    for x in foodPos:
        dist = manhattanDistance(x, newPos)
        xcom = xcom + x[0]
        ycom = ycom + x[1]
        if dist < minDist or minDist == -1:
            minDist = dist
    score = score + 100 / (minDist)
    foodcom = (xcom/(len(foodPos)+1), ycom/(len(foodPos) + 1))
    score = score + 10 * currentGameState.getScore()
    # print(foodcom)
    score = score - 0.75 * manhattanDistance(newPos, foodcom)
    # score = 1 * currentGameState.getScore() - 1.5 * minDist - 2
    for pos in newGhostStates:
        if pos.getPosition() == newPos:
            return float('-inf')
    return score

# Abbreviation
better = betterEvaluationFunction

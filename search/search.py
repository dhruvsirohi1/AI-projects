# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

class Cell:
    def __init__(self, location, path, parent, cost, fullInfo = None):
        self.position = location
        self.path = path
        self.parent = parent
        self.cost = cost
        self.fullInfo = fullInfo

    def getState(self):
        return self.fullInfo

    def getCell(self):
        return self.position

    def getPath(self):
        return self.path

    def getParent(self):
        return self.parent

    def getCost(self):
        return self.cost


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    startState = problem.getStartState()
    parent = startState
    from util import Stack
    children = problem.getSuccessors(startState)
    successors = Stack()
    visited = list()
    visited.append(startState)
    for child in children:
        holder = list()
        holder.append(child[1])
        nextChild = Cell(child[0], holder, parent, child[2], child)
        successors.push(nextChild)
    while successors.isEmpty() == False:
        thisparent = successors.pop()
        visited.append(thisparent.getCell())
        if problem.isGoalState(thisparent.getCell()):
            return thisparent.getPath()
        newchildren = problem.getSuccessors(thisparent.getCell())
        for child in newchildren:
            if child[0] not in visited:
                updatePath = thisparent.getPath().copy()
                updatePath.append(child[1])
                nextCell = Cell(child[0], updatePath, thisparent, child[2], child)
                successors.push(nextCell)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    start = problem.getStartState()
    visited = [start]
    children = problem.getSuccessors(start)
    succ = Queue()
    for x in children:
        newchild = Cell(x[0], [x[1]], start, x[2], x)
        succ.push(newchild)
    while not succ.isEmpty():
        nxt = succ.pop()
        if nxt.getCell() not in visited:
            if problem.isGoalState(nxt.getCell()):
                return nxt.getPath()
            newchildren = problem.getSuccessors(nxt.getCell())
            visited.append(nxt.getCell())
            for x in newchildren:
                if x[0] not in visited:
                    newPath = nxt.getPath().copy()
                    newPath.append(x[1])
                    Child = Cell(x[0], newPath, nxt, x[2], x)
                    succ.push(Child)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    startState = problem.getStartState()
    parent = startState
    from util import PriorityQueue
    children = problem.getSuccessors(startState)
    successors = PriorityQueue()
    visited = list()
    visited.append(startState)
    cost = 0
    for child in children:
        holder = list()
        holder.append(child[1])
        nextChild = Cell(child[0], holder, parent, cost + child[2], child)
        successors.push(nextChild, cost + child[2])
    while successors.isEmpty() == False:
        thisparent = successors.pop()
        if thisparent.getCell() not in visited:
            visited.append(thisparent.getCell())
            if problem.isGoalState(thisparent.getCell()):
                return thisparent.getPath()
            newchildren = problem.getSuccessors(thisparent.getCell())
            for child in newchildren:
                if child[0] not in visited:
                    updatePath = thisparent.getPath().copy()
                    updatePath.append(child[1])
                    nextCell = Cell(child[0], updatePath, thisparent, thisparent.getCost() + child[2], child)
                    successors.push(nextCell, nextCell.getCost() + child[2])


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    startState = problem.getStartState()
    parent = startState
    from util import PriorityQueue
    children = problem.getSuccessors(startState)
    successors = PriorityQueue()
    visited = list()
    visited.append(parent)
    cost = 0
    for child in children:
        holder = list()
        holder.append(child[1])
        nextChild = Cell(child[0], holder, parent, cost + child[2])
        successors.push(nextChild, cost + child[2] + heuristic(child[0], problem))
    while not successors.isEmpty():
        thisparent = successors.pop()
        if thisparent.getCell() not in visited:
            visited.append(thisparent.getCell())
            if problem.isGoalState(thisparent.getCell()):
                return thisparent.getPath()
            newchildren = problem.getSuccessors(thisparent.getCell())
            for child in newchildren:
                if child[0] not in visited:
                    updatePath = thisparent.getPath().copy()
                    updatePath.append(child[1])
                    nextCell = Cell(child[0], updatePath, thisparent, thisparent.getCost() + child[2])
                    successors.push(nextCell, nextCell.getCost() + heuristic(child[0], problem))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

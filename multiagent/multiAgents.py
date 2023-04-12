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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        "*** YOUR CODE HERE ***"
        newFoodList = newFood.asList()
        foodDistance = -1
        for food in newFoodList:
            distance = util.manhattanDistance(newPos, food)
            if distance <= foodDistance or foodDistance == -1:
                foodDistance = distance
        foodScore = 1.0 / foodDistance
        ghostDistance = 999999999
        for ghost in successorGameState.getGhostPositions():
            distance = util.manhattanDistance(newPos, ghost)
            if ghostDistance < distance:
                ghostDistance = util.manhattanDistance(newPos, ghost)
        ghostScore = -(1.0 / ghostDistance)
        totalScareTime = sum(newScaredTimes)
        return successorGameState.getScore() + foodScore + ghostScore + totalScareTime


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


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

        def minimax(agent, depth, curGameState):
            if curGameState.isLose() or curGameState.isWin() or depth == self.depth:
                return self.evaluationFunction(curGameState)
            if agent == 0:
                return max(minimax(1, depth, curGameState.generateSuccessor(agent, action)) for action in curGameState.getLegalActions(agent))
            else:
                nextAgent = agent + 1
                if curGameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                    depth += 1
                return min(minimax(nextAgent, depth, curGameState.generateSuccessor(agent, action)) for action in curGameState.getLegalActions(agent))

        temp = -1
        for agentAction in gameState.getLegalActions(0):
            bestMove = minimax(1, 0, gameState.generateSuccessor(0, agentAction))
            if bestMove > temp or temp == -1:
                temp = bestMove
                action = agentAction

        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphabeta(agent, depth, curGameState, a, b):
            if curGameState.isLose() or curGameState.isWin() or depth == self.depth:
                return self.evaluationFunction(curGameState)
            if agent == 0:
                v = float("-inf")
                for action in curGameState.getLegalActions(agent):
                    v = max(v, alphabeta(1, depth, curGameState.generateSuccessor(agent, action), a, b))
                    if v > b:
                        return v
                    a = max(a, v)
                return v
            else:
                v = float("inf")
                next_agent = agent + 1
                if curGameState.getNumAgents() == next_agent:
                    next_agent = 0
                    depth += 1
                for action in curGameState.getLegalActions(agent):
                    v = min(v, alphabeta(next_agent, depth, curGameState.generateSuccessor(agent, action), a, b))
                    if v < a:
                        return v
                    b = min(b, v)
                return v

        Max = -1
        alpha = float("-inf")
        beta = float("inf")
        for agentAction in gameState.getLegalActions(0):
            temp = alphabeta(1, 0, gameState.generateSuccessor(0, agentAction), alpha, beta)
            if temp > Max or Max == -1:
                Max = temp
                action = agentAction
            if Max > beta:
                return Max
            alpha = max(alpha, Max)
        return action


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

        def expectimax(curGameState, agent, depth):
            if curGameState.isLose() or curGameState.isWin() or depth == self.depth:
                return self.evaluationFunction(curGameState)
            if agent == 0:
                return max(expectimax(curGameState.generateSuccessor(agent, action), 1, depth) for action in curGameState.getLegalActions(agent))
            else:
                nextAgent = agent + 1
                if curGameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                    depth += 1
                return sum(expectimax(curGameState.generateSuccessor(agent, action), nextAgent, depth) for action in curGameState.getLegalActions(agent)) / len(curGameState.getLegalActions(agent))

        maximum = -1
        for agentState in gameState.getLegalActions(0):
            temp = expectimax(gameState.generateSuccessor(0, agentState), 1, 0)
            if temp > maximum or maximum == -1:
                maximum = temp
                action = agentState

        return action



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    I simply copied what I did in Part 1, what I did in part 1 is I calculate the distance from the
    furthest food, and get the according score by dividing it by 1, also get the closest ghost's distance
    and dividing it with negative 1 to get ghost's score, and their scare times, adding them all up with the
    current score.
    """

    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    newFoodList = newFood.asList()
    foodDistance = -1
    for food in newFoodList:
        distance = util.manhattanDistance(newPos, food)
        if distance <= foodDistance or foodDistance == -1:
            foodDistance = distance
    foodScore = 1.0 / foodDistance
    ghostDistance = 999999999
    for ghost in currentGameState.getGhostPositions():
        distance = util.manhattanDistance(newPos, ghost)
        if ghostDistance < distance:
            ghostDistance = util.manhattanDistance(newPos, ghost)
    ghostScore = -(1.0 / ghostDistance)
    totalScareTime = sum(newScaredTimes)
    return currentGameState.getScore() + foodScore + ghostScore + totalScareTime


# Abbreviation
better = betterEvaluationFunction

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
import time

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



def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    start=problem.getStartState()
    stack=[(start,[])]
    visited=[]
    while stack:
    	(vertex,direction)=stack.pop()
    	location=[]
    	l2d={}

    	if problem.isGoalState(vertex):
    		return direction

    	if vertex not in visited:
    		visited.append(vertex)
	    	for element in problem.getSuccessors(vertex):
	    		location.append(element[0])
	    		l2d[element[0]]=element[1]

			for next in (location):
				stack.append((next,direction+[l2d[next]]))
    #print len(visited), direction
        #return path
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    start=problem.getStartState()
    stack=[(start,[])]
    visited=[]
    while stack:
    	(vertex,direction)=stack.pop(0)
    	location=[]
    	l2d={}

    	if problem.isGoalState(vertex):
    		return direction

    	if vertex not in visited:
    		visited.append(vertex)
	    	for element in problem.getSuccessors(vertex):
	    		location.append(element[0])
	    		l2d[element[0]]=element[1]

			for next in (location):
				stack.append((next,direction+[l2d[next]]))
    #print len(visited), direction
    
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    start=problem.getStartState()
    stack=[(start,[start],[])]
    cost = [0]
    while stack:
    	index=cost.index(min(cost))
    	costCur=cost[index]
    	(vertex,path,direction)=stack.pop(index)

    	location=[]
    	l2d={}

    	for element in problem.getSuccessors(vertex):
    		location.append(element[0])
    		l2d[element[0]]=element[1]
    	for next in set(location)-set(path):
    		if problem.isGoalState(next):
    			path=path+[next]
    			direction=direction+[l2d[next]]
    			stack={}
    		else:
    			costDir=problem.getCostOfActions([l2d[next]])+costCur
    			cost.append(costDir)
    			stack.append((next,path+[next],direction+[l2d[next]]))
    print path,direction
    
    return direction
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
    

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    start=problem.getStartState()
    stack=[(start,[])]
    cost = [0]
    costHurestic = [0]
    visited=[]
    while stack:
    	index=costHurestic.index(min(costHurestic))
    	(vertex,direction)=stack.pop(index)

    	costHuresticCur=costHurestic.pop(index)
        costCur = cost.pop(index)
    	location=[]
    	l2d={}

    	if problem.isGoalState(vertex):
    		print direction
    		return direction

    	if vertex not in visited:
    		visited.append(vertex)
	    	for element in problem.getSuccessors(vertex):
	    		location.append(element[0])
	    		l2d[element[0]]=element[1]

			for next in (location):
				costDir=1+costCur+heuristic(next,problem)
    			#problem.getCostOfActions([l2d[next]])
    			#print "hurestic",heuristic(next,problem)
    			costHurestic.append(costDir)
    			cost.append(1+costCur)
    			stack.append((next,direction+[l2d[next]]))
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

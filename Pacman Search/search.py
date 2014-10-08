# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import string
from Queue import Queue
from game import Agent
"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from test.test_sax import start

from util import Stack
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
    # frontier will contain those node which needs to be explored.
    # Order will be Last In First Out
    frontier = util.Stack() 
    
    explored = list() # explored List will contain list of nodes which are already explored.
    startState = problem.getStartState()
    node_with_direction_list = [startState]
    frontier.push(node_with_direction_list) # frontier is list of list (Direction to reach a Node and a Node)
    # Example frontier = list(['North','South','B'],['East','West','C'])
    # Here frontier contains the list of list of (Direction to reach B along with node B) and
    # (Direction to reach C along with node C).
    
    while frontier.isEmpty() != True:
        
        node_with_direction_list = frontier.pop()
        currentNode = node_with_direction_list.pop()
        # After pop operation, nodeWithDirectionList will contain only path to reach currentNode
        explored.append(currentNode)

        if problem.isGoalState(currentNode) == True:
            return node_with_direction_list
        
        for successor in problem.getSuccessors(currentNode):
            if explored.count(successor[0]) == 0:
                # Creating a list which consist of a direction from currentNode to Successor and a Successor
                child_node_and_direction = []
                child_node_and_direction.append(successor[1])
                child_node_and_direction.append(successor[0])
                # Will adding successor to frontier, the full path from starting node to successor is require.
                # By merging path of currentNode and path from currentNode to Successor will give full path of
                # Successor from starting Node
                frontier.push(node_with_direction_list + child_node_and_direction)

    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    
    frontier = util.Queue() # frontier will contain those states which should to be expanded 
    #along with there path from start node of problem
    # Order will be First In First Out

    frontier_only_node = util.Queue() 
    # frontierOnlyNode contain only those nodes which should be expanded, not with there
    # path from start node of problem 

    explored = list() # explored List will contain list of nodes which are already expanded.
    start_state = problem.getStartState()
    frontier_only_node.push(start_state)
    node_with_direction_list = [start_state]

    frontier.push(node_with_direction_list) # frontier is list of list (Direction to read a Node and a Node)
    # Example frontier = list(['North','South','B'],['East','West','C'])
    # Here frontier contains the list of list of (Direction to reach B along with node B) and
    # (Direction to reach C along with node C).  

    while frontier.isEmpty() != True:

        node_with_direction_list = frontier.pop()
        frontier_only_node.pop()
        current_node = node_with_direction_list.pop()
        # After pop operation, nodeWithDirectionList will contain only path to reach currentNode
        explored.append(current_node)
        if problem.isGoalState(current_node) == True:
            return node_with_direction_list

        for successor in problem.getSuccessors(current_node):

            if (explored.count(successor[0]) == 0)  and (frontier_only_node.list.count(successor[0]) == 0):
                # Creating a list which consist of a direction from currentNode to Successor and a Successor
                child_node_and_direction = []
                child_node_and_direction.append(successor[1])
                child_node_and_direction.append(successor[0])
                # Will adding successor to frontier, the full path from starting node to successor is require.
                # By merging path of currentNode and path from currentNode to Successor will give full path of
                # Successor from starting Node
                frontier.push(node_with_direction_list + child_node_and_direction)
                frontier_only_node.push(successor[0])

    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # frontier_with_direction is a priority queue
    # Which will contain list of list of (Direction to reach a node and a node) based on priority.
    # pop order will be based on priority
    frontier_with_direction = util.PriorityQueue()
    
    # frontier_with_path_cost is a priority queue
    # Which will contain list of list of (Path cost to reach a node and a node) based on priority.
    # pop order will be based on priority
    frontier_with_path_cost = util.PriorityQueue()
    
    # explored list will contain all state which are been explored.
    explored = list()
    
    start_state = problem.getStartState()
    node_with_direction_list = [start_state]
    
    frontier_with_direction.push([node_with_direction_list],0)
    frontier_with_path_cost.push([0,start_state], 0)
    
    while frontier_with_path_cost.isEmpty() != True:
        
        # Taking a state from frontier to expand it
        node_with_direction_list = frontier_with_direction.pop()
        
        # Element of node_with_direction_list contain : Direction to a state and a state
        # So to use Direction to reach a state from starting state we need to pop up a Node from a list
        node_with_direction_list.pop()
        # Now node_with_direction_list will contains a direction to reach a current state from starting state

        node_and_path_cost = frontier_with_path_cost.pop()
        
        if (explored.count(node_and_path_cost[1]) != 0):
            continue
        if problem.isGoalState(node_and_path_cost[1]) == True:
            return node_with_direction_list
        
        explored.append(node_and_path_cost[1])
        cost_to_teach_current_node = node_and_path_cost[0]

        # Generating successor of current state
        for successor in problem.getSuccessors(node_and_path_cost[1]):
            if explored.count(successor[0]) == 0:

                child_node_and_direction = []
                child_node_and_direction.append(successor[1])
                child_node_and_direction.append(successor[0])
                
                # Priority Value of successor on which they will be expanded
                priority_value = cost_to_teach_current_node + successor[2]

                frontier_with_direction.push(node_with_direction_list + child_node_and_direction,priority_value)
                frontier_with_path_cost.push([priority_value,successor[0]],priority_value)

                        

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # frontier_with_direction is a priority queue
    # Which will contain list of list of (Direction to reach a node and a node) based on priority.
    # pop order will be based on priority
    frontier_with_direction = util.PriorityQueue()
    
    # frontier_with_path_cost is a priority queue
    # Which will contain list of list of (Path cost to reach a node and a node) based on priority.
    # pop order will be based on priority
    frontier_with_path_cost = util.PriorityQueue()
    
    # explored list will contain all state which are been explored.
    explored=[]
    
    start_state = problem.getStartState()
    node_with_direction_list = [start_state]
    
    heuristic_value_of_start_node = heuristic(start_state,problem)
    
    frontier_with_direction.push([node_with_direction_list],heuristic_value_of_start_node)
    frontier_with_path_cost.push([0,start_state],heuristic_value_of_start_node)

    while frontier_with_path_cost.isEmpty() != True:
        # Taking a state from frontier to expand it
        node_with_direction_list = frontier_with_direction.pop()
        
        # Element of frontierWithDirection contain : Direction to a state and a state
        # So to use Direction to reach a state from starting state we need to pop up a Node from a list
        node_with_direction_list.pop()
        # Now nodeWithDirectionList will contains a direction to reach a current state from starting state
        
        node_and_path_cost = frontier_with_path_cost.pop() 
        cost_to_reach_current_node = node_and_path_cost[0]
        
        if (explored.count(node_and_path_cost[1]) != 0):
            continue
        if problem.isGoalState(node_and_path_cost[1]) == True:
            return node_with_direction_list
        
        explored.append(node_and_path_cost[1])
        
        # Generating successor of current state
        for successor in problem.getSuccessors(node_and_path_cost[1]):
            if explored.count(successor[0]) == 0:
                child_node_and_direction = []
                child_node_and_direction.append(successor[1])
                child_node_and_direction.append(successor[0])
                # (g) Priority Value of successor with our heuristic value
                priority_value_with_out_heuristic = cost_to_reach_current_node + successor[2]
                # f = g + h : Priority Value of successor with  heuristic value
                priority_value_with_heuristic = priority_value_with_out_heuristic + heuristic(successor[0],problem)

                frontier_with_direction.push(node_with_direction_list + child_node_and_direction,priority_value_with_heuristic)
                frontier_with_path_cost.push([priority_value_with_out_heuristic,successor[0]],priority_value_with_heuristic) 

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

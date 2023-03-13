#!/usr/bin/env python
# coding: utf-8

# In[52]:


# IMPORTING NECESSARY LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import heapq
import matplotlib.animation as animation
import time
########## DEFINING A NODE CLASS TO STORE NODES AS OBJECTS ########

class Node:
    def __init__(self, x, y, cost, parent_id):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_id = parent_id

    def __lt__(self, other):
        return self.cost < other.cost


# Define actions and their costs
# Actions Set = {(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)}

ACTIONS = {
    'UP': (lambda x, y, cost: (x, y + 1, cost + 1)),
    'DOWN': (lambda x, y, cost: (x, y - 1, cost + 1)),
    'LEFT': (lambda x, y, cost: (x - 1, y, cost + 1)),
    'RIGHT': (lambda x, y, cost: (x + 1, y, cost + 1)),
    'UPRIGHT': (lambda x, y, cost: (x + 1, y + 1, cost + np.sqrt(2))),
    'UPLEFT': (lambda x, y, cost: (x - 1, y + 1, cost + np.sqrt(2))),
    'DOWNRIGHT': (lambda x, y, cost: (x + 1, y - 1, cost + np.sqrt(2))),
    'DOWNLEFT': (lambda x, y, cost: (x - 1, y - 1, cost + np.sqrt(2))),
}


def Action_set(action, x, y, cost):
    """Performs an action and returns the resulting x, y, and cost."""
    return ACTIONS[action](x, y, cost)


def get_actions(x, y, cost):
    """Returns the valid actions for a given position."""
    actions = []
    for action in ACTIONS:
        new_x, new_y, new_cost = Action_set(action, x, y, cost)
        if 0 <= new_x < 10 and 0 <= new_y < 10:
            actions.append((new_x, new_y, new_cost))
    return actions


############ CONFIGURATION SPACE CONSTRUCTION WITH OBSTACLES ############

def obstacle_space(width, height):
    Obstacle_Space = np.full((height, width), 0)
    for y in range(height):
        for x in range(width):

            # Rectangle obstacle with clearance
            Rec1_c = (x - 5) - 150
            Rec2_c = (x + 5) - 100
            Rec3_c = y - 5 - 100
            Rec4_c = y + 5 - 150

            if (Rec1_c < 0 and Rec2_c > 0):
                Obstacle_Space[y, x] = 1

            if (Rec3_c > 0 and Rec4_c < 0):
                Obstacle_Space[y, x] = 1

            # rectangle obstacle
            Rec1 = (x) - 150
            Rec2 = (x) - 100
            Rec3 = y - 100
            Rec4 = y - 150

            if (Rec1 < 0 and Rec2 > 0):
                Obstacle_Space[y, x] = 2

            if (Rec3 > 0 and Rec4 < 0):
                Obstacle_Space[y, x] = 0

            # Hexagon Obstacle with clearance
            hexa1_c = y - 0.577 * x + 123.21 + 5.7726
            hexa2_c = x - 364.95 - 5.7726
            hexa3_c = y + 0.577 * x - 373.21 - 5.7726
            hexa4_c = y - 0.577 * x - 26.92 - 5.7726
            hexa5_c = x - 235 + 5.7726
            hexa6_c = y + 0.577 * x - 223.08 + 5.7726

            if (hexa2_c < 0 and hexa5_c > 0 and hexa1_c > 0 and hexa3_c < 0 and hexa4_c < 0 and hexa6_c > 0):
                Obstacle_Space[y, x] = 1

            # Hexagon Obstacle
            hexa1 = y - 0.577 * x + 123.21
            hexa2 = x - 364.95
            hexa3 = y + 0.577 * x - 373.21
            hexa4 = y - 0.577 * x - 26.92
            hexa5 = x - 235
            hexa6 = y + 0.577 * x - 223.08

            if (hexa2 < 0 and hexa5 > 0 and hexa1 > 0 and hexa3 < 0 and hexa4 < 0 and hexa6 > 0):
                Obstacle_Space[y, x] = 2

            # triangle obstacle with clearance
            tri1_c = (x + 5) - 460
            tri2_c = (y) + 2 * x - 1156.1803
            tri3_c = (y) - 2 * x + 906.1803
            if (tri1_c > 0 and tri2_c < 0 and tri3_c > 0):
                Obstacle_Space[y, x] = 1

            # traiangle obstacle
            tri1 = (x) - 460
            tri2 = (y) + 2 * (x) - 1145
            tri3 = (y) - 2 * (x) + 895

            if (tri1 > 0 and tri2 < 0 and tri3 > 0):
                Obstacle_Space[y, x] = 2

    # Map Surrrounding Clearnce
    Obstacle_Space[:5, :width] = 1
    Obstacle_Space[height - 5:height, :width] = 1
    Obstacle_Space[:height, :5] = 1
    Obstacle_Space[:height, width - 5:width] = 1

    return Obstacle_Space


########## TO SEE IF THE MOVE IS VALID OR NOT #########

def Validmove(x, y, Obstacle_Space):
    """Checks if a move is valid given the current position and obstacle space."""
    height, width = Obstacle_Space.shape

    # Check if the move is within the bounds of the obstacle space
    if x < 0 or x >= width or y < 0 or y >= height:
        return False

    # Check if the move would intersect with an obstacle (represented by 1 or 2 in Obstacle_Space)
    if Obstacle_Space[y][x] in [1, 2]:
        return False

    return True


########## DEFINING A FUNCTION TO CHECK IF THE PRESENT NODE IS GOAL NODE ##########

def Check_goal(current_node, goal_node):
    if current_node.x == goal_node.x and current_node.y == goal_node.y:
        return True
    else:
        return False


########## DIJKSTRA ALGORITHM ###########

def dijkstra(start, goal, Obstacle_Space):
    if Check_goal(start, goal):
        return None, 1

    goal_node = goal
    start_node = start

    moves = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']
    unexplored_nodes = {}  # List of all open nodes
    unexplored_nodes[(start_node.x, start_node.y)] = start_node

    explored_nodes = {}  # List of all closed nodes
    priority_list = []  # List to store all dictionary entries with cost as the sorting variable
    heapq.heappush(priority_list, [start_node.cost, start_node])

    nodes = []  # stores all nodes that have been traversed, for visualization purposes.

    while len(priority_list) != 0:
        # popping the first element in the priority list to create child nodes for exploration
        present_node = heapq.heappop(priority_list)[1]
        # appending all child nodes so that the explored region of the map can be plotted.
        nodes.append([present_node.x, present_node.y])
        # creating a dict key for identification of node individually
        present_id = (present_node.x, present_node.y)

        # The program will exist if the present node is the goal node
        if Check_goal(present_node, goal_node):
            goal_node.parent_id = present_node.parent_id
            goal_node.cost = present_node.cost
            print("Goal Node found")
            return nodes, 1

        if present_id in explored_nodes:
            continue
        else:
            explored_nodes[present_id] = present_node

        # deleting the node from the open nodes list because it has been explored and further its child nodes will be generated
        del unexplored_nodes[present_id]

        # For all actions in action set, a new child node has to be formed if it is not already explored
        for move in moves:
            x, y, cost = Action_set(move, present_node.x, present_node.y, present_node.cost)

            # Creating a node class object for all coordinates being explored
            new_node = Node(x, y, cost, present_node)
            new_node_id = (new_node.x, new_node.y)

            if not Validmove(new_node.x, new_node.y, Obstacle_Space):
                continue
            elif new_node_id in explored_nodes:
                continue

            if new_node_id in unexplored_nodes:
                if new_node.cost < unexplored_nodes[new_node_id].cost:
                    unexplored_nodes[new_node_id].cost = new_node.cost
                    unexplored_nodes[new_node_id].parent_id = new_node.parent_id
            else:
                unexplored_nodes[new_node_id] = new_node

            heapq.heappush(priority_list, [new_node.cost, new_node])

    return nodes, 0


########### BACKTRACK AND GENERATE SHORTEST PATH ############

def Backtrack(goal_node):
    X = []
    Y = []

    # Add the goal node's position to the path
    X.append(goal_node.x)
    Y.append(goal_node.y)

    # Follow the parent pointers until the start node is reached
    current_node = goal_node
    while current_node.parent_id != -1:
        parent_node = current_node.parent_id
        X.append(parent_node.x)
        Y.append(parent_node.y)
        current_node = parent_node

    # Reverse the paths to get them in the correct order
    X.reverse()
    Y.reverse()

    return X, Y


#########  PLOT OBSTACLES SPACE, EXPLORED NODES, SHORTEST PATH  #######

def plot(start_node, goal_node, X, Y, nodes, Obstacle_Space):
    ### Start node and Goal node ###
    plt.plot(start_node.x, start_node.y, "Dw")
    plt.plot(goal_node.x, goal_node.y, "Dg")

    ### Configuration Space for Obstacles ####
    plt.imshow(Obstacle_Space, "GnBu")
    ax = plt.gca()
    ax.invert_yaxis()  # y-axis inversion

    ### All visited nodes ###
    for i in range(len(nodes)):
        plt.plot(nodes[i][0], nodes[i][1], "2g")
        # plt.pause(0.0001)

    ### Shortest path found ###
    plt.plot(X, Y, ":r")
    plt.show()
    plt.pause(0.001)
    plt.close('all')




######### CALLING ALL MY FUNCTIONS TO IMPLEMENT dijkstra ALGORITHM ON A POINT ROBOT ###########

if __name__ == '__main__':
    width = 600
    height = 250
    Obstacle_Space = obstacle_space(width, height)

    # ask for start node coordinates until a valid input is entered
    while True:
        start_coordinates = input("Enter the coordinates  for Start Node: ")
        Start_X, Start_Y = map(int, start_coordinates.split())
        if Validmove(Start_X, Start_Y, Obstacle_Space):
            break
        else:
            print("Start node is not valid or it is in obstacle space")

    # ask for goal node coordinates until a valid input is entered
    while True:
        goal_coordinates = input("Enter the coordinates  for Goal Node: ")
        goal_X, goal_Y = map(int, goal_coordinates.split())
        if Validmove(goal_X, goal_Y, Obstacle_Space):
            break
        else:
            print("Goal node is not valid or it is in the obstacle space")

    # compute shortest path using Dijkstra's algorithm
    timer_start = time.time()
    start_node = Node(Start_X, Start_Y, 0.0, -1)
    goal_node = Node(goal_X, goal_Y, 0.0, -1)
    nodes, a = dijkstra(start_node, goal_node, Obstacle_Space)
    timer_stop = time.time()
    
    
    # if a path was found, plot it
    if a == 1:
        X, Y = Backtrack(goal_node)
        plot(start_node, goal_node, X, Y, nodes, Obstacle_Space)
    else:
        print("Path is not found")

    # print runtime
    time = timer_stop - timer_start
    print("The Total Runtime taken to reach the goal is: ", time)

# IMPORTING NECESSARY LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import heapq
import matplotlib.animation as animation
import time


''' DEFINING A NODE CLASS TO STORE NODES AS OBJECTS'''

class Node:
    def __init__(self, x, y, cost, parent_id):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_id = parent_id

    def __lt__(self, other):
        return self.cost < other.cost


# Define actions and their costs
# Actions Set = {(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)}

ACTIONS = {
    'UP': (lambda x, y, cost: (x, y + 1, cost + 1)),
    'DOWN': (lambda x, y, cost: (x, y - 1, cost + 1)),
    'LEFT': (lambda x, y, cost: (x - 1, y, cost + 1)),
    'RIGHT': (lambda x, y, cost: (x + 1, y, cost + 1)),
    'UPRIGHT': (lambda x, y, cost: (x + 1, y + 1, cost + np.sqrt(2))),
    'UPLEFT': (lambda x, y, cost: (x - 1, y + 1, cost + np.sqrt(2))),
    'DOWNRIGHT': (lambda x, y, cost: (x + 1, y - 1, cost + np.sqrt(2))),
    'DOWNLEFT': (lambda x, y, cost: (x - 1, y - 1, cost + np.sqrt(2))),
}


def Action_set(action, x, y, cost):
    """Performs an action and returns the resulting x, y, and cost."""
    return ACTIONS[action](x, y, cost)


def get_actions(x, y, cost):
    """Returns the valid actions for a given position."""
    actions = []
    for action in ACTIONS:
        new_x, new_y, new_cost = Action_set(action, x, y, cost)
        if 0 <= new_x < 10 and 0 <= new_y < 10:
            actions.append((new_x, new_y, new_cost))
    return actions


''' CONFIGURATION OF OBSTACLE SPACE  '''

def obstacle_space(width, height):
    Obstacle_Space = np.full((height, width), 0)
    for y in range(height):
        for x in range(width):

            # Rectangle obstacle with clearance
            Rec1_c = (x - 5) - 150
            Rec2_c = (x + 5) - 100
            Rec3_c = y - 5 - 100
            Rec4_c = y + 5 - 150

            if (Rec1_c < 0 and Rec2_c > 0):
                Obstacle_Space[y, x] = 1

            if (Rec3_c > 0 and Rec4_c < 0):
                Obstacle_Space[y, x] = 1

            # rectangle obstacle
            Rec1 = (x) - 150
            Rec2 = (x) - 100
            Rec3 = y - 100
            Rec4 = y - 150

            if (Rec1 < 0 and Rec2 > 0):
                Obstacle_Space[y, x] = 2

            if (Rec3 > 0 and Rec4 < 0):
                Obstacle_Space[y, x] = 0

            # Hexagon Obstacle with clearance
            hexa1_c = y - 0.577 * x + 123.21 + 5.7726
            hexa2_c = x - 364.95 - 5.7726
            hexa3_c = y + 0.577 * x - 373.21 - 5.7726
            hexa4_c = y - 0.577 * x - 26.92 - 5.7726
            hexa5_c = x - 235 + 5.7726
            hexa6_c = y + 0.577 * x - 223.08 + 5.7726

            if (hexa2_c < 0 and hexa5_c > 0 and hexa1_c > 0 and hexa3_c < 0 and hexa4_c < 0 and hexa6_c > 0):
                Obstacle_Space[y, x] = 1

            # Hexagon Obstacle
            hexa1 = y - 0.577 * x + 123.21
            hexa2 = x - 364.95
            hexa3 = y + 0.577 * x - 373.21
            hexa4 = y - 0.577 * x - 26.92
            hexa5 = x - 235
            hexa6 = y + 0.577 * x - 223.08

            if (hexa2 < 0 and hexa5 > 0 and hexa1 > 0 and hexa3 < 0 and hexa4 < 0 and hexa6 > 0):
                Obstacle_Space[y, x] = 2

            # triangle obstacle with clearance
            tri1_c = (x + 5) - 460
            tri2_c = (y) + 2 * x - 1156.1803
            tri3_c = (y) - 2 * x + 906.1803
            if (tri1_c > 0 and tri2_c < 0 and tri3_c > 0):
                Obstacle_Space[y, x] = 1

            # traiangle obstacle
            tri1 = (x) - 460
            tri2 = (y) + 2 * (x) - 1145
            tri3 = (y) - 2 * (x) + 895

            if (tri1 > 0 and tri2 < 0 and tri3 > 0):
                Obstacle_Space[y, x] = 2

    # Map Surrrounding Clearnce
    Obstacle_Space[:5, :width] = 1
    Obstacle_Space[height - 5:height, :width] = 1
    Obstacle_Space[:height, :5] = 1
    Obstacle_Space[:height, width - 5:width] = 1

    return Obstacle_Space


''' TO SEE IF THE MOVE IS VALID OR NOT '''

def Validmove(x, y, Obstacle_Space):
    """Checks if a move is valid given the current position and obstacle space."""
    height, width = Obstacle_Space.shape

    # Check if the move is within the bounds of the obstacle space
    if x < 0 or x >= width or y < 0 or y >= height:
        return False

    # Check if the move would intersect with an obstacle (represented by 1 or 2 in Obstacle_Space)
    if Obstacle_Space[y][x] in [1, 2]:
        return False

    return True


''' DEFINING A FUNCTION TO CHECK IF THE PRESENT NODE IS GOAL NODE'''

def Check_goal(current_node, goal_node):
    if current_node.x == goal_node.x and current_node.y == goal_node.y:
        return True
    else:
        return False


'''DIJKSTRA ALGORITHM '''

def dijkstra(start, goal, Obstacle_Space):
    if Check_goal(start, goal):
        return None, 1

    goal_node = goal
    start_node = start

    moves = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']
    unexplored_nodes = {}  # List of all open nodes
    unexplored_nodes[(start_node.x, start_node.y)] = start_node

    explored_nodes = {}  # List of all closed nodes
    priority_list = []  # List to store all dictionary entries with cost as the sorting variable
    heapq.heappush(priority_list, [start_node.cost, start_node])

    nodes = []  # stores all nodes that have been traversed, for visualization purposes.

    while len(priority_list) != 0:
        # popping the first element in the priority list to create child nodes for exploration
        present_node = heapq.heappop(priority_list)[1]
        # appending all child nodes so that the explored region of the map can be plotted.
        nodes.append([present_node.x, present_node.y])
        # creating a dict key for identification of node individually
        present_id = (present_node.x, present_node.y)

        # The program will exist if the present node is the goal node
        if Check_goal(present_node, goal_node):
            goal_node.parent_id = present_node.parent_id
            goal_node.cost = present_node.cost
            print("Goal Node found")
            return nodes, 1

        if present_id in explored_nodes:
            continue
        else:
            explored_nodes[present_id] = present_node

        # deleting the node from the open nodes list because it has been explored and further its child nodes will be generated
        del unexplored_nodes[present_id]

        # For all actions in action set, a new child node has to be formed if it is not already explored
        for move in moves:
            x, y, cost = Action_set(move, present_node.x, present_node.y, present_node.cost)

            # Creating a node class object for all coordinates being explored
            new_node = Node(x, y, cost, present_node)
            new_node_id = (new_node.x, new_node.y)

            if not Validmove(new_node.x, new_node.y, Obstacle_Space):
                continue
            elif new_node_id in explored_nodes:
                continue

            if new_node_id in unexplored_nodes:
                if new_node.cost < unexplored_nodes[new_node_id].cost:
                    unexplored_nodes[new_node_id].cost = new_node.cost
                    unexplored_nodes[new_node_id].parent_id = new_node.parent_id
            else:
                unexplored_nodes[new_node_id] = new_node

            heapq.heappush(priority_list, [new_node.cost, new_node])

    return nodes, 0


''' BACKTRACK AND GENERATE SHORTEST PATH '''

def Backtrack(goal_node):
    X = []
    Y = []

    # Add the goal node's position to the path
    X.append(goal_node.x)
    Y.append(goal_node.y)

    # Follow the parent pointers until the start node is reached
    current_node = goal_node
    while current_node.parent_id != -1:
        parent_node = current_node.parent_id
        X.append(parent_node.x)
        Y.append(parent_node.y)
        current_node = parent_node

    # Reverse the paths to get them in the correct order
    X.reverse()
    Y.reverse()

    return X, Y


'''  PLOT OBSTACLES SPACE, EXPLORED NODES, SHORTEST PATH '''

# def plot(start_node, goal_node, X, Y, nodes, Obstacle_Space):
#     ### Start node and Goal node ###
#     plt.plot(start_node.x, start_node.y, "Dw")
#     plt.plot(goal_node.x, goal_node.y, "Dg")

#     ### Configuration Space for Obstacles ####
#     plt.imshow(Obstacle_Space, "GnBu")
#     ax = plt.gca()
#     ax.invert_yaxis()  # y-axis inversion

#     ### All visited nodes ###
#     for i in range(len(nodes)):
#         plt.plot(nodes[i][0], nodes[i][1], "2g")
#         plt.pause(0.0001)

#     ### Shortest path found ###
#     plt.plot(X, Y, ":r")
#     plt.show()
#     plt.pause(0.01)
#     plt.close('all')

'''  PLOT OBSTACLES SPACE, EXPLORED NODES, SHORTEST PATH FOR ANIMATION  '''

def plot(start_node, goal_node, X, Y, nodes, Obstacle_Space):

    fig, ax = plt.subplots()

    def update(i):
        ax.clear()

        ### Start node and Goal node ###
        ax.plot(start_node.x, start_node.y, "Dw")
        ax.plot(goal_node.x, goal_node.y, "Dg")

        ### Configuration Space for Obstacles ####
        ax.imshow(Obstacle_Space, "GnBu")
        ax.invert_yaxis() #y-axis inversion

        ### All visited nodes ###
        for j in range(i):
            ax.plot(nodes[j][0], nodes[j][1], "2g")

        ### Shortest path found ###
        ax.plot(X[:i], Y[:i], ":r")

    ani = animation.FuncAnimation(fig, update, frames=len(nodes), interval=500)
    plt.show()


'''CALLING ALL MY FUNCTIONS TO IMPLEMENT dijkstra ALGORITHM ON A POINT ROBOT '''

if __name__ == '__main__':
    width = 600
    height = 250
    Obstacle_Space = obstacle_space(width, height)

    # ask for start node coordinates until a valid input is entered
    while True:
        start_coordinates = input("Enter the coordinates  for Start Node: ")
        Start_X, Start_Y = map(int, start_coordinates.split())
        if Validmove(Start_X, Start_Y, Obstacle_Space):
            break
        else:
            print("Start node is not valid or it is in obstacle space")

    # ask for goal node coordinates until a valid input is entered
    while True:
        goal_coordinates = input("Enter the coordinates  for Goal Node: ")
        goal_X, goal_Y = map(int, goal_coordinates.split())
        if Validmove(goal_X, goal_Y, Obstacle_Space):
            break
        else:
            print("Goal node is not valid or it is in the obstacle space")

    # compute shortest path using Dijkstra's algorithm
    timer_start = time.time()
    start_node = Node(Start_X, Start_Y, 0.0, -1)
    goal_node = Node(goal_X, goal_Y, 0.0, -1)
    nodes, a = dijkstra(start_node, goal_node, Obstacle_Space)
    timer_stop = time.time()

    # if a path was found, plot it
    if a == 1:
        X, Y = Backtrack(goal_node)
        plot(start_node, goal_node, X, Y, nodes, Obstacle_Space)
    else:
        print("Path is not found")

    # print runtime
    time = timer_stop - timer_start
    print("The Total Runtime taken to reach the goal is: ", time)






# In[ ]:





# In[ ]:





# In[ ]:





# Write your code here :-)
import sys


class Node():
    def __init__(self, state, parent, action, h, g):
        self.state = state
        self.parent = parent
        self.action = action
        self.h_n = h
        self.g_n = g
        self.sum = self.h_n + self.g_n

# Frontier class for DFS Algorithm
class StakeFrontier():
    def __init__(self):
        # DFS and BFS frontier attribute
        self.frontier = []
        # GBPS frontier attribute
        self.list_hn = []
        # A star search algorithm attribute
        self.list_sum = []

    def add(self, node):
        self.frontier.append(node)

    """The method contains_state checks if any node in self.frontier has a state equal to the given state.
    If it finds such a node, it returns True; otherwise, it returns False.
    This is useful in search algorithms to determine if a certain state has already been encountered."""

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    """The empty method checks whether self.frontier is empty (i.e., contains no elements).
    If the length of self.frontier is 0, it returns True, indicating that the frontier is empty.
    If the length is not 0, it returns False, indicating that the frontier contains one or more elements."""

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("Empty Frontier, There may be no solution")
        else:
            node = self.frontier[-1]
            # This line updates self.frontier by removing the last element.
            # The slice [:-1] means "all elements except the last one.
            self.frontier = self.frontier[:-1]
            return node

# Frontier class for BFS
class QueueFrontier(StakeFrontier):
    def remove(self):
        if self.empty():
            raise Exception("Empty Frontier, There may be no solution")
        else:
            node = self.frontier[0]
            # This line updates self.frontier by removing the first element.
            self.frontier = self.frontier[1:]
            return node

# Frontier class for GBFS
class GBFS(StakeFrontier):
    def add(self, node):
        self.frontier.append(node)
        self.list_hn.append(node.h_n)

    def remove(self):
        if self.empty():
            raise Exception("Empty Frontier, There may be no solution")
        else:
            if len(self.list_hn) == len(self.frontier):
                minimum_value = self.list_hn[0]
                for i in range(1, len(self.list_hn)):
                    if (self.list_hn[i] < minimum_value):
                        minimum_value = self.list_hn[i]
                min_index = self.list_hn.index(minimum_value)
                node = self.frontier[min_index]
                # This line updates self.frontier by removing the node selected based on minimum Manhattan distance h_n .
                self.frontier.remove(node)
                self.list_hn.remove(minimum_value)
                return node

# Frontier class for A-Star Search Algorithm
class StarSearch(StakeFrontier):
    def add(self, node):
        self.frontier.append(node)
        self.list_sum.append(node.sum)

    def remove(self):
        if self.empty():
            raise Exception("Empty Frontier, There may be no solution")
        else:
            if len(self.list_sum) == len(self.frontier):
                minimum_value = self.list_sum[0]
                for i in range(1, len(self.list_sum)):
                    if (self.list_sum[i] < minimum_value):
                        minimum_value = self.list_sum[i]
                min_index = self.list_sum.index(minimum_value)
                node = self.frontier[min_index]
                # This line updates self.frontier by removing the node selected based on minimum Manhattan distance h_n .
                self.frontier.remove(node)
                self.list_sum.remove(minimum_value)
                return node

class Maze:
    def __init__(self, filename):
        # Read file and set height and width of maze
        with open(filename) as f:
            #Reads the entire contents of the file into the variable contents.
            contents = f.read()
        # validate single-point start and single point goal
        if contents.count("A") != 1:
            if contents.count("S") != 1:
                raise Exception("Maze must have exactly one start point")
            else:
                pass
        if contents.count("B") != 1:
            if contents.count("F") != 1:
                raise Exception("Maze must have exactly one goal point")
            else:
                pass
        # determining height and width of maze
        # The splitlines() method splits a string into a list. The splitting is done at line breaks.
        #Splits the contents of the file into a list of strings, where each string represents a line in the file (a row in the maze).
        contents = contents.splitlines()
        # Sets the height of the maze as the number of lines (rows) in the file.
        #only the starting space is count not the end space
        self.height = len(contents)
        # Determines the width of the maze by finding the length of the longest line in the file.
        self.width = max(len(line) for line in contents)

        # keep track of walls
        # Initializes an empty list to store the maze's structure, where each element represents a row of the maze.
        self.walls = []
        # Loops over each row in the maze.
        for i in range(self.height):
            # Initializes an empty list to store the information for the current row.
            row = []
            # Loops over each column in the current row.
            for j in range(self.width):
                # The try block is used to handle cases where the current line might be shorter than the maximum width of the maze.
                try:
                    # to track start point
                    # If the current cell contains "A", it marks the start position (self.start = (i, j)) and
                    # appends False to row to indicate that this cell is not a wall.
                    if contents[i][j] == "A" or contents[i][j] == "S":
                        self.start = (i, j)
                        row.append(False)
                        # to track goal point
                    elif contents[i][j] == "B" or contents[i][j] == "F":
                        self.goal = (i, j)
                        row.append(False)
                    # to track walls
                    # If the current cell is a space (" "), it appends False to row to indicate that this cell is open (not a wall).
                    elif contents[i][j] == " " or contents[i][j] == ".":
                        row.append(False)
                    # For any other character (typically representing a wall, such as "#"), it appends True to row to indicate that this cell is a wall.
                    else:
                        row.append(True)
                # If the current row is shorter than self.width, an IndexError would occur.
                # In that case, it appends False to row, treating the out-of-bounds cells as open spaces.
                except IndexError:
                    row.append(False)
            self.walls.append(row)

        self.solution = None
        self.start_dist = (abs(self.goal[0] - self.start[0]) + abs(self.goal[1] - self.start[1]))

    def print(self):
        # because cell state row in index 1. index 0 is actions
        solution = self.solution[1] if self.solution is not None else None
        print()
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print("█", end="")
                elif (i, j) == self.start:
                    print("A", end="")
                elif (i, j) == self.goal:
                    print("B", end="")
                elif solution is not None and (i, j) in solution:
                    print("*", end="")
                else:
                    print(" ", end="")
            print()
        print()

    """The state is expected to be a tuple representing the position (row, col) in the maze.
    The method returns a list of possible moves and their resulting states."""

    def neighbors(self, state):
        #state defined as ith row and jth column
        row, col = state
        i_goal = self.goal[0]
        j_goal = self.goal[1]
        candidates = [
            ("up", (row - 1, col), (abs(i_goal-(row-1)) + abs(j_goal - col))),
            ("down", (row + 1, col), (abs(i_goal-(row+1)) + abs(j_goal - col))),
            ("left", (row, col - 1), (abs(i_goal-row) + abs(j_goal - (col-1)))),
            ("right", (row, col + 1), (abs(i_goal-row) + abs(j_goal - (col+1))))
        ]
        # This empty list will store the valid neighboring states and their associated actions.
        result = []
        # Checking Validity of Each Candidate Move
        for action, (r, c), dist in candidates:
            # Checks if the new row r is within the bounds of the maze.
            # Checks if the new column c is within the bounds of the maze.
            # Checks if the new position (r, c) is not a wall (False in self.walls).
            # less than maximum height and width because row and column starts with 0
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                # If all conditions are met, the move is considered valid, and the (action, (r, c)) tuple is appended to the result list.
                result.append((action, (r, c), dist))
        return result

    def solve(self):
        """Find a solution to the Maze problem if one exist"""
        # This variable keeps track of the number of states (or nodes) that have been explored during the search process.
        self.number_explored = 0
        # initialising the frontier
        # A Node object representing the starting position of the maze.
        # The current state (position) is the starting point of the maze. The start node has no parent because it's the first node.
        # No action was taken to arrive at the start node.
        start = Node(state=self.start, parent=None, action=None, h = self.start_dist, g = 0)
        # A StackFrontier object is used to manage the frontier (the set of nodes that are candidates for exploration).
        frontier = StarSearch()
        frontier.add(start)
        # A set used to keep track of all the states (positions) that have already been explored, to avoid revisiting them.
        self.explored = set()

        # This while True loop continues until a solution is found or it is determined that no solution exists.
        while True:
            """If the frontier is empty (i.e., there are no more nodes to explore),
            it means that all possible paths have been exhausted and no solution exists. An exception is raised to indicate this."""
            if frontier.empty():
                raise Exception("No solution for this maze")
            # Choose a node from the frontier
            node = frontier.remove()
            self.number_explored += 1

            # if chosen node is th goal, then we have solution
            if node.state == self.goal:
                # These lists will store the sequence of actions and the corresponding states (positions) that lead from the start to the goal.
                actions = []
                cells = []
                # This loop traces back from the goal to the start by following the parent nodes.
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                    # Reverse the lists since they were collected from the goal back to the start.
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                # Exits the method since the solution has been found.
                return
            # Adds the current node's state to the explored set to prevent revisiting it.
            self.explored.add(node.state)

            # Iterates over all the neighboring states (and corresponding actions) that can be reached from the current node.
            for action, state, dist in self.neighbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    # The new state (position).The current node is the parent of this new node. The action that led to this state.
                    child = Node(state=state, parent=node, action=action, h = dist, g = node.g_n + 1)
                    # add this new node to frontier for future exploration
                    frontier.add(child)

    def output_image(self, filename, show_solution=True, show_explored=False):
        from PIL import Image, ImageDraw

        # This variable sets the size of each cell in the maze. Here, each cell will be 50x50 pixels.
        cell_size = 50
        # This variable sets the border width around each cell. The border helps visually distinguish cells.
        cell_border = 2
        # creating blank canvas
        # Creates a new blank image with RGBA mode (Red, Green, Blue, Alpha for transparency).
        # The size of the image is calculated based on the dimensions of the maze (self.width and self.height) multiplied by the cell size.
        # The background color of the image is initially set to black.
        img = Image.new(
            "RGBA", (self.width * cell_size, self.height * cell_size), "black"
        )
        # Creates an ImageDraw object associated with img, allowing you to draw shapes and text on the image.
        draw = ImageDraw.Draw(img)

        # If a solution has been found (self.solution is not None), this retrieves the list of states (cells) that make up the solution path.
        solution = self.solution[1] if self.solution is not None else None
        # Loops over each row (row) in self.walls, with i being the row index.
        for i, row in enumerate(self.walls):
            # Loops over each cell (col) in the row, with j being the column index.
            for j, col in enumerate(row):
                # walls
                # The fill variable determines the color of each cell based on its type:
                if col:
                    # If the cell is a wall (col is True), it is filled with a dark gray color (40, 40, 40).
                    fill = (40, 40, 40)
                # start
                elif (i, j) == self.start:
                    # If the cell is the start position, it is filled with red (255, 0, 0).
                    fill = (255, 0, 0)
                # if goal
                elif (i, j) == self.goal:
                    # If the cell is the goal position, it is filled with green (0, 171, 28).
                    fill = (0, 171, 28)
                # if solution
                elif solution is not None and show_solution and (i, j) in solution:
                    # If the solution exists, and the show_solution flag is True,
                    # and the cell is part of the solution path, it is filled with a light yellow color (220, 235, 113).
                    fill = (220, 235, 113)
                elif solution is not None and show_explored and (i, j) in self.explored:
                    # If the solution exists, and the show_explored flag is True,
                    # and the cell was explored during the search, it is filled with a light red color (212, 97, 85).
                    fill = (212, 97, 85)
                else:
                    # All other cells (i.e., open paths that are not part of the solution or explored) are filled with a light blue color (237, 240, 252).
                    fill = (237, 240, 252)
                # draw each cell
                # Draws a rectangle representing the current cell.
                # The rectangle's coordinates are calculated based on the cell's row (i) and column (j), scaled by cell_size and adjusted by cell_border.
                # The color of the rectangle is set to the value of fill, determined earlier
                draw.rectangle(
                    (
                        [
                            (j * cell_size + cell_border, i * cell_size + cell_border),
                            (
                                (j + 1) * cell_size - cell_border,
                                (i + 1) * cell_size - cell_border,
                            ),
                        ]
                    ),
                    fill=fill,
                )
        # saving the image
        img.save(filename)


if len(sys.argv) != 2:
    sys.exit("Usage: python maze.py maze.txt")
# for filename as input to Maze class
m = Maze(sys.argv[1])
print("Maze:")
m.print()
print("Solving.....")
m.solve()
print("States explored: ", m.number_explored)
print("Solution:")
m.print()
m.output_image("maze_solution.png", show_solution=True, show_explored=True)

import heapq
import time


class VacuumCleanerEnvironment:
    def __init__(self, gridSize=(4, 5), DirtLocations=None, InitialLocation=(1, 1)):
        # initialize the environment with the grid, vacuum position, and dirt locations.
        self.gridSize = gridSize
        self.DirtLocations = DirtLocations or []
        self.InitialLocation = InitialLocation

    def GoalState(self, state):
        # Check if the goal state of no dirty rooms has been reached
        return len(state["dirt"]) == 0

    def potentialSuccessors(self, state):
        # Create all possible successor states given the current state
        # possible moves are up, left, right, down, or suck
        successors = []
        agentRow, AgentCol = state["agent"]
        DirtLocations = state["dirt"]

        actions = [
            ("left", (0, -1), 1.0),
            ("right", (0, 1), 0.9),
            ("up", (-1, 0), 0.8),
            ("down", (1, 0), 0.7),
            ("suck", (0, 0), 0.6),
        ]

        for action, move, cost in actions:
            newRow = agentRow + move[0]
            newCol = AgentCol + move[1]

            # check to see if the move will stay within grid
            if 1 <= newRow <= self.gridSize[0] and 1 <= newCol <= self.gridSize[1]:
                newDirtLocations = list(DirtLocations)

                # Check if suck action was performed and remove dirt from room
                if action == "suck" and (agentRow, AgentCol) in newDirtLocations:
                    newDirtLocations.remove((agentRow, AgentCol))

                # adjust the new state to what the action will result in
                NewState = {"agent": (newRow, newCol), "dirt": newDirtLocations}
                successors.append((NewState, action, cost))
        return successors


class SearchNode:
    def __init__(self, state, parent=None, action=None, path_cost=0, depth=0):
        # Search node in problem space with current state
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = depth

    def __lt__(self, other):
        # Define comparison based on path_cost
        return self.path_cost < other.path_cost

    def print_solution(self):
        # Utility function to trace the solution path from the final node to the start.
        actions = []
        node = self
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        return actions[::-1]  # Reverse the actions list to get the correct order


class UniformCostTreeSearch:
    def __init__(self, environment):
        self.environment = environment

    def uniform_cost_tree_search(self, initial_state):

        # record the start time
        start_time = time.time()

        # Initialize the priority queue (fringe) with the initial node
        root_node = SearchNode(state=initial_state)
        fringe = [(0, root_node)]  # Priority queue: (path_cost, node)
        explored = set()  # To track explored states

        # Loop until the goal is found or fringe is empty
        while fringe:
            # Pop the node with the lowest cost
            current_cost, current_node = heapq.heappop(fringe)

            # Check if the current node is the goal state
            if self.environment.GoalState(current_node.state):
                print("Goal reached!")

                # Record the end time and calculate the duration
                end_time = time.time()
                duration = end_time - start_time
                print(f"Search completed in {duration:.4f} seconds")

                return current_node  # Return the goal node

            # Add the current state to the explored set
            explored.add(
                tuple(current_node.state["agent"])
                + tuple(sorted(current_node.state["dirt"]))
            )

            # Expand the current node and add its successors to the fringe
            successors = self.environment.potentialSuccessors(current_node.state)
            for succ_state, action, action_cost in successors:
                # Calculate the cumulative path cost for the successor
                new_cost = current_node.path_cost + action_cost
                # Create a new search node for the successor
                new_node = SearchNode(
                    state=succ_state,
                    parent=current_node,
                    action=action,
                    path_cost=new_cost,
                )
                # Convert state to a tuple for comparison
                state_tuple = (
                    tuple(new_node.state["agent"]),
                    tuple(sorted(new_node.state["dirt"])),
                )

                if state_tuple not in explored:
                    # Add the successor node to the fringe
                    heapq.heappush(fringe, (new_cost, new_node))
                    # Print debug information

        # If no solution is found, return failure
        print("No solution found.")
        return None

    def print_solution(self, goal_node):
        # If goal node is reached, print the solution path
        if goal_node:
            actions = goal_node.print_solution()
            print(f"Solution found: {actions}")
            print(f"Number of moves: {len(actions)}")
            print(f"Total cost: {goal_node.path_cost}")
        else:
            print("No solution found.")


class TestCased:
    def test_vacuum_cleaner_world(self):
        # Define both instances of the environment
        instances = [
            VacuumCleanerEnvironment(
                gridSize=(4, 5),
                DirtLocations=[(1, 2), (2, 4), (3, 5)],
                InitialLocation=(2, 2),
            ),
            VacuumCleanerEnvironment(
                gridSize=(4, 5),
                DirtLocations=[(1, 2), (2, 1), (2, 4), (3, 3)],
                InitialLocation=(3, 2),
            ),
        ]

        # Iterate over both instances and run the test for each
        for idx, instance in enumerate(instances, start=1):
            # Define the initial state for the current instance
            initial_state = {
                "agent": instance.InitialLocation,
                "dirt": instance.DirtLocations,
            }

            # Print the initial state
            print(f"\nRunning Uniform Cost Tree Search on Instance #{idx}...")
            print(f"Initial state: {initial_state}")

            # Create a Uniform Cost Tree Search instance and run the search
            uct_search = UniformCostTreeSearch(environment=instance)
            goal_node = uct_search.uniform_cost_tree_search(initial_state)

            # Print the solution
            uct_search.print_solution(goal_node)


# Create an instance of the test class
test_case = TestCased()

# Call the test method to run the test for both instances
test_case.test_vacuum_cleaner_world()

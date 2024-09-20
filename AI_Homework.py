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
            current_cost, current_node = heapq.heappop(fringe)

            if self.environment.GoalState(current_node.state):
                print("Goal reached!")

                # Record the end time and calculate the duration
                end_time = time.time()
                duration = end_time - start_time
                print(f"Search completed in {duration:.4f} seconds")

                return current_node

            state_tuple = (
                tuple(current_node.state["agent"]),
                tuple(sorted(current_node.state["dirt"])),
            )
            explored.add(state_tuple)

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
                new_state_tuple = (
                    tuple(new_node.state["agent"]),
                    tuple(sorted(new_node.state["dirt"])),
                )

                if new_state_tuple not in explored:
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


class TestCasedTree:
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


class UniformCostGraphSearch:
    def __init__(self, environment):
        self.environment = environment

    def uniform_cost_graph_search(self, initial_state):

        # record the start time
        start_time = time.time()

        # Initialize the closed set and fringe (priority queue)
        closed = set()
        root_node = SearchNode(state=initial_state)
        fringe = [(0, root_node)]  # Priority queue: (path_cost, node)

        # Loop until the goal is found or the fringe is empty
        while fringe:
            current_cost, current_node = heapq.heappop(fringe)

            if self.environment.GoalState(current_node.state):
                print("Goal reached!")

                # Record the end time and calculate the duration
                end_time = time.time()
                duration = end_time - start_time
                print(f"Search completed in {duration:.4f} seconds")
                return current_node

            # Convert the state to a tuple for checking in the closed set
            state_tuple = (
                tuple(current_node.state["agent"]),
                tuple(sorted(current_node.state["dirt"])),
            )

            # If the current state is not in the closed set, process it
            if state_tuple not in closed:
                closed.add(state_tuple)

                # Expand the current node
                successors = self.environment.potentialSuccessors(current_node.state)
                for succ_state, action, action_cost in successors:
                    new_cost = current_node.path_cost + action_cost
                    new_node = SearchNode(
                        state=succ_state,
                        parent=current_node,
                        action=action,
                        path_cost=new_cost,
                    )

                    new_state_tuple = (
                        tuple(new_node.state["agent"]),
                        tuple(sorted(new_node.state["dirt"])),
                    )

                    if new_state_tuple not in closed:
                        heapq.heappush(fringe, (new_cost, new_node))

        print("No solution found.")
        return None

    def print_solution(self, goal_node):
        if goal_node:
            actions = goal_node.print_solution()
            print(f"Solution found: {actions}")
            print(f"Number of moves: {len(actions)}")
            print(f"Total cost: {goal_node.path_cost}")
        else:
            print("No solution found.")


class TestCasedGraph:
    def test_vacuum_cleaner_world(self):
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

        for idx, instance in enumerate(instances, start=1):
            initial_state = {
                "agent": instance.InitialLocation,
                "dirt": instance.DirtLocations,
            }

            print(f"\nRunning Uniform Cost Graph Search on Instance #{idx}...")
            print(f"Initial state: {initial_state}")

            uct_graph_search = UniformCostGraphSearch(environment=instance)
            goal_node = uct_graph_search.uniform_cost_graph_search(initial_state)

            uct_graph_search.print_solution(goal_node)


class IterativeDeepeningTreeSearch:
    def __init__(self, environment):
        self.environment = environment

    def depth_limited_search(self, node, depth_limit):
        # Perform a depth-limited search (DFS) with a depth limit
        if self.environment.GoalState(node.state):
            return node

        if depth_limit == 0:
            return None  # Reached the depth limit, stop expanding

        successors = self.environment.potentialSuccessors(node.state)
        for succ_state, action, action_cost in successors:
            new_node = SearchNode(
                state=succ_state,
                parent=node,
                action=action,
                path_cost=node.path_cost + action_cost,  # Add the action cost for each move
                depth=node.depth + 1,
            )
            result = self.depth_limited_search(new_node, depth_limit - 1)
            if result:
                return result  # Return the goal node if the goal is found

        return None  # No solution at this depth

    def iterative_deepening_tree_search(self, initial_state):
        # Record the start time
        start_time = time.time()

        # Start with depth limit 0 and increase the limit with each iteration
        depth_limit = 0
        root_node = SearchNode(state=initial_state)

        # Loop to increment depth limit and perform a depth-limited search
        while True:
            result = self.depth_limited_search(root_node, depth_limit)

            if result:
                print("Goal reached!")
                end_time = time.time()
                duration = end_time - start_time
                print(f"Search completed in {duration:.4f} seconds")
                return result

            depth_limit += 1  # Increment the depth limit for the next iteration

    def print_solution(self, goal_node):
        if goal_node:
            actions = goal_node.print_solution()
            print(f"Solution found: {actions}")
            print(f"Number of moves: {len(actions)}")
            print(f"Total cost: {goal_node.path_cost}")
        else:
            print("No solution found.")


class TestCasedIterativeDeepeningTree:
    def test_vacuum_cleaner_world(self):
        # Define two instances of the vacuum cleaner environment
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
            print(f"\nRunning Iterative Deepening Tree Search on Instance #{idx}...")
            print(f"Initial state: {initial_state}")

            # Create an Iterative Deepening Tree Search instance and run the search
            idt_search = IterativeDeepeningTreeSearch(environment=instance)
            goal_node = idt_search.iterative_deepening_tree_search(initial_state)

            # Print the solution
            idt_search.print_solution(goal_node)


# Create an instance of the test class
test_case1 = TestCasedTree()

# Call the test method to run the test for both instances
test_case1.test_vacuum_cleaner_world()

# Create an instance of the test class
test_case2 = TestCasedGraph()

# Call the test method to run the test for both instances
test_case2.test_vacuum_cleaner_world()

# Create an instance of the test class
test_case3 = TestCasedIterativeDeepeningTree()

# Call the test method to run the test for both instances
test_case3.test_vacuum_cleaner_world()

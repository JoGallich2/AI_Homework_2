import heapq


class VacuumCleanerWorld:
    def WorldCreation(
        self, gridSize=(4, 5), DirtLocations=None, InitialLocation=(1, 1)
    ):

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
            ("up", (1, 0), 0.8),
            ("down", (-1, 0), 0.7),
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
    def CreateNode(self, state, parent=None, action=None, path_cost=0, depth=0):
        # Search node in problem space with current state
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = depth

    def compare_priority(self, other):
        #Defines the less-than operator to priorize certain moves based on agent position

        return (self.state["agent"], self.path_cost) < (
            other.state["agent"],
            other.path_cost,
        )


def print_solution(node):

    #Utility function to trace the solution path from the final node to the start.

    actions = []
    while node.parent is not None:
        actions.append(node.action)
        node = node.parent
    return actions[::-1]  # Reverse the actions list to get the correct order

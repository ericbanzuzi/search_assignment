#!/usr/bin/env python3
import random
import numpy as np
import time

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """
        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()
        self.start_time = None
        self.elapsed_time = 0
        self.time_limit = 62 * 1e-3
        self.transposition_table = {}
        self.timeout = False

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            max_depth = 10
            best_move = self.search_best_next_move_ids(initial_tree_node=node, depth=max_depth)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move_ids(self, initial_tree_node, depth):
        """
        Use iterative deepening search to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :param depth: depth of the search
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """
        self.elapsed_time = 0
        self.timeout = False
        self.start_time = time.time()
        self.transposition_table = {}

        children = initial_tree_node.compute_and_get_children()
        if len(children) == 1:  # a fish was caught and the only option is up
            return ACTION_TO_STR[children[0].move]

        moves = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
        best_move = None
        for max_depth in range(1, depth + 1):
            self.elapsed_time = time.time() - self.start_time
            if self.elapsed_time < self.time_limit:
                move, temp_moves = self.search_at_depth(initial_tree_node, max_depth, moves.copy(), best_move)
                if not self.timeout:
                    moves = temp_moves
                    best_move = move
            else:
                break

        return best_move

    def search_best_next_move(self, initial_tree_node, depth):
        """
        Use minimax with alphabeta pruning to find the best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :param depth: depth of the search
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """
        # initialize timer
        self.elapsed_time = 0
        self.start_time = time.time()

        children = initial_tree_node.compute_and_get_children()
        # check if a fish was caught and the only option is up -> return it
        if len(children) == 1:
            return ACTION_TO_STR[children[0].move]

        # initialize the values for moves, alpha and beta
        moves = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
        alpha, beta = -np.inf, np.inf
        for child in initial_tree_node.children:
            # if there is time, search the next child, else stop the search
            self.elapsed_time = time.time() - self.start_time
            if self.elapsed_time < self.time_limit:
                value = self.alphabeta(child, depth - 1, alpha, beta, 1)
                moves[child.move] = value
                alpha = np.max([value, alpha])
            else:
                break

        highest_indices = np.where(moves == np.max(moves))[0]
        if len(highest_indices) >= 2:
            return ACTION_TO_STR[random.choice(highest_indices)]
        else:
            return ACTION_TO_STR[np.argmax(moves)]

    def search_at_depth(self, initial_tree_node, depth, old_moves, move):
        """
        Use alphabeta for certain depth to find the best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :param depth: depth of the search
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """
        children = initial_tree_node.compute_and_get_children()
        # check if a fish was caught and the only option is up -> return it
        if len(children) == 1:
            return ACTION_TO_STR[children[0].move], old_moves

        children = self.sort_children(initial_tree_node, old_moves)  # more options than up
        moves = [0, 0, 0, 0, 0]  # keep track of old moves
        alpha, beta = -np.inf, np.inf
        for child in children:
            # if there is time, search the next child, else stop the search
            self.elapsed_time = time.time() - self.start_time
            if self.elapsed_time < self.time_limit:
                value = self.alphabeta(child, depth - 1, alpha, beta, 1)
                moves[child.move] = value
                alpha = np.max([value, alpha])
            else:
                self.timeout = True  # keep track of when we have timed out
                break

        highest_indices = np.where(moves == np.max(moves))[0]
        # if there are many possible best moves, and we haven't decided on a move before, pick randomly
        if len(highest_indices) >= 2 and move is None:
            return ACTION_TO_STR[random.choice(highest_indices)], moves
        # if there are multiple positions giving the same score but a search with a lower depth had only one best move
        # stay with that
        elif len(highest_indices) >= 2 and move is not None:
            return move, old_moves
        else:
            return ACTION_TO_STR[np.argmax(moves)], moves

    def alphabeta(self, node, depth, alpha, beta, player):
        """
        Use minimax with alpha beta pruning to search the value of the state
        :param node: game tree node
        :param depth: depth of the search
        :param alpha: lower bound of the search
        :param beta: upper bound of the search
        :param player: player currently playing
        :return v: value for the node
        """
        # check if there is still have time to explore, if not -> return
        self.elapsed_time = time.time() - self.start_time
        if self.elapsed_time >= self.time_limit:
            self.timeout = True if depth != 0 else False
            return self.heuristic(node)

        # not at maximum depth -> expand children
        if depth != 0:
            _ = node.compute_and_get_children()

        # at maximum depth or terminal state
        if depth == 0 or len(node.children) == 0:
            return self.heuristic(node)

        # perform alpha beta minimax search
        elif player == 0:
            v = - np.inf
            children = self.sort_children_heuristic(node, 0)
            for child in children:
                game_state = self.hashing(child)
                if game_state in self.transposition_table.keys() and self.transposition_table[game_state][1] < child.depth:
                    v = self.transposition_table[game_state][0]
                else:
                    v = np.max([v, self.alphabeta(child, depth - 1, alpha, beta, 1)])
                alpha = np.max([v, alpha])
                if beta <= alpha:
                    break
        else:
            v = np.inf
            children = self.sort_children_heuristic(node, 1)
            for child in children:
                game_state = self.hashing(child)
                if game_state in self.transposition_table.keys() and self.transposition_table[game_state][1] < child.depth:
                    v = self.transposition_table[game_state][0]
                else:
                    v = np.min([v, self.alphabeta(child, depth - 1, alpha, beta, 0)])
                beta = np.min([v, beta])
                if beta <= alpha:
                    break
        return v

    def heuristic(self, node):
        """
        Calculate the value of the state as an approximation (using heuristics)
        :param node: a game tree node
        :return result: evaluation of the state based on heuristic
        """
        state = node.state
        fish_positions = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        hook_positions = state.get_hook_positions()
        current_score = self.utility(node)
        fish_distance_p0, fish_distance_p1 = 0, 0
        caught_fish_p0 = 0

        # Weights found with parameter tuning
        score_w = 100
        fish_w = 10
        caught_w = 1000

        # Calculate the distance of fish to the hook for both players
        for fish_type, fish_pos in fish_positions.items():
            distance_p0_x_round_world = 20 - hook_positions[0][0] + fish_pos[0]
            distance_p0_x = abs(fish_pos[0] - hook_positions[0][0])
            distance_p1_x_round_world = 20 - hook_positions[1][0] + fish_pos[0]
            distance_p1_x = abs(fish_pos[0] - hook_positions[1][0])

            # Check for shortest distance: straight or around the map
            if distance_p0_x_round_world > distance_p0_x:
                # The fish is reachable for p_0 faster by going straight
                distance_p0 = distance_p0_x + abs(fish_pos[1] - hook_positions[0][1])
                alternative_distance_p0 = distance_p0_x_round_world + abs(fish_pos[1] - hook_positions[0][1])
                round_world_p0 = False
            else:
                # The fish is reachable faster by going around the world
                distance_p0 = distance_p0_x_round_world + abs(fish_pos[1] - hook_positions[0][1])
                alternative_distance_p0 = distance_p0_x + abs(fish_pos[1] - hook_positions[0][1])
                round_world_p0 = True

            # Check for shortest distance: straight or around the map
            if distance_p1_x_round_world > distance_p1_x:
                # The fish is reachable for p_1 faster by going straight
                distance_p1 = distance_p1_x + abs(fish_pos[1] - hook_positions[1][1])
            else:
                # The fish is reachable faster for p_1 by going around the world
                distance_p1 = distance_p1_x_round_world + abs(fish_pos[1] - hook_positions[1][1])

            if distance_p0 == 0:
                # If fish is caught
                caught_fish_p0 = fish_scores[fish_type]
            else:
                if distance_p1 != 0:
                    # If fish is not caught by p_1
                    if self.boat_blocking_path(fish_pos, hook_positions, 0, round_world_p0):
                        fish_distance_p0 += fish_scores[fish_type] / alternative_distance_p0
                    else:
                        fish_distance_p0 += fish_scores[fish_type] / distance_p0

        # check terminal state win-lose conditions and update value of state accordingly
        if len(fish_positions.keys()) == 0 and current_score > 0:
            result = np.inf
        elif len(fish_positions.keys()) == 0 and current_score < 0:
            result = -np.inf
        else:
            result = score_w * current_score + fish_w * fish_distance_p0 + caught_w * caught_fish_p0
        self.add_to_transposition(node, result)
        return result

    def boat_blocking_path(self, fish_pos, hook_positions, player_id, round_world):
        """
        Check if the opponent is blocking the shortest path that player wants to take
        :param fish_pos: positions of the fishes
        :param hook_positions: positions of the hooks (players)
        :param player_id: player playing
        :param round_world: boolean flag, tells whether the shortest path is to go around the map
        """
        player_x = hook_positions[player_id][0]
        opponent_x = hook_positions[1 - player_id][0]
        if not round_world:
            if player_x > opponent_x:
                return fish_pos[0] <= opponent_x
            else:
                return fish_pos[0] >= opponent_x
        else:
            if player_x > opponent_x:
                return fish_pos[0] >= opponent_x
            else:
                return fish_pos[0] <= opponent_x

    def utility(self, node):
        """
        Calculate the difference in scores of the players
        :param node: game tree node
        :return difference of scores
        """
        scores = node.state.get_player_scores()
        return scores[0] - scores[1]

    def hashing(self, node):
        """
        Given a state, return a unique key that represents the state of the map and the players and fishes on it
        :param node: game tree node
        :return state key
        """
        player_positions = node.state.get_hook_positions()
        fish_positions = node.state.get_fish_positions()
        state_key = f'{player_positions[0]}G_{player_positions[0]}R_{fish_positions}F'
        return state_key

    def sort_children(self, node, moves):
        """
        Sort a list of children for MAX player based on previously done searches
        :param node: game tree node
        :param moves: list of values from search with lower depth
        :return sorted children of the node based on their values
        """
        indices = np.argsort(-np.array(moves))
        return [node.children[i] for i in indices]

    def sort_children_heuristic(self, node, player):
        """
        Sort a list of children based on heuristic and the player (ascending or descending)
        :param node: game tree node
        :param player: 0=MAX or 1=MIN
        :return sorted children of the node based on their values
        """
        if player == 0:
            return sorted(node.children, key=lambda child: self.heuristic(child), reverse=True)
        return sorted(node.children, key=lambda child: self.heuristic(child))

    def add_to_transposition(self, node, value):
        """
        Add state, value and depth to the transposition table that keeps track of the visited states
        :param node: game tree node
        :param value: of the game
        :return
        """
        state_key = self.hashing(node)
        if state_key in self.transposition_table.keys() and self.transposition_table[state_key][0] < value:
            self.transposition_table[state_key][0] = value
            self.transposition_table[state_key][1] = node.depth
        else:
            self.transposition_table[state_key] = [value, node.depth]
        return

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
        self.time_limit = 70 * 1e-3  # in seconds
        self.random_values_M = None
        self.transposition_table = {}

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
            max_depth = 4
            best_move = self.search_best_next_move(initial_tree_node=node, depth=max_depth)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node, depth):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :param depth: depth of the search
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!
        self.elapsed_time = 0
        self.start_time = time.time()
        _ = initial_tree_node.compute_and_get_children()
        # children = sorted(initial_tree_node.children, key=lambda child: self.heuristic_simple(child))
        moves = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
        alpha, beta = -np.inf, np.inf
        for i, child in enumerate(initial_tree_node.children):
            self.elapsed_time = time.time() - self.start_time
            if self.elapsed_time < self.time_limit:
                value = self.alphabeta(child, depth - 1, alpha, beta, 1)
                moves[i] = value
                alpha = np.max([value, alpha])
            else:
                break
        # print(moves)
        highest_indices = np.where(moves == np.max(moves))[0]
        if len(highest_indices) >= 2:
            return ACTION_TO_STR[random.choice(highest_indices)]
        else:
            return ACTION_TO_STR[np.argmax(moves)]

    def search_best_next_move_ids(self, initial_tree_node, depth):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :param depth: depth of the search
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """
        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!
        self.elapsed_time = 0
        self.start_time = time.time()
        _ = initial_tree_node.compute_and_get_children()
        moves = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
        best_move = None
        for max_depth in range(1, depth+1):
            print('IDS=', max_depth)
            self.elapsed_time = time.time() - self.start_time
            if self.elapsed_time < self.time_limit:
                move = self.search_best_next_move(initial_tree_node, max_depth)
                best_move = move
                # alpha, beta = -np.inf, np.inf
                # children = self.sort_children(initial_tree_node, moves)
                # print(moves)
                # print([node.move for node in children])
                # for i, child in enumerate(children):
                #     value = self.alphabeta(child, max_depth - 1, alpha, beta, 1)
                #     moves[i] = value
                #     alpha = np.max([value, alpha])
                #
                # highest_indices = np.where(moves == np.max(moves))[0]
                # if len(highest_indices) >= 2:
                #     best_move = ACTION_TO_STR[random.choice(highest_indices)]
                # else:
                #     best_move = ACTION_TO_STR[np.argmax(moves)]
            else:
                break

        #print(moves)
        return best_move

    def alphabeta(self, node, depth, alpha, beta, player):

        self.elapsed_time = time.time() - self.start_time
        if self.elapsed_time >= self.time_limit:
            # self.transposition_table[game_state] = (score, node.move)
            return self.heuristic_simple(node)

        if depth != 0:
            _ = node.compute_and_get_children()

        if depth == 0 or len(node.children) == 0:
            return self.heuristic_simple(node)

        elif player == 0:
            v = - np.inf
            for child in node.children:
                v = np.max([v, self.alphabeta(child, depth - 1, alpha, beta, 1)])
                alpha = np.max([v, alpha])
                if beta <= alpha:
                    break
        else:
            v = np.inf
            for child in node.children:
                v = np.min([v, self.alphabeta(child, depth - 1, alpha, beta, 0)])
                beta = np.min([v, beta])
                if beta <= alpha:
                    break
        return v

    def heuristic_simple(self, node):

        # state_key = self.hash_zobrist(node)
        # if state_key in self.transposition_table.keys():
        #     return self.transposition_table[state_key][0]
        state = node.state
        fish_positions = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        hook_positions = state.get_hook_positions()

        current_score = self.heuristic(node)

        fish_distance_p0, fish_distance_p1 = 0, 0
        caught_fish_p0 = 0

        # Calculate the proximity of fish to the hook for both players
        for fish_type, fish_pos in fish_positions.items():
            distance_p0_x_round_world = 20 - hook_positions[0][0] + fish_pos[0]
            distance_p0_x = abs(fish_pos[0] - hook_positions[0][0])
            distance_p1_x_round_world = 20 - hook_positions[1][0] + fish_pos[0]
            distance_p1_x = abs(fish_pos[0] - hook_positions[1][0])

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

        result = current_score + fish_distance_p0 + caught_fish_p0
        return result

    def boat_blocking_path(self, fish_pos, hook_positions, player_id, round_world):
        player_x = hook_positions[player_id][0]
        opponent_x = hook_positions[1-player_id][0]
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

    def heuristic(self, node):
        scores = node.state.get_player_scores()
        return scores[0] - scores[1]

    def init_zobrist(self):
        np.random.seed(42)
        self.random_values_M = np.random.randint(64**2, size=(20, 20))

    def hash_zobrist(self, node):
        fish_scores = node.state.get_fish_scores()
        player_positions = node.state.get_hook_positions()
        fish_positions = node.state.get_fish_positions()
        p0, p1 = 1, 2

        state_key = self.random_values_M[player_positions[0][0]][player_positions[0][1]] * p0
        state_key += self.random_values_M[player_positions[1][0]][player_positions[1][1]] * p1
        for fish_type, fish_pos in fish_positions.items():
            state_key += self.random_values_M[fish_pos[0]][fish_pos[1]] * fish_scores[fish_type]
        return state_key

    def sort_children(self, node, moves):
        indices = np.argsort(-np.array(moves))
        return [node.children[i] for i in indices]


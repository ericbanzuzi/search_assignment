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
        self.time_limit = 70*1e-3  # in seconds
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
            max_depth = 8
            best_move = self.search_best_next_move(node=node, max_depth=max_depth)
            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, node, max_depth):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param node: Initial game tree node
        :param max_depth: Maximum depth for search
        :type node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """
        self.elapsed_time = 0
        self.init_zobrist()
        self.transposition_table = {}
        self.start_time = time.time()
        prev_best_value, prev_best_move, best_move = 0, 0, 0
        # for i in range(1, max_depth+1):
        #     print('Iterative stuff:', i)
        #     self.elapsed_time = time.time() - self.start_time
        #     if self.elapsed_time < self.time_limit:
        #         best_value, best_move = self.alpha_beta_search(node, max_depth, -np.inf, np.inf, 0)
        #         # if best_value < prev_best_value:
        #         #     best_value = prev_best_value
        #         #     best_move = prev_best_move
        #         # else:
        #         #     prev_best_move = best_move
        #         #     prev_best_value = best_value
        best_value, best_move = self.alpha_beta_search(node, max_depth, -np.inf, np.inf, 0)
        print('Move:', ACTION_TO_STR[best_move], 'Score:', best_value)
        return ACTION_TO_STR[best_move]

    def alpha_beta_search(self, node, max_depth, alpha, beta, player):
        # print('At depth=', node.depth)
        # if node.depth == 5:
        #     print('Score at depth=5:', self.heuristic_simple(node))
        # game_state = self.hash_zobrist(node)
        # if game_state in self.transposition_table.keys():
        #     return self.transposition_table[game_state][0], self.transposition_table[game_state][1]

        self.elapsed_time = time.time() - self.start_time
        if self.elapsed_time >= self.time_limit:
            score = self.heuristic_simple(node)
            # self.transposition_table[game_state] = (score, node.move)
            return score, node.move

        if max_depth != 0:
            _ = node.compute_and_get_children()
            # node.children.reverse()
            # node.children = self.sort_children(node, player)

        if max_depth == 0 or (len(node.children) == 0 and node.parent is not None):  # max depth or terminal
            score = self.heuristic_simple(node)
            # self.transposition_table[game_state] = (score, node.move if node.move is not None else 0)
            return score, node.move

        # children = node.compute_and_get_children()
        if player == 0:
            v, move = -np.inf, np.random.randint(0, 5)
            for child in node.children:
                next_v, next_move = self.alpha_beta_search(child, max_depth-1, alpha, beta, 1)
                if next_v > v:
                    v = next_v
                    move = next_move
                alpha = max([alpha, v])
                if beta <= alpha:
                    break
        else:
            v, move = np.inf, np.random.randint(0, 5)
            for child in node.children:
                next_v, next_move = self.alpha_beta_search(child, max_depth-1, alpha, beta, 0)
                if next_v < v:
                    v = next_v
                    move = next_move
                beta = min([beta, v])
                if beta <= alpha:
                    break
        # self.transposition_table[game_state] = (v, move)
        return v, move

    def heuristic_simple(self, node):

        # state_key = self.hash_zobrist(node)
        # if state_key in self.transposition_table.keys():
        #     return self.transposition_table[state_key][0]

        state = node.state
        fish_positions = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        hook_positions = state.get_hook_positions()
        # TODO: tune
        catch_weight = 1
        score_weight = 1
        distance_weight = 1

        current_score = self.utility_function(node)

        fish_distance_p0, fish_distance_p1 = 0, 0
        caught_fish = 0
        caught_fish_p1 = 0
        distance_p0, distance_p1 = 0, 0
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
                caught_fish = fish_scores[fish_type]
            else:
                if distance_p1 != 0:
                    # If fish is not caught by p_1
                    if self.boat_blocking_path(fish_pos, hook_positions, 0, round_world_p0):
                        fish_distance_p0 += fish_scores[fish_type] / alternative_distance_p0
                    else:
                        fish_distance_p0 += fish_scores[fish_type] / distance_p0

            if distance_p1 == 0:
                # If fish is caught
                caught_fish_p1 = fish_scores[fish_type]

        if current_score > 0 and (len(fish_positions.keys()) == 0 or (len(fish_positions.keys()) == 1 and distance_p0 == 0)):
            temp = current_score + caught_fish
            if temp > 0:
                result = np.inf  # win
                print('-----------------  WON: ---------------------')
                print(ACTION_TO_STR[node.move], 1-node.state.player)
                parent = node.parent
                while parent is not None:
                    move = parent.move
                    if move is not None:
                        print(ACTION_TO_STR[parent.move],  1-parent.state.player)
                    parent = parent.parent
            else:
                result = -np.inf
                print('LOST')

        elif current_score < 0 and (len(fish_positions.keys()) == 0 or (len(fish_positions.keys()) == 1 and distance_p1 == 0)):
            temp = current_score - caught_fish_p1
            if temp < 0:
                result = -np.inf
                print('LOST')
            else:
                result = np.inf  # win
                print('-----------------  WON: ---------------------')
                print(ACTION_TO_STR[node.move], 1 - node.state.player)
                parent = node.parent
                while parent is not None:
                    move = parent.move
                    if move is not None:
                        print(ACTION_TO_STR[parent.move], 1 - parent.state.player)
                    parent = parent.parent
        else:
            result = score_weight*current_score + distance_weight*fish_distance_p0 + catch_weight*caught_fish
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

    def utility_function(self, node):
        # TODO: change to take into consideration winning and loosing states
        scores = node.state.get_player_scores()
        return scores[0] - scores[1]

    def sort_children_old(self, node, player):
        if player == 0:
            sorted_children = sorted(node.children, key=lambda child: self.heuristic_simple(child), reverse=True)
        else:
            sorted_children = sorted(node.children, key=lambda child: self.heuristic_simple(child), reverse=False)
        return sorted_children

    def sort_children(self, node, player):
        if node.depth == 0 or len(self.transposition_table) == 0:
            sorted_children = sorted(node.children, key=lambda child: self.heuristic_simple(child), reverse=True)
        else:
            game_state = self.hash_zobrist(node)
            if game_state == 21804:
                print(node.depth)
                print(node.children)
                print(node.move)
            print(self.transposition_table)
            if player == 0:
                sorted_children = sorted(node.children, key=lambda child: self.transposition_table[game_state][0], reverse=True)
            else:
                sorted_children = sorted(node.children, key=lambda child: self.transposition_table[game_state][0], reverse=False)
        return sorted_children

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

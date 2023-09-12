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
        self.time_limit = 68*1e-3  # in seconds

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
            max_depth = 2
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
        self.start_time = time.time()
        best_move = 0
        best_value = -np.inf
        # prev_best_move = 0
        # for depth in range(7, max_depth):
        #     # print('Elapsed:', self.elapsed_time)
        #     self.elapsed_time = time.time() - self.start_time
        #     # print('Elapsed:', self.elapsed_time)
        #     if self.elapsed_time >= self.time_limit:
        #         # print('STOPPED HERE IN LOOP')
        #         break
        #     else:
        #         best_value, best_move = self.alpha_beta_search(node, depth, -np.inf, np.inf, True)
        #         best_move = best_move if best_move is not None else prev_best_move  # get the prior best if we don't manage to compute
        best_value, best_move = self.alpha_beta_search(node, max_depth, -np.inf, np.inf)
        print('Best............', ACTION_TO_STR[best_move])
        print('Utility.........', best_value)
        # print('Time taken:', time.time() - self.start_time)
        return ACTION_TO_STR[best_move]

    def alpha_beta_search(self, node, max_depth, alpha, beta):

        self.elapsed_time = time.time() - self.start_time
        if self.elapsed_time >= self.time_limit:
            # print('---------------- STOPPED HERE BECAUSE OF TIME --------------------')
            # print(f'At depth={node.depth} From MOVE by P1',
            #       ACTION_TO_STR[node.move] if node.move is not None else node.move)
            # print('Next, unsorted P0 at DEPTH', node.depth + 1)
            # print('SCORE WAS: ', self.eval_function(node))
            return self.eval_function(node), node.move

        if max_depth == 0 or (len(node.children) == 0 and node.parent is not None):  # max state or terminal
            # print(f'At depth={node.depth} From MOVE by P1',
            #       ACTION_TO_STR[node.move] if node.move is not None else node.move)
            # print('Next, unsorted P0 at DEPTH', node.depth + 1)
            return self.eval_function(node), node.move if node.move is not None else 0

        children = node.compute_and_get_children()
        if node.state.player == 0:
            # print(f'At depth={node.depth} From MOVE by P1', ACTION_TO_STR[node.move] if node.move is not None else node.move)
            # print('Next, unsorted P0 at DEPTH', node.depth+1)
            # self.visualize_scores(children, 0)
            v, move = -np.inf, np.random.randint(0, 5)
            for child in children:
                next_v, next_move = self.alpha_beta_search(child, max_depth-1, alpha, beta)
                if next_v > v:
                    v = next_v
                    move = next_move
                alpha = max([alpha, v])
                if beta <= alpha:
                    break
        else:
            # print(f'At depth={node.depth} From MOVE by P0', ACTION_TO_STR[node.move] if node.move is not None else node.move)
            # print('Next, unsorted P1 at DEPTH', node.depth+1)
            # self.visualize_scores(children, 1)
            v, move = np.inf, np.random.randint(0, 5)
            for child in children:
                next_v, next_move = self.alpha_beta_search(child, max_depth-1, alpha, beta)
                if next_v < v:
                    v = next_v
                    move = next_move
                beta = min([beta, v])
                if beta <= alpha:
                    break
        return v, move

    def minimax_decision(self, node, max_depth):
        v, action = -np.inf, None
        children = node.compute_and_get_children()
        for child in children:
            result = self.min_value(child, max_depth)
            if result > v:
                v = result
                action = child
        return action.move

    def max_value(self, node, max_depth):
        # check terminal state
        children = node.compute_and_get_children()
        if len(children) == 0 or node.depth == max_depth:
            return self.eval_function(node)
        v = -np.inf
        for child in children:
            v = max([v, self.min_value(child, max_depth)])
        return v

    def min_value(self, node, max_depth):
        # check terminal state
        children = node.compute_and_get_children()
        if len(children) == 0 or node.depth == max_depth:
            return self.eval_function(node)
        v = np.inf
        for child in children:
            v = min([v, self.max_value(child, max_depth)])
        return v

    def eval_function(self, node):
        state = node.state
        fish_positions = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        hook_positions = state.get_hook_positions()
        fish_weight_p0 = 1
        fish_weight_p1 = 1
        score_weight = 1
        illegal_weight = 0

        current_score = self.utility_function(node)

        illegal_flag = 1 if hook_positions[0][0] == hook_positions[1][0] else 0
        illegal_flag = illegal_flag if (1-node.state.player) == 0 else -illegal_flag

        fish_distance_p0, fish_distance_p1 = 0, 0
        # Calculate the proximity of fish to the hook for both players
        for fish_type, fish_pos in fish_positions.items():
            distance_p0_x_round_world = 20 - hook_positions[0][0] + fish_pos[0]
            distance_p0_x = abs(fish_pos[0] - hook_positions[0][0])
            distance_p1_x_round_world = 20 - hook_positions[1][0] + fish_pos[0]
            distance_p1_x = abs(fish_pos[0] - hook_positions[1][0])

            if distance_p0_x_round_world > distance_p0_x:
                distance_p0 = distance_p0_x + abs(fish_pos[1] - hook_positions[0][1])
                alternative_distance_p0 = distance_p0_x_round_world + abs(fish_pos[1] - hook_positions[0][1])
                round_world_p0 = False
            else:
                distance_p0 = distance_p0_x_round_world + abs(fish_pos[1] - hook_positions[0][1])
                alternative_distance_p0 = distance_p0_x + abs(fish_pos[1] - hook_positions[0][1])
                round_world_p0 = True

            if distance_p1_x_round_world > distance_p1_x:
                distance_p1 = distance_p1_x + abs(fish_pos[1] - hook_positions[1][1])
                alternative_distance_p1 = distance_p1_x_round_world + abs(fish_pos[1] - hook_positions[1][1])
                round_world_p1 = False
            else:
                distance_p1 = distance_p1_x_round_world + abs(fish_pos[1] - hook_positions[1][1])
                alternative_distance_p1 = distance_p1_x + abs(fish_pos[1] - hook_positions[1][1])
                round_world_p1 = True

            if distance_p0 == 0:
                fish_distance_p0 += fish_scores[fish_type] - abs(19-fish_pos[1])
            else:
                if distance_p1 != 0:  # P1 did not catch a fish
                    if self.boat_blocking_path(fish_pos, hook_positions, 0, round_world_p0):
                        fish_distance_p0 += fish_scores[fish_type] / alternative_distance_p0
                    else:
                        fish_distance_p0 += fish_scores[fish_type] / distance_p0
                else:
                    # TODO: TUNE THIS SO THAT RIGHT IS PREFERRED OVER STAY/UP
                    fish_distance_p0 -= 0.7*fish_scores[fish_type] / abs(19-fish_pos[1])

            if distance_p1 == 0:
                fish_distance_p1 += fish_scores[fish_type] - abs(19-fish_pos[1])
            else:
                if distance_p0 != 0:  # P0 did not catch a fish
                    if self.boat_blocking_path(fish_pos, hook_positions, 1, round_world_p1):
                        fish_distance_p1 += fish_scores[fish_type] / alternative_distance_p1
                    else:
                        fish_distance_p1 += fish_scores[fish_type] / distance_p1
                else:
                    # TODO: TUNE THIS SO THAT RIGHT IS PREFERRED OVER STAY/UP
                    fish_distance_p1 -= 0.7*fish_scores[fish_type] / abs(19-fish_pos[1])

        result = (score_weight*current_score + fish_weight_p0*fish_distance_p0 - fish_weight_p1*fish_distance_p1 -
                  illegal_weight*illegal_flag)
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
        scores = node.state.get_player_scores()
        return scores[0] - scores[1]

    def sort_children(self, children, player):
        if player == 0:
            sorted_children = sorted(children, key=lambda child: self.eval_function(child), reverse=True)
        else:
            sorted_children = sorted(children, key=lambda child: self.eval_function(child), reverse=False)
        return sorted_children

    def visualize_scores(self, nodes, player):
        score = [self.utility_function(child) for child in nodes]
        values = [self.eval_function(child) for child in nodes]
        print('Scores:', score)
        print('Player:', player, values)
        return

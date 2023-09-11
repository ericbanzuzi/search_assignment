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
        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###
        # random_move = random.randrange(5)
        # return ACTION_TO_STR[random_move]
        self.elapsed_time = 0
        self.start_time = time.time()
        best_move = 0
        best_value = -np.inf
        prev_best_move = 0
        for depth in range(7, max_depth):
            # print('Elapsed:', self.elapsed_time)
            self.elapsed_time = time.time() - self.start_time
            # print('Elapsed:', self.elapsed_time)
            if self.elapsed_time >= self.time_limit:
                # print('STOPPED HERE IN LOOP')
                break
            else:
                best_value, best_move = self.alpha_beta_search(node, depth, -np.inf, np.inf, True)
                best_move = best_move if best_move is not None else prev_best_move
        # best_value, best_move = self.alpha_beta_search(node, max_depth, -np.inf, np.inf)
        print('Best............', ACTION_TO_STR[best_move])
        print('Utility.........', best_value)
        # print('Time taken:', time.time() - self.start_time)
        return ACTION_TO_STR[best_move]

    def alpha_beta_search(self, node, max_depth, alpha, beta, print_val):

        self.elapsed_time = time.time() - self.start_time
        if self.elapsed_time >= self.time_limit:
            # print('STOPPED HERE')
            if print:
                print('DEPTH............', max_depth)
                print('PLAYER...........', 1-node.state.player)
                print('MOVE.............', node.move)
                print('SCORE............', self.eval_function(node))
            return self.eval_function(node), node.move

        children = node.compute_and_get_children()
        if max_depth == 0 or len(children) == 0:
            if print:
                print('DEPTH............', max_depth)
                print('PLAYER...........', 1-node.state.player)
                print('MOVE.............', node.move)
                print('SCORE............', self.eval_function(node))
            return self.eval_function(node), node.move if node.move is not None else 0

        if node.state.player == 0:
            # print('Unsorted P0:')
            # self.visualize_scores(children, 0)
            # children = self.sort_children(children, 0)
            # print('Sorted P0:')
            # self.visualize_scores(children, 0)
            v, move = -np.inf, np.random.randint(0, 5)
            for child in children:
                # next_v, next_move = self.alpha_beta_search(child, max_depth-1, alpha, beta)
                if (child.move == 4 and node.depth == 0) or (node.depth == 0 and child.move == 2):
                    next_v, next_move = self.alpha_beta_search(child, max_depth - 1, alpha, beta, True)
                    print('CALLED Ppppppppkknnnksnkksnkksnknsknfnflanal CALLED Ppppppppkknnnksnkksnkksnknsknfnflanal CALLED Ppppppppkknnnksnkksnkksnknsknfnflanal CALLED Ppppppppkknnnksnkksnkksnknsknfnflanal')
                else:
                    next_v, next_move = self.alpha_beta_search(child, max_depth - 1, alpha, beta, False)
                if next_v > v:
                    # print('New alpha:', next_v)
                    v = next_v
                    move = next_move
                alpha = max([alpha, v])
                if beta <= alpha:
                    break
        else:
            # print('Unsorted P1:')
            # self.visualize_scores(children, 1)
            # children = self.sort_children(children, 1)
            # print('Sorted P1:')
            # self.visualize_scores(children, 1)
            v, move = np.inf, np.random.randint(0, 5)
            for child in children:
                # next_v, next_move = self.alpha_beta_search(child, max_depth-1, alpha, beta)
                next_v, next_move = self.alpha_beta_search(child, max_depth - 1, alpha, beta, print_val)
                if next_v < v:
                    # print('New beta:', next_v)
                    v = next_v
                    move = next_move
                beta = min([beta, v])
                if beta <= alpha:
                    break
        # print('DEPTH............', max_depth)
        # print('PLAYER...........', 1-node.state.player)
        # print('MOVE.............', node.move)
        # print('SCORE............', self.eval_function(node))
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
        illegal_weight = 10
        # caught_fish_p0 = fish_scores[caught_fish_type[0]] if caught_fish_type[0] is not None else 0
        # caught_fish_p0_distance = 19-hook_positions[0][1]
        # if caught_fish_type[0] is not None:
        #     fish_weight_p0 = 0.4
        # caught_fish_p1 = fish_scores[caught_fish_type[1]] if caught_fish_type[1] is not None else 0
        # caught_fish_p1_distance = 19 - hook_positions[1][1]
        # if caught_fish_type[1] is not None:
        #     fish_weight_p1 = 0.4

        current_score = self.utility_function(node)

        illegal_flag = 1 if hook_positions[0][0] == hook_positions[1][0] else 0
        illegal_flag = illegal_flag if (1-node.state.player) == 0 else -illegal_flag

        # print('MOVE', node.move)
        # print('SCORE', current_score)
        fish_distance_p0, fish_distance_p1 = 0, 0
        # Calculate the proximity of fish to the hook for both players
        for fish_type, fish_pos in fish_positions.items():
            distance_p0_x_round_world = 19 - hook_positions[0][0] + fish_pos[0]
            distance_p0_x = abs(fish_pos[0] - hook_positions[0][0])
            distance_p1_x_round_world = 19 - hook_positions[1][0] + fish_pos[0]
            distance_p1_x = abs(fish_pos[0] - hook_positions[1][0])

            if distance_p0_x_round_world > distance_p0_x:
                distance_p0 = abs(fish_pos[0] - hook_positions[0][0]) + abs(fish_pos[1] - hook_positions[0][1])
            else:
                distance_p0 = distance_p0_x_round_world + abs(fish_pos[1] - hook_positions[0][1])

            if distance_p1_x_round_world > distance_p1_x:
                distance_p1 = abs(fish_pos[0] - hook_positions[1][0]) + abs(fish_pos[1] - hook_positions[1][1])
            else:
                distance_p1 = distance_p1_x_round_world + abs(fish_pos[1] - hook_positions[1][1])

            # print('DISTANCE P0', distance_p0)
            # distance_p1 = abs(fish_pos[0] - hook_positions[1][0]) + abs(fish_pos[1] - hook_positions[1][1])
            # print('DISTANCE P1', distance_p1)
            # distance_p0 = np.sqrt((fish_pos[0] - hook_positions[0][0])**2 + (fish_pos[1] - hook_positions[0][1])**2)
            # distance_p1 = np.sqrt((fish_pos[0] - hook_positions[1][0])**2 + (fish_pos[1] - hook_positions[1][1])**2)
            # fish_distance_p0 += fish_scores[fish_type] / (distance_p0 + 1)  # Avoid division by zero, if fish caught give full points
            # fish_distance_p1 += fish_scores[fish_type] / (distance_p1 + 1)  # Avoid division by zero, if fish caught give full points
            if distance_p0 == 0:
                # print('CAUGHT FISH P0 IN HOOK')
                fish_distance_p0 += 1.1*fish_scores[fish_type] / (distance_p0 + 1)
            else:
                fish_distance_p0 += fish_scores[fish_type] / distance_p0
            if distance_p1 == 0:
                # print('CAUGHT FISH P1 IN HOOK')
                fish_distance_p1 += 1.1*fish_scores[fish_type] / (distance_p1 + 1)
            else:
                fish_distance_p1 += fish_scores[fish_type] / distance_p1

        # print('Playing.......', node.state.player)
        # print('p0 fish:', fish_distance_p0)
        # print('p1 fish:', fish_distance_p1)
        result = (score_weight*current_score + fish_weight_p0*fish_distance_p0 - fish_weight_p1*fish_distance_p1 -
                  illegal_weight*illegal_flag)

        # caught_weight*(caught_fish_p0/(caught_fish_p0_distance+1)) - caught_weight*(caught_fish_p1/(caught_fish_p1_distance+1)))
        # print('total evaluation:', result)
        return result

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
        values = [self.eval_function(child) for child in nodes]
        print('Player:', player, values)
        return

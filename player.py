#!/usr/bin/env python3
import random
import numpy as np

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
            best_move = self.search_best_next_move(node=node)
            print('Best............', best_move)
            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param node: Initial game tree node
        :type node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """
        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###
        # random_move = random.randrange(5)
        # return ACTION_TO_STR[random_move]
        max_depth = 5
        v, move = self.alpha_beta_search(node, max_depth, -np.inf, np.inf)
        return ACTION_TO_STR[move]

    def alpha_beta_search(self, node, max_depth, alpha, beta):

        children = node.compute_and_get_children()
        if max_depth == 0 or len(children) == 0:
            return self.eval_function(node), node.move

        if node.state.player == 0:
            v, move = -np.inf, None
            for child in children:
                next_v, next_move = self.alpha_beta_search(child, max_depth-1, alpha, beta)
                if next_v > v:
                    v = next_v
                    move = next_move
                alpha = max([alpha, v])
                if beta <= alpha:
                    break
        else:
            v, move = np.inf, None
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
        scores = node.state.get_player_scores()
        return scores[0] - scores[1]

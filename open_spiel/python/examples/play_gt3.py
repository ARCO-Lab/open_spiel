'''This is where you will test your algorithm against other players'''

import random
import numpy as np

from open_spiel.python.games import gt3
from open_spiel.python.bots import human

def playGT3(board_size, win_condition):
    game = gt3.GT3Game({"board_size": board_size, "win_condition": win_condition})
    state = game.new_initial_state()
    # print(state) # display board
    while not state.is_terminal():
        # print("Player {}'s turn".format(state.current_player()))
        # print("Legal moves:", state.legal_actions())
        action = np.random.choice(state.legal_actions())
        state.apply_action(action) # make move
        # print(state, '\n')
    
    print(state)
    # print("Returns: {}".format(state.returns()))
    if state.returns()[0] == 1:
       print("x wins!\n")
       return 0
    elif state.returns()[1] == 1:
       print("o wins!\n")
       return 1
    else:
       print("This game is a tie!\n")
       return -1

if __name__ == "__main__":
    # Get user inputs
    while True:
        try:
            size = int(input('Enter board size: '))
            if size < 1:
                raise ValueError
            break
        except ValueError:
            print('Please enter a positive integer')

    while True:
        try:
            win_cond = int(input('Enter win condition (number of x or o to connect for a win): '))
            if win_cond < 1 or win_cond > size:
                raise ValueError
            break
        except ValueError:
            print('Please enter a positive integer smaller than the board size')

    while True:
        try:
            gens = int(input('Enter number of generations to play: '))
            if gens < 1:
                raise ValueError
            break
        except ValueError:
            print('Please enter a positive integer')
    
    player_0_wins = 0
    player_1_wins = 0
    draws = 0

    for i in range(gens):
        winner = playGT3(size, win_cond)
        if winner == 0:
            player_0_wins += 1
        elif winner == 1:
            player_1_wins += 1
        else:
            draws += 1
    
    print("\nGame Over!\n")
    print("Number of x wins:", player_0_wins)
    print("Number of o wins:", player_1_wins)
    print("Draws:", draws)

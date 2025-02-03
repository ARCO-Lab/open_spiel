'''This is where you will test your algorithm against other players'''

import sys
import numpy as np

from open_spiel.python.games import gt3
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import my_dqn

BOT_1_TRAINED_MODEL_PATH = 'my_models/trained_models/trained_mlp.pth'
BOT_1_NAME = BOT_1_TRAINED_MODEL_PATH.split('/')[-1].split('.')[-2]
BOT_2_TRAINED_MODEL_PATH = 'my_models/trained_models/trained_fc.pth'
BOT_2_NAME = BOT_2_TRAINED_MODEL_PATH.split('/')[-1].split('.')[-2]

def main_menu(board_size, win_condition):
    game_mode = 0
    num_generations = 1
    while not game_mode:
        print("\n\n***********************")
        print("Generalized Tic-Tac-Toe")
        print("Main Menu")
        print("Board Size:", board_size)
        print("Win Condition:", win_condition)
        print(f"1 - Play {BOT_1_NAME} vs {BOT_2_NAME}")
        print(f"2 - Play human vs {BOT_1_NAME}")
        print(f"3 - Play human vs {BOT_2_NAME}")
        print("4 - Play human vs human")
        print("5 - Change game settings")
        print("6 - Quit Game")
        print("***********************\n")

        while True:
            try:
                userinput = int(input("Enter choice: "))
                if userinput < 1 or userinput > 6:
                    raise ValueError
                break
            except ValueError:
                print('Please enter a valid choice')

        if userinput >= 1 and userinput <= 4:
            game_mode = userinput
            if game_mode == 1:
                while True:
                    try:
                        num_generations = int(input('Enter number of generations to play: '))
                        if num_generations < 1:
                            raise ValueError
                        break
                    except ValueError:
                        print('Please enter a positive integer')
        elif userinput == 5:
            # select board size
            while True:
                try:
                    board_size = int(input('Enter board size: '))
                    if board_size < 1:
                        raise ValueError
                    break
                except ValueError:
                    print('Please enter a positive integer')
            # select win condition
            while True:
                try:
                    win_condition = int(input('Enter new win condition (number of x or o to connect for a win): '))
                    if win_condition < 1 or win_condition > board_size:
                        raise ValueError
                    break
                except ValueError:
                    print('Please enter a positive integer smaller than the board size')
        elif userinput == 6:
            game_mode = -1
            print("\n\nGoodbye!\n")
            sys.exit()

    return board_size, win_condition, game_mode, num_generations

def pretty_board(board_size, time_step):
  """Returns the board in `time_step` in a human readable format."""
  info_state = time_step.observations["info_state"][0]
  o_locations = np.nonzero(info_state[board_size*board_size:board_size*board_size*2])[0]
  x_locations = np.nonzero(info_state[board_size*board_size*2:])[0]
  board = np.full(board_size * board_size, ".")
  board[x_locations] = "X"
  board[o_locations] = "0"
  board = np.reshape(board, (board_size, board_size))
  print(board)

def get_player_move(time_step):
  """Gets a valid action from the user on the command line."""
  current_player = time_step.observations["current_player"]
  legal_actions = time_step.observations["legal_actions"][current_player]
  action = -1
  while action not in legal_actions:
    print("Choose an action from {}:".format(legal_actions))
    sys.stdout.flush()
    action_str = input()
    try:
      action = int(action_str)
    except ValueError:
      continue
  return action

def playGT3(board_size, win_condition, game_mode, gen):
    game = gt3.GT3Game({"board_size": board_size, "win_condition": win_condition})
    env = rl_environment.Environment(game)
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    time_step = env.reset()

    bot_1 = my_dqn.DQN(
            player_id = gen % 2,
            state_representation_size = state_size,
            num_actions = num_actions)
    bot_1.load(BOT_1_TRAINED_MODEL_PATH)

    bot_2 = my_dqn.DQN(
        player_id = 1 - gen % 2,
        state_representation_size = state_size,
        num_actions = num_actions)
    bot_2.load(BOT_2_TRAINED_MODEL_PATH)
    
    if game_mode == 1:
        # Bot 1 vs Bot 2
        if gen % 2:
            # agent_1 = Bot 2, agent_2 = Bot 1
            agents = [bot_2, bot_1]
        else:
            # agent_1 = Bot 1, agent_2 = Bot 2
            agents = [bot_1, bot_2]
        
        while not time_step.last():
            current_player = time_step.observations["current_player"]
            agent_output = agents[current_player].step(time_step, is_evaluation=True)
            time_step = env.step([agent_output.action])
        print("final reward: ", time_step.rewards)
        
        if time_step.rewards[0] == 1:
            if gen % 2:
                return 1
            else:
                return 0
        elif time_step.rewards[1] == 1:
            if gen % 2:
                return 0
            else:
                return 1
        else:
            return -1

    elif game_mode != 4:
        player_char = None
        while player_char not in ['x', 'o']:
            print("Play as x or o: ")
            player_char = input()
        if player_char == 'x':
            print("You are playing as x")
            player_int = 0
        else:
            print("You are playing as o")
            player_int = 1

        if game_mode == 2:
            # human vs bot 1
            bot_1.player_id = 1 - player_int
            opponent = bot_1
        else:
            # human vs bot 2
            bot_2.player_id = 1 - player_int
            opponent = bot_2

        while not time_step.last():
            current_player = time_step.observations["current_player"]
            if current_player == player_int:
                pretty_board(board_size, time_step)
                action = get_player_move(time_step)
            else:
                agent_output = opponent.step(time_step, is_evaluation=True)
                action = agent_output.action
            time_step = env.step([action])

    else:
        # human vs human
        while not time_step.last():
            pretty_board(board_size, time_step)
            action = get_player_move(time_step)
            time_step = env.step([action])

    pretty_board(board_size, time_step)
    if time_step.rewards[0] == 1:
        # x wins
        return 0
    elif time_step.rewards[1] == 1:
        # o wins
        return 1
    else:
        # draw
        return -1           
            


if __name__ == "__main__":

    # Set default game settings
    size = 3
    win_cond = 3
    mode = 0

    # Play game until user chooses to quit
    while True:
        size, win_cond, mode, gens = main_menu(size, win_cond)
    

        if mode == 1:
            bot_1_wins = 0
            bot_2_wins = 0
            draws = 0
            for gen in range(gens):
                winner = playGT3(size, win_cond, mode, gen)
                if winner == 0:
                    bot_1_wins += 1
                    print("Bot 1 wins game {}!\n".format(gen+1))
                elif winner == 1:
                    bot_2_wins += 1
                    print("Bot 2 wins game {}!\n".format(gen+1))
                else:
                    draws += 1
                    print("Game {} is a tie!\n".format(gen+1))
            
        
            print("\n\n***********************")
            print("Game Over!\n")
            print("Number of Bot 1 wins:", bot_1_wins)
            print("Number of Bot 2 wins:", bot_2_wins)
            print("Number of draws:", draws)
            print("***********************")

        else:
            winner = playGT3(size, win_cond, mode, 0)
            print("\n\n***********************")
            print("Game Over!\n")
            if winner == 0:
                print("Player x wins!")
            elif winner == 1:
                print("Player o wins!")
            else:
                print("This game is a draw!")
            print("***********************")

        # Get user input to go back to main menu
        input("Press Enter to continue...")

from open_spiel.python import games
import pyspiel

'''Some examples of using the Python API of OpenSpiel.
See more at https://openspiel.readthedocs.io/en/latest/api_reference.html'''

# Showing the list of supported games.
print(pyspiel.registered_names())

# Loading a game (with parameters).
game_params = {
        "board_size": 5,
        "win_condition": 4
    }
game = pyspiel.load_game("python_gt3", game_params)
print("game:", game)

# Some properties of the game.
print("num_players():", game.num_players())
print("max_utility():", game.max_utility())
print("min_utility():", game.min_utility())
print("num_distinct_actions():", game.num_distinct_actions())

# Creating initial state.
state = game.new_initial_state()
print("state:\n", state, sep="")

# Basic information about states.
print("current_player():", state.current_player())
print("is_terminal():", state.is_terminal())
print("returns():", state.returns())
print("legal_actions():", state.legal_actions())

# Playing the game: applying actions.
state.apply_action(1)
print(state)
print("current_player():", state.current_player())
state.apply_action(2)
state.apply_action(6)
state.apply_action(0)
state.apply_action(11)
state.apply_action(3)
state.apply_action(16)
print(state)
print("is_terminal():", state.is_terminal())
print("player_return(0):", state.player_return(0))   # win for x (player 0)
print("player_return(1):", state.player_return(1))   # loss for o (player 1)
print("current_player():", state.current_player())

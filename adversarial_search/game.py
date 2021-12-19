import numpy as np
import math
import time

class Game:
    """
    Class that creates and plays the Connect Game.
    ---
    Designed to find the best actions using both Minimax and 
    Alpha-Beta Pruning algorithms.

    User plays as MAX by default. Can be unselected so the 
    computer plays itself.
    """

    def __init__(self, columns: int, rows: int, win_length: int, prune: bool = False, user: bool = True):
        # Simple check to ensure the game is winnable
        if win_length > max(columns, rows):
            raise ValueError('Win length cannot be larger than the column or row sizes.')

        self.columns = columns
        self.rows = rows  

        # The number of adjacent tokens needed to win the game
        self.win_length = win_length
        # Allow alpha-beta pruning
        self.prune = prune
        # Allow user to play as MAX
        self.user = user

        # Used to record the times taken by both MAX and MIN to find the best action
        self.max_action_history = list()
        self.min_action_history = list()

        self.turns = 0

        self.initialize_game()

    def initialize_game(self):
        # Sets the board to be an array of zeros of size columns x rows
        self.board = np.zeros((self.rows, self.columns), dtype=int)

    def is_valid(self, board, col):
        """
        Method to determine whether placing a token in the given column is a valid move.
        ---
        Args:
            - `board` (np.ndarray): The board to check.
            - `column` (int): The column index to check.

        Returns:
            - `is_valid` (bool): Whether the move is valid.
        """
        return col >= 0 and col < self.columns \
            and board[self.rows - 1, col] == 0

    def is_terminal_horizontally(self, board, player):
        """
        Method to check if a player has successfully placed enough 
        coins along any row to win the game.
        ---
        Args:
            - `board` (np.ndarray): The board to check.
            - `player` (int): Either 1 for MAX player or 2 for MIN player. 

        Returns:
            - `is_terminal` (bool): Whether the game has a horizontal win.
        """
        for col in range(self.columns - (self.win_length - 1)):
            for row in range(self.rows):
                if np.all(board[row, col:col+self.win_length] == player):
                    return True
        
        return False

    def is_terminal_vertically(self, board, player):
        """
        Method to check if a player has successfully placed enough 
        coins along any column to win the game.
        ---
        Args:
            - `board` (np.ndarray): The board to check.
            - `player` (int): Either 1 for MAX player or 2 for MIN player. 

        Returns:
            - `is_terminal` (bool): Whether the game has a vertical win.
        """
        for col in range(self.columns):
            for row in range(self.rows - (self.win_length - 1)):
                if np.all(board[row:row+self.win_length, col] == player):
                    return True

        return False

    def is_terminal_diagonally(self, board, player):
        """
        Method to check if a player has successfully placed enough 
        coins along any diagonal to win the game.
        ---
        Args:
            - `board` (np.ndarray): The board to check.
            - `player` (int): Either 1 for MAX player or 2 for MIN player. 

        Returns:
            - `is_terminal` (bool): Whether the game has a diagonal win.
        """
        min_length = min(self.columns, self.rows)
        
        diagonal_indices = range(
            self.win_length - self.rows, 
            self.columns - (self.win_length - 1))
    
        for diagonal_index in diagonal_indices:
            diagonal_locations = range(min_length - (self.win_length - 1) - abs(diagonal_index))
            
            # Mirrors the board vertically to see if they have won along the other diagonal
            for version in [board, np.fliplr(board)]:
                diagonal = version.diagonal(diagonal_index)

                for i in diagonal_locations:
                    diagonal_slice = diagonal[i:i+self.win_length]
                    
                    if np.all(diagonal_slice == player):
                        return True
        
        return False

    def board_is_full(self, board):
        """
        Method to determine whether the board is full and the game cannot be continued.
        ---
        Args:
            - `board` (np.ndarray): The board to check.

        Returns:
            - `is_full` (bool): Whehther the board has no more empty cells.
        """
        return not np.any(board == 0)

    def is_terminal(self, board):
        """
        Method that checks whether the game has finished or not.
        ---
        Args:
            - `board` (np.ndarray): The board to check.

        Returns:
            - `is_terminal` (bool): Whether the game has finished.
            - `value` (int): The game result:
                - `1` means the MAX player has won.
                - `-1` means the MIN player has won.
                - `0` means it's a draw.
        """
        for player in [1, 2]:
            if self.is_terminal_horizontally(board, player) \
            or self.is_terminal_vertically(board, player) \
            or self.is_terminal_diagonally(board, player):
                return True, 3 - 2 * player

        if self.board_is_full(board):
            return True, 0

        return False, 0

    def drop_token(self, board, player, col):
        """
        Method to add a token to the board.
        ---
        Given a column, the token will be placed in the lowest empty
        cell of the column specified. Raises an error if the column is full.

        Args:
            - `board` (np.ndarray): The board to add a token to.
            - `player` (int): The player who is adding the token:
                - `1` for the MAX player.
                - `2` for the MIN player.
        """
        if not self.is_valid(board, col):
            raise ValueError(f"Cannot place token into column {col}")

        for row in range(self.rows):
            if board[row, col] == 0:
                board[row, col] = player
                
                return board

    def max(self, board):
        """
        Method to determine the best action of the MAX player.

        This function considers every possible immediate action and 
        calculates its minimum value. The action with the highest minimum
        value is returned as the best action for the MAX player.
        """
        # List of possible actions given the current board state
        actions = list()

        for column in range(self.columns):
            # Ignore the action if a token cannot be placed
            if not self.is_valid(board, column):
                continue

            self.visited_states += 1
            
            # Test the board with this action and calculate the 
            # minimum possible value resulting from it
            test_board = self.drop_token(board.copy(), 1, column)
            action_value = self.min_value(test_board)
            actions.append(tuple([column, action_value]))
        
        # Choose the action with the greatest minimum value
        max_col, _ = max(actions, key=lambda v: v[1])

        return max_col

    def min(self, board):
        """
        Method to determine the best action of the MIN player.

        This function considers every possible immediate action and
        calculates its maximum value. The MIN player is trying to minimise
        the reward, so the action with the lowest maximum  value is the best
        action for the MIN player.
        """
        # List of possible actions given the current board state
        actions = list()

        for column in range(self.columns):
            # Ignore the action if a token cannot be placed
            if not self.is_valid(board, column):
                continue

            self.visited_states += 1

            # Test the board with this action and calculate the 
            # maximum possible value resulting from it
            test_board = self.drop_token(board.copy(), 2, column)
            action_value = self.max_value(test_board)
            actions.append(tuple([column, action_value]))
        
        # Choose the action with the lowest maximum value
        min_col, _ = min(actions, key=lambda v: v[1])

        return min_col

    def max_value(self, board, alpha=-math.inf, beta=math.inf):
        """
        Method to determine the maximum value possible given the current board state.
        ---
        This is used recursively with Game.min_value(). If no more actions can be taken,
        the value of the board is returned. Else, return the value of the action that 
        results yields the greatest value. If multiple actions have the same value, 
        the first action in the list is returned.

        If Game.prune is set to true, then the algorithm will apply alpha-beta pruning.

        Args:
            - `board` (np.ndarray): The board to find the maximum possible value from.
            - `alpha` (float, automatically set): The lower threshold for alpha-beta
                pruning.
            - `beta` (float, automatically set): The upper threshold for alpha-beta
                pruning.
        
        Returns:
            - `min_value` (float/int): The minimum value possible in the board.
        """
        # If it's the end of the game, return the game value
        is_terminal, value = self.is_terminal(board)
        
        if is_terminal:
            return value
        
        value = -math.inf

        for column in range(self.columns):
            # Ignore this action if a token cannot be placed
            if not self.is_valid(board, column):
                continue
            
            self.visited_states += 1

            # Find the minimum value of this action and keep track of the largest value found
            test_board = self.drop_token(board.copy(), 1, column)
            value = max(value, self.min_value(test_board, alpha, beta))

            # Stop iterating through the actions if a value has been found 
            # that is above the upper threshold and pruning has been enabled.            
            if self.prune and value >= beta:
                return value

            # Update lower threshold to be as high or more as the current value
            alpha = max(alpha, value)

        return value

    def min_value(self, board, alpha=-math.inf, beta=math.inf):
        """
        Method to determine the minimum value possible given the current board state.
        ---
        This is used recursively with `Game.min_value()`. If no more actions can be
        taken, the value of the board is returned. Else, return the value of the action
        that results in the lowest value. If multiple actions have the same value, the
        first action in the list is returned.

        If Game.prune is set to true, then the algorithm will apply alpha-beta pruning.

        Args:
            - `board` (np.ndarray): The board to find the minimum possible value from.
            - `alpha` (float, automatically set): The lower threshold for alpha-beta
                pruning.
            - `beta` (float, automatically set): The upper threshold for alpha-beta
                pruning.
        
        Returns:
            - `min_value` (float/int): The minimum value possible in the board.
        """
        # If it's the end of the game, return the final value
        is_terminal, value = self.is_terminal(board)
        
        if is_terminal:
            return value
        
        value = math.inf

        for column in range(self.columns):
            # Ignore action if a token cannot be placed
            if not self.is_valid(board, column):
                continue

            self.visited_states += 1

            # Calculate the maximum value achievable from this action and keep track of
            # the smallest value found from all actions.
            test_board = self.drop_token(board.copy(), 2, column)
            value = min(value, self.max_value(test_board, alpha, beta))

            # Stop iterating through the actions if a value has been found 
            # that is below the lower threshold and pruning has been enabled.
            if self.prune and value <= alpha:
                return value

            # Update upper threshold to be as low or more as the current value
            beta = min(beta, value)

        return value

    def best_action(self, player):
        """
        Method to return best action of each player.
        ---
        Args:
            - `player` (int): The player number:
                - `1` is the MAX player.
                - `2` is the MIN player.

        Returns:
            - `best_action` (int): The column index that corresponds to the best action
                given the current board state.
        """
        if player == 1:
            return self.max(self.board)
        else:
            return self.min(self.board)

    def user_select_move(self, best_action):
        """
        Method to allow the user to select an action.
        ---
        Suggests to the player the best move, requests an input number which is
        converted to a column index.

        Args:
            - `best_action` (int): The column index that corresponds to the best action 
                given the current board state.

        Returns:
            - `action` (int): The column index that the user chose.
        """
        print(f"Suggested move for MAX is: {best_action}.")
        
        while True:
            try:
                column_string = input('Enter column to drop token: ')
                column = int(column_string)
            except ValueError:
                print(f"'{column_string}' is not an integer.")
                continue

            if self.is_valid(self.board, column):
                print(f"MAX drops X in {column}.")
                return column
            else:
                print(f"{column} cannot be chosen.")

    def turn(self, player):
        """
        Method to allow a player to take a turn.
        ---
        Calculates the best action of the player, records the time it took to find this
        action, the number of states and drops the token.

        Args:
            - `player` (int): The player number:
                - `1` is the MAX player.
                - `2` is the MIN player.
        """
        self.visited_states = 0

        start_time = time.time()
        best_action = self.best_action(player)
        end_time = time.time()

        if player == 1:
            if self.user:
                best_action = self.user_select_move(best_action)

            self.max_action_history.append({
                "move": best_action, 
                "time": end_time - start_time, 
                "visited_states": self.visited_states
            })

            print(f"MAX plays: {best_action}.")
        
        else:
            self.min_action_history.append({
                "move": best_action, 
                "time": end_time - start_time, 
                "visited_states": self.visited_states
            })

            print(f"MIN plays: {best_action}.")

        self.drop_token(self.board, player, best_action)

    def drawboard(self):
        """
        Method to draw the board in the command line.
        """
        for row in reversed(self.board):
            row_string = ' '.join(list(map(str, row)))

            row_string = row_string.replace('0', '_')
            row_string = row_string.replace('1', 'X')
            row_string = row_string.replace('2', 'O')

            print(row_string)

    def report_winner(self, value):
        """
        Method that draws the board, and reports who won based on the value of the game.
        """
        self.drawboard()
        
        if value == 1:
            print('MAX wins!')
        elif value == -1:
            print('MIN wins!')
        else:
            print('Draw!')

    def play(self):
        """
        Main method of the Game class, that allows two players to compete.
        ---
        Continues in a loop forever until one player loses.

        Returns:
            - `value` (float, int): The final value of the game.
        """
        self.turns = 0
        
        while True:
            self.turns += 1
            
            for player in [1, 2]:
                print(f"Turn {self.turns}. Player {player}.")
                
                self.drawboard()
                self.turn(player)

                # Check for win
                is_terminal, value = self.is_terminal(self.board)
        
                if is_terminal:
                    self.report_winner(value)
                    return value

if __name__ == '__main__':
    game = Game(4, 4, 2, True)
    game.play()
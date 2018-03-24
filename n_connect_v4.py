import tensorflow as tf
import numpy as np


class Constant(object):
    class Board(object):
        ROW_SIZE = 6
        COL_SIZE = 6
        WIN_NUMBER = 4

    class Model(object):
        class Dense(object):
            LAYER_SIZE = 16
            UNIT_SIZE = 128

        class Training(object):
            RANDOM_CHOICE_PERCENTAGE = 0.05
            ONE_DATA_SET_GAME_NUMBER = 1000
            EPOCHS = 1

    class Player(object):
        A = 0
        B = 1


def my_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(
        input_shape=(Constant.Board.ROW_SIZE, Constant.Board.COL_SIZE, 3)
    ))

    for _ in range(Constant.Model.Dense.LAYER_SIZE):
        model.add(tf.keras.layers.Dense(
            units=Constant.Model.Dense.UNIT_SIZE,
            activation='relu',
        ))

    model.add(tf.keras.layers.Dense(
        units=Constant.Board.ROW_SIZE * Constant.Board.COL_SIZE,
        activation='linear',
    ))

    model.add(tf.keras.layers.Reshape(target_shape=(Constant.Board.ROW_SIZE, Constant.Board.COL_SIZE)))
    model.compile(loss='mean_squared_error',
                  optimizer='Adam')

    return model


class Game(object):
    class Players(object):
        class Player(object):
            def __init__(self, value):
                self.value = value
                self.board_log = list()
                self.choice_log = list()
                self.model_input_log = list()
                self.model_output_log = list()

        def __init__(self):
            self.A = self.Player(Constant.Player.A)
            self.B = self.Player(Constant.Player.B)
            self.A.next_player = self.B
            self.B.next_player = self.A

    def __init__(self):
        self.players = self.Players()
        self.current_player = self.players.A
        self.current_board = {self.players.A: np.zeros([Constant.Board.ROW_SIZE, Constant.Board.COL_SIZE]),
                              self.players.B: np.zeros([Constant.Board.ROW_SIZE, Constant.Board.COL_SIZE])}

    def put_stone(self, location):
        self.current_player.board_log.append({
                self.players.A: self.current_board[self.players.A].copy(),
                self.players.B: self.current_board[self.players.B].copy(),
            }) #deep copy
        self.current_player.choice_log.append(location)

        self.current_board[self.current_player][location] = True

        def is_current_player_winner():
            for i in range(Constant.Board.WIN_NUMBER):
                try:
                    if np.array([
                        self.current_board[self.current_player]
                        [location[0]]
                        [location[1] - i + j]

                        for j in range(Constant.Board.WIN_NUMBER)
                    ]).all() and location[1] - i >= 0:
                        return True

                except IndexError:
                    pass

                try:
                    if np.array([
                        self.current_board[self.current_player]
                        [location[0] - i + j]
                        [location[1]]

                        for j in range(Constant.Board.WIN_NUMBER)
                    ]).all() and location[0] - i >= 0:
                        return True

                except IndexError:
                    pass

                try:
                    if np.array([
                        self.current_board[self.current_player]
                        [location[0] - i + j]
                        [location[1] - i + j]

                        for j in range(Constant.Board.WIN_NUMBER)
                    ]).all() and location[0] - i >= 0 and location[1] - i >= 0:
                        return True

                except IndexError:
                    pass

                try:
                    if np.array([
                        self.current_board[self.current_player]
                        [location[0] + i - j]
                        [location[1] - i + j]

                        for j in range(Constant.Board.WIN_NUMBER)
                    ]).all() and location[0] + i - Constant.Board.WIN_NUMBER + 1 >= 0\
                            and location[1] - i >= 0:
                        return True

                except IndexError:
                    pass

            return False

        if is_current_player_winner():
            return self.current_player

        else:
            if not self.get_available_location(self.current_board).any():
                return self.players.B
            else:
                self.current_player = self.current_player.next_player
                return None

    def get_available_location(self, board):
        return np.where((board[self.players.A] + board[self.players.B]) == True, False, True)

    def put_stone_by_model(self, model, random_percentage):
        model_input = list()
        model_input.append(self.current_board[self.players.A])
        model_input.append(self.current_board[self.players.B])
        model_input.append(np.full([Constant.Board.ROW_SIZE, Constant.Board.COL_SIZE], self.current_player.value))

        model_input = np.swapaxes(np.swapaxes(model_input, 0, 1), 1, 2)

        self.current_player.model_input_log.append(model_input)

        model_output = model.predict_on_batch(np.array([model_input]))[0]
        self.current_player.model_output_log.append(model_output)

        if np.random.rand() < random_percentage:
            prediction = np.multiply(
                np.random.rand(Constant.Board.ROW_SIZE, Constant.Board.COL_SIZE),
                self.get_available_location(self.current_board)
            )
        else:
            prediction = np.multiply(
                model_output,
                self.get_available_location(self.current_board),
            )
        #print(prediction)
        location = np.unravel_index(prediction.argmax(), prediction.shape)
        return self.put_stone(location)

    def print_board(self):
        print(self.current_board[self.players.A])
        print(self.current_board[self.players.B])


def get_episode_from_one_game(model):
    game = Game()
    while True:
        winner = game.put_stone_by_model(model, Constant.Model.Training.RANDOM_CHOICE_PERCENTAGE)
        if winner != None:
            break
    loser = winner.next_player

    x = np.append(winner.model_input_log, loser.model_input_log, axis=0)

    y = list()
    for player in [winner, loser]:
        for board, choice, model_output in zip(player.board_log, player.choice_log, player.model_output_log):
            temp_y = model_output
            if player == winner:
                temp_y[choice] += 1
            else:
                temp_y[choice] -= 1
            temp_y -= (1 - game.get_available_location(board))
            y.append(temp_y)

    return x, y


def train_one_data_set(model):
    x = list()
    y = list()

    print('generating data set...')
    for i in range(Constant.Model.Training.ONE_DATA_SET_GAME_NUMBER):
        episode = get_episode_from_one_game(model)
        for data in episode[0]:
            x.append(data)
        for data in episode[1]:
            y.append(data)

        if i % (Constant.Model.Training.ONE_DATA_SET_GAME_NUMBER / 10) == 0:
            print(i/Constant.Model.Training.ONE_DATA_SET_GAME_NUMBER*100, '%')

    x = np.array(x)
    y = np.array(y)

    print('data generation complete')
    model.fit(x=x, y=y, verbose=1, epochs=Constant.Model.Training.EPOCHS)
    print('model train complete')


def play_game_with_human(model):
    game = Game()
    while True:
        game.put_stone_by_model(model, 0.0)
        game.print_board()

        print('row : ')
        row = int(input())
        print('col : ')
        col = int(input())
        game.put_stone(location=(row, col))


def main():
    print('initialize model?(y/n)')
    model_file_path = './saved_model/my_model.h5'
    if input() == 'y':
        model = my_model()
        tf.keras.models.save_model(model=model, filepath=model_file_path)

    else:
        model = tf.keras.models.load_model(model_file_path)

    print('data set number')
    for _ in range(int(input())):
        train_one_data_set(model)
        print('iteration ', _)
        tf.keras.models.save_model(model=model, filepath=model_file_path)
        print('model saved')

    play_game_with_human(model)


main()

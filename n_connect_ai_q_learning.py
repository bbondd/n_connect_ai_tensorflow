import tensorflow as tf
import numpy as np


def one_hot(shape, index):
    ret = np.zeros(shape)
    ret[index] = 1
    return ret


np.one_hot = one_hot
del one_hot


class Constant(object):
    class Board(object):
        ROW_SIZE = 6
        COL_SIZE = 6

        WIN_NUMBER = 4

    class Model(object):
        input_turn_number = 5

        class Reward(object):
            A_WIN_BONUS = 10
            A_DRAW_BONUS = -1
            A_LOSE_BONUS = -1
            B_WIN_BONUS = 10
            B_DRAW_BONUS = 10
            B_LOSE_BONUS = -1

        class Conv2D(object):
            FILTER_SIZE = 64
            KERNEL_SIZE = (1, 1)

        class Dense(object):
            LAYER_SIZE = 16
            UNIT_SIZE = 128

        class Dropout(object):
            RATE = 0.5

    class Player(object):
        A = 0
        B = 1
        DRAW = 'draw'


def my_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(
        input_shape=(Constant.Board.ROW_SIZE, Constant.Board.COL_SIZE, Constant.Model.input_turn_number * 2 + 1)
    ))

    for _ in range(Constant.Model.Dense.LAYER_SIZE):
        model.add(tf.keras.layers.Dense(
            units=Constant.Model.Dense.UNIT_SIZE,
            activation='relu',
        ))

    model.add(tf.keras.layers.Dense(
        units=Constant.Board.ROW_SIZE * Constant.Board.COL_SIZE,
        activation='sigmoid',
    ))
    model.add(tf.keras.layers.Reshape(target_shape=(Constant.Board.ROW_SIZE, Constant.Board.COL_SIZE)))
    model.compile(loss='binary_crossentropy',
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

        def __init__(self):
            self.A = self.Player(Constant.Player.A)
            self.B = self.Player(Constant.Player.B)
            self.A.next_player = self.B
            self.B.next_player = self.A

            self.Draw = self.Player(Constant.Player.DRAW)

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
                return self.players.Draw
            else:
                self.current_player = self.current_player.next_player
                return None

    def get_available_location(self, board):
        return np.where((board[self.players.A] + board[self.players.B]) == True, False, True)

    def put_stone_by_model(self, model):
        model_input = list()
        model_input.append(self.current_board[self.players.A])
        model_input.append(self.current_board[self.players.B])
        for i in range(Constant.Model.input_turn_number - 1):
            try:
                model_input.append(self.players.A.board_log[-i - 1][self.players.A])
            except IndexError:
                model_input.append(np.zeros([Constant.Board.ROW_SIZE, Constant.Board.COL_SIZE]))

            try:
                model_input.append(self.players.B.board_log[-i - 1][self.players.B])
            except IndexError:
                model_input.append(np.zeros([Constant.Board.ROW_SIZE, Constant.Board.COL_SIZE]))
        model_input.append(np.full([Constant.Board.ROW_SIZE, Constant.Board.COL_SIZE], self.current_player.value))

        model_input = np.swapaxes(np.swapaxes(model_input, 0, 1), 1, 2)

        self.current_player.model_input_log.append(model_input)
        prediction = np.multiply(
            model.predict_on_batch(np.array([model_input]))[0],
            self.get_available_location(self.current_board),
        )
        location = np.unravel_index(prediction.argmax(), prediction.shape)
        return self.put_stone(location)

    def print_board(self):
        print(self.current_board[self.players.A])
        print(self.current_board[self.players.B])


def play_one_game(model):
    game = Game()
    while True:
        winner = game.put_stone_by_model(model)
        if winner != None:
            #print(winner.value)
            #game.print_board()
            break

    x = np.append(game.players.A.model_input_log, game.players.B.model_input_log, axis=0)

    A_y = []
    for board, choice in zip(game.players.A.board_log, game.players.A.choice_log):
        temp_A_y = np.one_hot([Constant.Board.ROW_SIZE, Constant.Board.COL_SIZE], choice)

        if winner == game.players.A:
            temp_A_y *= Constant.Model.Reward.A_WIN_BONUS
        elif winner == game.players.B:
            temp_A_y *= Constant.Model.Reward.A_LOSE_BONUS
        else:  # draw
            temp_A_y *= Constant.Model.Reward.A_DRAW_BONUS

        temp_A_y += game.get_available_location(board)
        temp_A_y = temp_A_y / temp_A_y.sum()
        A_y.append(temp_A_y)

    B_y = []
    for board, choice in zip(game.players.B.board_log, game.players.B.choice_log):
        temp_B_y = np.one_hot([Constant.Board.ROW_SIZE, Constant.Board.COL_SIZE], choice)

        if winner == game.players.A:
            temp_B_y *= Constant.Model.Reward.B_LOSE_BONUS
        elif winner == game.players.B:
            temp_B_y *= Constant.Model.Reward.B_WIN_BONUS
        else:  # draw
            temp_B_y *= Constant.Model.Reward.B_DRAW_BONUS

        temp_B_y += game.get_available_location(board)
        temp_B_y = temp_B_y / temp_B_y.sum()
        B_y.append(temp_B_y)

    y = np.append(A_y, B_y, axis=0)

    model.fit(x, y, verbose=0)


def play_game_with_human(model):
    game = Game()
    while True:
        game.put_stone_by_model(model)
        game.print_board()

        print('row : ')
        row = int(input())
        print('col : ')
        col = int(input())
        game.put_stone(location=(row, col))


def self_play(model_a, model_b):
    game = Game()
    while True:
        winner = game.put_stone_by_model(model_a)
        if winner != None:
            break
        winner = game.put_stone_by_model(model_b)
        if winner != None:
            break

    if winner == game.players.A:
        return model_a
    elif winner == game.players.B:
        return model_b
    else:
        return None


def self_two_play(model_a, model_b):
    model_a_win_number = 0
    model_b_win_number = 0
    draw_number = 0

    win_model = self_play(model_a, model_b)
    if win_model == model_a:
        model_a_win_number += 1
    elif win_model == model_b:
        model_b_win_number += 1
    else:
        draw_number += 1

    win_model = self_play(model_b, model_a)
    if win_model == model_a:
        model_a_win_number += 1
    elif win_model == model_b:
        model_b_win_number += 1
    else:
        draw_number += 1

    return model_a_win_number, model_b_win_number, draw_number


def main():
    print('initialize model?(y/n)')
    model_file_path = './saved_model/my_model.h5'
    old_model_file_path = './saved_model/old_model.h5'
    if input() == 'y':
        model = my_model()
        old_model = my_model()
        tf.keras.models.save_model(model=model, filepath=model_file_path)
        tf.keras.models.save_model(model=old_model, filepath=old_model_file_path)
    else:
        model = tf.keras.models.load_model(model_file_path)
        old_model = tf.keras.models.load_model(old_model_file_path)

    print('epoch')
    for _ in range(int(input())):
        play_one_game(model)

        if _ % 1000 == 0:
            print(_)
            self_two_play_result = self_two_play(model, old_model)
            print(self_two_play_result)

            if self_two_play_result[0] > 1:
                tf.keras.models.save_model(model=model, filepath=old_model_file_path)
                old_model = tf.keras.models.load_model(old_model_file_path)
                print('old_model_renewed')

            tf.keras.models.save_model(model=model, filepath=model_file_path)
            print('model saved')

    play_game_with_human(old_model)


main()


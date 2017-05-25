import numpy as np

cimport numpy as np

from libc.stdlib cimport malloc, free

from nose.tools import ok_, eq_

cimport go
cimport policy_feature
cimport parseboard
cimport printer


def test_stone_color():
    game = go.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             "a . .|"
                             ". b .|"
                             ". . c|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    go.put_stone(game, moves['a'], go.S_BLACK)
    go.put_stone(game, moves['b'], go.S_WHITE)

    game.current_color = go.S_BLACK
    policy_feature.update(feature, game)

    eq_(planes[0, pure_moves['a']], 1)
    eq_(planes[1, pure_moves['b']], 1)
    eq_(planes[2, pure_moves['c']], 1)
    eq_(planes[0].sum(), 1)
    eq_(planes[1].sum(), 1)
    eq_(planes[2].sum(), 7)

    game.current_color = go.S_WHITE
    policy_feature.update(feature, game)

    eq_(planes[0, pure_moves['b']], 1)
    eq_(planes[1, pure_moves['a']], 1)
    eq_(planes[2, pure_moves['c']], 1)
    eq_(planes[0].sum(), 1)
    eq_(planes[1].sum(), 1)
    eq_(planes[2].sum(), 7)

    go.free_game(game)
    policy_feature.free_feature(feature)


def test_turns_since():
    game = go.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             ". . . . .|"
                             ". a b c .|"
                             ". d e f .|"
                             ". g . . .|"
                             ". . . . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = go.S_BLACK

    # move 'a'
    go.do_move(game, moves['a'])
    policy_feature.update(feature, game)
    eq_(planes[4, pure_moves['a']], 1) # 0 age
    eq_(planes[4].sum(), 1)

    # move 'b'
    go.do_move(game, moves['b'])
    policy_feature.update(feature, game)
    eq_(planes[5, pure_moves['a']], 1) # 1 age
    eq_(planes[4, pure_moves['b']], 1) # 0 age
    eq_(planes[5].sum(), 1)
    eq_(planes[4].sum(), 1)

    # PASS
    go.do_move(game, go.PASS)
    go.do_move(game, go.PASS)
    policy_feature.update(feature, game)
    eq_(planes[7, pure_moves['a']], 1) # 3 age
    eq_(planes[6, pure_moves['b']], 1) # 2 age
    eq_(planes[7].sum(), 1)
    eq_(planes[6].sum(), 1)

    go.do_move(game, moves['c'])
    go.do_move(game, moves['d'])
    go.do_move(game, moves['e'])
    go.do_move(game, moves['f'])
    policy_feature.update(feature, game)
    eq_(planes[11, pure_moves['a']], 1) # 7 age
    eq_(planes[10, pure_moves['b']], 1) # 6 age
    eq_(planes[7, pure_moves['c']], 1)  # 3 age
    eq_(planes[6, pure_moves['d']], 1)  # 2 age
    eq_(planes[4, pure_moves['f']], 1)  # 0 age

    go.do_move(game, moves['g'])
    policy_feature.update(feature, game)
    eq_(planes[11, pure_moves['a']], 1) # 7 age
    eq_(planes[11, pure_moves['b']], 1) # 7 age
    eq_(planes[8, pure_moves['c']], 1)  # 4 age
    eq_(planes[7, pure_moves['d']], 1)  # 3 age
    eq_(planes[5, pure_moves['f']], 1)  # 1 age
    eq_(planes[4, pure_moves['g']], 1)  # 0 age

    go.free_game(game)
    policy_feature.free_feature(feature)


def test_liberties():
    game = go.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             ". . . . .|"
                             ". . a . .|"
                             ". . b . .|"
                             ". . c . .|"
                             ". . d . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    go.put_stone(game, moves['a'], go.S_BLACK)
    policy_feature.update(feature, game)
    eq_(planes[12 + 4 - 1, pure_moves['a']], 1)

    go.put_stone(game, moves['b'], go.S_BLACK)
    go.put_stone(game, moves['c'], go.S_BLACK)
    policy_feature.update(feature, game)
    eq_(planes[12 + 8 - 1, pure_moves['a']], 1)
    eq_(planes[12 + 8 - 1, pure_moves['b']], 1)
    eq_(planes[12 + 8 - 1, pure_moves['c']], 1)

    go.put_stone(game, moves['d'], go.S_BLACK)
    policy_feature.update(feature, game)
    eq_(planes[12 + 8 - 1, pure_moves['a']], 1)
    eq_(planes[12 + 8 - 1, pure_moves['b']], 1)
    eq_(planes[12 + 8 - 1, pure_moves['c']], 1)
    eq_(planes[12 + 8 - 1, pure_moves['d']], 1)

    go.free_game(game)
    policy_feature.free_feature(feature)


def test_capture_size():
    game = go.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             "O a . . O X X X X|"
                             "X . . . . O X X X|"
                             ". . . . . . O X X|"
                             ". . . . . . . O c|"
                             ". . . . . . . . .|"
                             ". . . . . . . . .|"
                             ". . . . . . . . .|"
                             "X X X X X X X X .|"
                             "O O O O O O O O b|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = go.S_BLACK
    policy_feature.update(feature, game)

    eq_(planes[20, pure_moves['a']], 1)
    eq_(planes[20].sum(), 1)
    eq_(planes[27, pure_moves['b']], 1)
    eq_(planes[27].sum(), 1)

    game.current_color = go.S_WHITE
    policy_feature.update(feature, game)

    eq_(planes[27, pure_moves['c']], 1)
    eq_(planes[27].sum(), 1)

    go.free_game(game)
    policy_feature.free_feature(feature)


def test_self_atari_size():
    game = go.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             "O a . . O X X X X|"
                             "X . . . . O X X X|"
                             ". . . . . . O X X|"
                             ". . . . . . . O c|"
                             ". . . . . . . . .|"
                             ". . . . . . . . .|"
                             ". . . . . . . . .|"
                             "X X X X X X X X .|"
                             "O O O O O O O O b|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = go.S_WHITE
    policy_feature.update(feature, game)

    eq_(planes[28 + 1 - 1, pure_moves['a']], 1)
    eq_(planes[28 + 1 - 1].sum(), 1)
    eq_(planes[28 + 8 - 1, pure_moves['b']], 1)
    eq_(planes[28 + 8 - 1].sum(), 1)

    game.current_color = go.S_BLACK
    policy_feature.update(feature, game)

    eq_(planes[28 + 8 - 1, pure_moves['c']], 1)
    eq_(planes[28 + 8 - 1].sum(), 1)

    go.free_game(game)
    policy_feature.free_feature(feature)


def test_liberties_after_move_1():
    game = go.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             ". . . . .|"
                             ". . a . .|"
                             ". . b . .|"
                             ". . c . .|"
                             ". . d e .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = go.S_BLACK

    # after B 'a'
    policy_feature.update(feature, game)
    eq_(planes[36 + 4 - 1, pure_moves['a']], 1)

    # after B 'b'
    go.put_stone(game, moves['a'], go.S_BLACK)
    policy_feature.update(feature, game)
    eq_(planes[36 + 6 - 1, pure_moves['b']], 1)

    # after B 'c'
    go.put_stone(game, moves['b'], go.S_BLACK)
    policy_feature.update(feature, game)
    eq_(planes[36 + 8 - 1, pure_moves['c']], 1)

    # after B 'd'
    go.put_stone(game, moves['c'], go.S_BLACK)
    policy_feature.update(feature, game)
    eq_(planes[36 + 8 - 1, pure_moves['d']], 1)

    game.current_color = go.S_WHITE

    # after W 'e'
    go.put_stone(game, moves['d'], go.S_BLACK)
    policy_feature.update(feature, game)
    eq_(planes[36 + 2 - 1, pure_moves['e']], 1)

    go.free_game(game)
    policy_feature.free_feature(feature)


def test_sensibleness():
    pass


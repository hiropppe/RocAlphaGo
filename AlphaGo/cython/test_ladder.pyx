import numpy as np

cimport numpy as np

from libc.stdlib cimport malloc, free

from nose.tools import ok_, eq_

cimport go
cimport policy_feature
cimport parseboard
cimport printer


def test_captured_1():
    game = go.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                             "d b c . . . .|"
                             "B W a . . . .|"
                             ". B . . . . .|"
                             ". . . . . . .|"
                             ". . . . . . .|"
                             ". . . . . W .|")

    game.current_color = go.S_BLACK

    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)

    planes = np.asarray(feature.planes)

    # 'a' should catch white in a ladder, but not 'b'
    policy_feature.update(feature, game)
    eq_(planes[44, pure_moves['a']], 1)
    eq_(planes[44, pure_moves['b']], 0)
    eq_(planes[44].sum(), 1)
    eq_(planes[45].sum(), 0)

    # 'b' should not be an escape move for white after 'a'
    go.do_move(game, moves['a'])
    policy_feature.update(feature, game)
    eq_(planes[45, pure_moves['b']], 0)
    eq_(planes[44].sum(), 0)
    eq_(planes[45].sum(), 0)

    # W at 'b', check 'c' and 'd'
    go.do_move(game, moves['b'])
    policy_feature.update(feature, game)
    eq_(planes[44, pure_moves['c']], 1)
    eq_(planes[44, pure_moves['d']], 0)
    eq_(planes[44].sum(), 1)
    eq_(planes[45].sum(), 0)

    go.free_game(game)
    policy_feature.free_feature(feature)


def test_breaker_1():
    game = go.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                                 ". B . . . . .|"
                                 "B W a . . W .|"
                                 "B b . . . . .|"
                                 ". c . . . . .|"
                                 ". . . . . . .|"
                                 ". . . . . W .|"
                                 ". . . . . . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = go.S_BLACK
    
    # 'a' should not be a ladder capture, nor 'b'
    policy_feature.update(feature, game)
    eq_(planes[44, pure_moves['a']], 0)
    eq_(planes[44, pure_moves['b']], 0)
    eq_(planes[44].sum(), 0)
    eq_(planes[45].sum(), 0)

    # after 'a', 'b' should be an escape
    go.do_move(game, moves['a'])
    #policy_feature.is_ladder_escape(game, game.string_id[moves['a']-1], moves['b'], True, feature.search_games, 0)
    #policy_feature.is_ladder_capture(game, game.string_id[25], 26, 24, feature.search_games, 0)
    policy_feature.update(feature, game)
    eq_(planes[45, pure_moves['b']], 1)
    eq_(planes[44].sum(), 0)
    eq_(planes[45].sum(), 1)

    # after 'b', 'c' should not be a capture
    go.do_move(game, moves['b'])
    policy_feature.update(feature, game)
    eq_(planes[44, pure_moves['c']], 0)
    eq_(planes[44].sum(), 0)
    eq_(planes[45].sum(), 0)

    go.free_game(game)
    policy_feature.free_feature(feature)


def test_missing_ladder_breaker_1():
    game = go.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                                 ". B . . . . .|"
                                 "B W B . . W .|"
                                 "B a c . . . .|"
                                 ". b . . . . .|"
                                 ". . . . . . .|"
                                 ". W . . . . .|"
                                 ". . . . . . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = go.S_WHITE

    # a should not be an escape move for white
    policy_feature.update(feature, game)
    eq_(planes[45, pure_moves['a']], 0)

    # after 'a', 'b' should still be a capture ...
    # ... but 'c' should not
    go.do_move(game, moves['a'])
    policy_feature.update(feature, game)
    eq_(planes[44, pure_moves['b']], 1)
    eq_(planes[44, pure_moves['c']], 0)

    go.free_game(game)
    policy_feature.free_feature(feature)


def test_capture_to_escape_1():
    game = go.allocate_game()
    (moves, pure_moves) = parseboard.parse(game,
                                ". O X . . . .|"
                                ". X O X . . .|"
                                ". . O X . . .|"
                                ". . a . . . .|"
                                ". O . . . . .|"
                                ". . . . . . .|")
    feature = policy_feature.allocate_feature()
    policy_feature.initialize_feature(feature)
    planes = np.asarray(feature.planes)

    game.current_color = go.S_BLACK

    # 'a' is not a capture because of ataris
    # policy_feature.is_ladder_capture(game, game.string_id[48], 59, 47, feature.search_games, 0)
    policy_feature.update(feature, game)
    eq_(planes[44, pure_moves['a']], 0)
    eq_(planes[44].sum(), 0)
    #print np.where(planes[45] == 1)
    #eq_(planes[45].sum(), 0)

    go.free_game(game)
    policy_feature.free_feature(feature)


def test_throw_in_1():
    pass


def test_snapback_1():
    pass


def test_two_captures():
    pass


def test_two_escapes():
    pass

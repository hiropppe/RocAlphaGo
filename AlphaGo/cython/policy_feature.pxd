# -*- coding: utf-8 -*-

cimport go

cdef enum:
    MAX_POLICY_PLANES = 48

ctypedef struct policy_feature_t:
    int n_planes
    int[:, ::1] planes
    go.game_state_t *search_game


cdef policy_feature_t *allocate_feature()
cdef void initialize_feature(policy_feature_t *feature)
cdef void free_feature(policy_feature_t *feature)

cdef void update(policy_feature_t *feature, go.game_state_t *game)

import numpy as np

import AlphaGo.ai as ai

from AlphaGo.models.policy import CNNPolicy
from AlphaGo.models.value import CNNValue

from interface.gtp_wrapper import GTPGameConnector, ExtendedGtpEngine


def apply_temperature(distribution, beta=0.67):
    log_probabilities = np.log(distribution)
    # apply beta exponent to probabilities (in log space)
    log_probabilities = log_probabilities * beta
    # scale probabilities to a more numerically stable range (in log space)
    log_probabilities = log_probabilities - log_probabilities.max()
    # convert back from log space
    probabilities = np.exp(log_probabilities)
    # re-normalize the distribution
    return probabilities / probabilities.sum()


def f_policy(policy, state):
    legal_moves = state.get_legal_moves(include_eyes=False)
    if len(legal_moves) > 0:
        move_probs = policy.eval_state(state, legal_moves)
        moves, probs = zip(*move_probs)
        probs = apply_temperature(probs)
        return zip(moves, probs)


def f_value(value, state):
    return value.eval_state(state)


def f_rollout(rollout, state):
    legal_moves = state.get_legal_moves(include_eyes=False)
    if len(legal_moves) > 0:
        move_probs = rollout.eval_state(state, legal_moves)
        return zip(*move_probs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run GTP')
    parser.add_argument("policy_model", help="Policy network model (json)")
    parser.add_argument("policy_weight", help="Policy network weight (hdf5)")
    parser.add_argument("value_model", help="Value network model (json)")
    parser.add_argument("value_weight", help="Value network weight (hdf5)")
    parser.add_argument("--server", default=False, action="store_true",
                        help="Server mode")
    parser.add_argument("--n-playout", type=int, default=10,
                        help="Number of simulation for each play")

    args = parser.parse_args()

    policy_net = CNNPolicy.load_model(args.policy_model)
    policy_net.model.load_weights(args.policy_weight)

    value_net = CNNValue.load_model(args.value_model)
    value_net.model.load_weights(args.value_weight)

    player = ai.MCTSPlayer(lambda state: f_value(value_net, state),
                           lambda state: f_policy(policy_net, state),
                           lambda state: f_rollout(policy_net, state),
                           n_playout=args.n_playout)

    gtp_game = GTPGameConnector(player)
    gtp_engine = ExtendedGtpEngine(gtp_game, name='RocAlphaGo', version='0.0')

    if args.server:
        from flask import Flask
        from flask import request

        app = Flask(__name__)

        @app.route('/gtp')
        def gtp():
            cmd = request.args.get('cmd')
            print(cmd)
            engine_reply = gtp_engine.send(cmd)
            print(engine_reply)
            return engine_reply

        app.run(host="0.0.0.0", debug=True)
    else:
        from interface.gtp_wrapper import run_gtp
        run_gtp(player)

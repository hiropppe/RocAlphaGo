#!/usr/bin/env python
import numpy as np
from AlphaGo.preprocessing.rollout_preprocessing import RolloutPreprocess
from AlphaGo.util import sgf_iter_states
import AlphaGo.go as go
import os
import warnings
import sgf
import tables

from tqdm import tqdm


class SizeMismatchError(Exception):
    pass


class GameRolloutConverter:

    def __init__(self, pat_3x3_file, pat_dia_file, features):
        self.feature_processor = RolloutPreprocess(pat_3x3_file, pat_dia_file, features)
        self.n_features = self.feature_processor.output_dim

    def convert_game(self, file_name, bd_size):
        """Read the given SGF file into an iterable of (input,output) pairs
        for neural network training

        Each input is a GameState converted into one-hot neural net features
        Each output is an action as an (x,y) pair (passes are skipped)

        If this game's size does not match bd_size, a SizeMismatchError is raised
        """
        with open(file_name, 'r') as file_object:
            state_action_iterator = sgf_iter_states(file_object.read(), include_end=False)

        for (state, move, player) in state_action_iterator:
            if state.size != bd_size:
                raise SizeMismatchError()
            if move != go.PASS_MOVE:
                nn_input = self.feature_processor.state_to_tensor(state)
                yield (nn_input, move)

    def sgfs_to_hdf5(self, sgf_files, hdf5_file, bd_size=19, ignore_errors=True, verbose=False):
        """Convert all files in the iterable sgf_files into an hdf5 group to be stored in hdf5_file

        Arguments:
        - sgf_files : an iterable of relative or absolute paths to SGF files
        - hdf5_file : the name of the HDF5 where features will be saved
        - bd_size : side length of board of games that are loaded

        - ignore_errors : if True, issues a Warning when there is an unknown
            exception rather than halting. Note that sgf.ParseException and
            go.IllegalMove exceptions are always skipped

        The resulting file has the following properties:
            states  : dataset with shape (n_data, n_features, board width, board height)
            actions : dataset with shape (n_data, 2) (actions are stored as x,y tuples of
                      where the move was played)
            file_offsets : group mapping from filenames to tuples of (index, length)

        For example, to find what positions in the dataset come from 'test.sgf':
            index, length = file_offsets['test.sgf']
            test_states = states[index:index+length]
            test_actions = actions[index:index+length]

        """

        h5 = tables.open_file(hdf5_file, mode="w")

        state_root = h5.create_group(h5.root, 'state')
        data_root = h5.create_group(state_root, 'data')
        indices_root = h5.create_group(state_root, 'indices')
        indptr_root = h5.create_group(state_root, 'indptr')

        action_root = h5.create_group(h5.root, 'action')

        filters = tables.Filters(complevel=5, complib="zlib")

        h5.set_node_attr(h5.root, 'features', ','.join(self.feature_processor.feature_list))

        group_size = 100
        h5.set_node_attr(h5.root, 'group_size', group_size)
        try:
            if verbose:
                print("created HDF5 dataset in {}".format(hdf5_file))

            next_idx = 0
            for file_name in tqdm(sgf_files):
                #if verbose:
                #    print(file_name)
                # count number of state/action pairs yielded by this game
                n_pairs = 0
                try:
                    for state, move in self.convert_game(file_name, bd_size):
                        # assert state.shape == (361, 182789 + 1), 'unexpected shape'
                        # assert state.shape == (361, 7090 + 1), 'unexpected shape'  # non_response_pattern
                        assert state.shape == (361, 175689 + 1), 'unexpected shape'  # response_pattern
                        # assert state.shape == (361, 1 + 1), 'unexpected shape'  # save_atari
                        # assert state.shape == (361, 8 + 1), 'unexpected shape'  # neighbour 
                        if next_idx % group_size == 0:
                            group_id = 'g' + str(next_idx/group_size).rjust(5, '0')
                            current_data = h5.create_group(data_root, group_id,
                                                           filters=filters)
                            current_indices = h5.create_group(indices_root, group_id,
                                                              filters=filters)
                            current_indptr = h5.create_group(indptr_root, group_id,
                                                             filters=filters)
                            current_action = h5.create_group(action_root, group_id,
                                                             filters=filters)

                        name = "s" + str(next_idx).rjust(8, '0')

                        data = state.data
                        data_atom = tables.Atom.from_dtype(data.dtype)
                        data_store = h5.create_carray(current_data, name, data_atom, data.shape)
                        data_store[:] = data

                        indices = state.indices
                        indices_atom = tables.Atom.from_dtype(indices.dtype)
                        indices_store = h5.create_carray(current_indices, name, indices_atom, indices.shape)
                        indices_store[:] = indices

                        indptr = state.indptr
                        indptr_atom = tables.Atom.from_dtype(indptr.dtype)
                        indptr_store = h5.create_carray(current_indptr, name, indptr_atom, indptr.shape)
                        indptr_store[:] = indptr

                        action = np.array(move)
                        action_atom = tables.Atom.from_dtype(action.dtype)
                        action_store = h5.create_carray(current_action, name, action_atom, action.shape)
                        action_store[:] = move

                        n_pairs += 1
                        next_idx += 1
                except go.IllegalMove:
                    warnings.warn("Illegal Move encountered in %s\n"
                                  "\tdropping the remainder of the game" % file_name)
                except sgf.ParseException:
                    warnings.warn("Could not parse %s\n\tdropping game" % file_name)
                except SizeMismatchError:
                    warnings.warn("Skipping %s; wrong board size" % file_name)
                except Exception as e:
                    # catch everything else
                    if ignore_errors:
                        warnings.warn("Unkown exception with file %s\n\t%s" % (file_name, e),
                                      stacklevel=2)
                    else:
                        raise e
                finally:
                    if n_pairs > 0:
                        if verbose:
                            print("\t%d state/action pairs extracted" % n_pairs)
                    elif verbose:
                        print("\t-no usable data-")

            h5.set_node_attr(h5.root, 'size', next_idx-1)
        except Exception as e:
            print("sgfs_to_hdf5 failed")
            os.remove(hdf5_file)
            raise e

        if verbose:
            print("finished.")

        h5.close()


def run_game_converter(cmd_line_args=None):
    """Run conversions. command-line args may be passed in as a list
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Prepare SGF Go game files for training the neural network model.',
        epilog="Available features are: response, save_atari, neighbour, \
                response_pattern, non_response_pattern")
    parser.add_argument("--features", "-f", help="Comma-separated list of features to compute and store or 'all'", default='all')  # noqa: E501
    parser.add_argument("--outfile", "-o", help="Destination to write data (hdf5 file)", required=True)  # noqa: E501
    parser.add_argument("--recurse", "-R", help="Set to recurse through directories searching for SGF files", default=False, action="store_true")  # noqa: E501
    parser.add_argument("--directory", "-d", help="Directory containing SGF files to process. if not present, expects files from stdin", default=None)  # noqa: E501
    parser.add_argument("--size", "-s", help="Size of the game board. SGFs not matching this are discarded with a warning", type=int, default=19)  # noqa: E501
    parser.add_argument("--pat-3x3", help="3x3 pattern file", default=None)
    parser.add_argument("--pat-dia", help="Diamond pattern file", default=None)
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    if args.features.lower() == 'rollout':
        feature_list = [
            "ones",
            "response",
            "save_atari",
            "neighbour",
            "response_pattern",
            "non_response_pattern",
        ]
    elif args.features.lower() == 'tree':
        feature_list = [
            "ones",
            "response",
            "save_atari",
            "neighbour",
            "response_pattern",
            "non_response_pattern"
            "distance"
        ]
    else:
        feature_list = args.features.split(",")

    if args.verbose:
        print("using features", feature_list)

    converter = GameRolloutConverter(args.pat_3x3, args.pat_dia, feature_list)

    def _is_sgf(fname):
        return fname.strip()[-4:] == ".sgf"

    def _walk_all_sgfs(root):
        """a helper function/generator to get all SGF files in subdirectories of root
        """
        for (dirpath, dirname, files) in os.walk(root):
            for filename in files:
                if _is_sgf(filename):
                    # yield the full (relative) path to the file
                    yield os.path.join(dirpath, filename)

    def _list_sgfs(path):
        """helper function to get all SGF files in a directory (does not recurse)
        """
        files = os.listdir(path)
        return (os.path.join(path, f) for f in files if _is_sgf(f))

    # get an iterator of SGF files according to command line args
    if args.directory:
        if args.recurse:
            files = _walk_all_sgfs(args.directory)
        else:
            files = _list_sgfs(args.directory)
    else:
        files = (f.strip() for f in sys.stdin if _is_sgf(f))

    converter.sgfs_to_hdf5(files, args.outfile, bd_size=args.size, verbose=args.verbose)


if __name__ == '__main__':
    run_game_converter()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import warnings
import sgf
import sys
import traceback
import h5py as h5
import itertools
import dill

# cimport numpy as np

import AlphaGo.go as go

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from AlphaGo.preprocessing.rollout_feature import RolloutFeature
from AlphaGo.util import sgf_iter_states


class SizeMismatchError(Exception):
    pass


class GameRolloutConverter(object):

    def __init__(self, bsize=19, x33_file=None, d12_file=None):
        self.bsize = bsize
        self.feature_processor = RolloutFeature(bsize, x33_file, d12_file)
        self.n_features = self.feature_processor.n_features

    def convert_game(self, file_name, feature_tensor):
        """Read the given SGF file into an iterable of (input,output) pairs
        for neural network training

        Each input is a GameState converted into one-hot neural net features
        Each output is an action as an (x,y) pair (passes are skipped)

        If this game's size does not match bsize, a SizeMismatchError is raised
        """
        with open(file_name, 'r') as file_object:
            state_action_iterator = sgf_iter_states(file_object.read(), include_end=False)

        feature_tensor = np.zeros((self.n_features, self.bsize, self.bsize), dtype=np.int)
        for (state, move, player) in state_action_iterator:
            if state.size != self.bsize:
                raise SizeMismatchError()

            if move != go.PASS_MOVE:
                self.feature_processor.update(state, feature_tensor)
                yield (feature_tensor, move)

    def sgfs_to_hdf5(self,
                     sgf_files,
                     hdf5_file,
                     ignore_errors=False, verbose=True):
        """Convert all files in the iterable sgf_files into an hdf5 group to be stored in hdf5_file

        Arguments:
        - sgf_files : an iterable of relative or absolute paths to SGF files
        - hdf5_file : the name of the HDF5 where features will be saved
        - bsize : side length of board of games that are loaded

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

        # make a hidden temporary file in case of a crash.
        # on success, this is renamed to hdf5_file
        tmp_file = os.path.join(os.path.dirname(hdf5_file), ".tmp." + os.path.basename(hdf5_file))
        h5f = h5.File(tmp_file, 'w')

        try:
            # see http://docs.h5py.org/en/latest/high/group.html#Group.create_dataset
            states = h5f.require_dataset(
                'states',
                dtype=np.uint8,
                shape=(1, self.n_features, self.bsize, self.bsize),
                maxshape=(None, self.n_features, self.bsize, self.bsize),  # 'None' == arbitrary size
                exact=False,  # allow non-uint8 datasets to be loaded, coerced to uint8
                chunks=(1, self.n_features, self.bsize, self.bsize),  # approximately 1MB chunks
                compression="lzf")
            actions = h5f.require_dataset(
                'actions',
                dtype=np.uint8,
                shape=(1, 2),
                maxshape=(None, 2),
                exact=False,
                chunks=(1024, 2),
                compression="lzf")

            # 'file_offsets' is an HDF5 group so that 'file_name in file_offsets' is fast
            file_offsets = h5f.require_group('file_offsets')

            if verbose:
                print("created HDF5 dataset in {}".format(tmp_file))

            next_idx = 0
            for file_name in tqdm(sgf_files):

                if verbose:
                    print(file_name)
                # count number of state/action pairs yielded by this game
                n_pairs = 0
                file_start_idx = next_idx
                try:
                    for state, move in self.convert_game(file_name, self.bsize):
                        if next_idx >= len(states):
                            states.resize((next_idx + 1, self.n_features, self.bsize, self.bsize))
                            actions.resize((next_idx + 1, 2))
                        states[next_idx] = state
                        actions[next_idx] = move
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
                    err, msg, _ = sys.exc_info()
                    sys.stderr.write("{} {}\n".format(err, msg))
                    sys.stderr.write(traceback.format_exc())
                    # catch everything else
                    if ignore_errors:
                        warnings.warn("Unkown exception with file %s\n\t%s" % (file_name, e),
                                      stacklevel=2)
                    else:
                        raise e
                finally:
                    if n_pairs > 0:
                        # '/' has special meaning in HDF5 key names, so they
                        # are replaced with ':' here
                        file_name_key = file_name.replace('/', ':')
                        file_offsets[file_name_key] = [file_start_idx, n_pairs]
                        if verbose:
                            print("\t%d state/action pairs extracted" % n_pairs)
                    elif verbose:
                        print("\t-no usable data-")
        except Exception as e:
            print("sgfs_to_hdf5 failed")
            err, msg, _ = sys.exc_info()
            sys.stderr.write("{} {}\n".format(err, msg))
            sys.stderr.write(traceback.format_exc())
            os.remove(tmp_file)
            raise e

        if verbose:
            print("finished. renaming %s to %s" % (tmp_file, hdf5_file))

        # processing complete; rename tmp_file to hdf5_file
        h5f.close()
        os.rename(tmp_file, hdf5_file)


def run_game_converter(cmd_line_args=None):
    """Run conversions. command-line args may be passed in as a list
    """
    import argparse
    import sys

    # cdef np.ndarray[DTYPE_t, ndim=2] feature_tensor

    parser = argparse.ArgumentParser(
        description='Prepare SGF Go game files for training the neural network model.',
        epilog="Available features are: board, ones, turns_since, liberties,\
        capture_size, self_atari_size, liberties_after, sensibleness, and zeros.\
        Ladder features are not currently implemented")
    parser.add_argument("--outfile", "-o", help="Destination to write data (hdf5 file)", required=True)  # noqa: E501
    parser.add_argument("--recurse", "-R", help="Set to recurse through directories searching for SGF files", default=False, action="store_true")  # noqa: E501
    parser.add_argument("--directory", "-d", help="Directory containing SGF files to process. if not present, expects files from stdin", default=None)  # noqa: E501
    parser.add_argument("--size", "-s", help="Size of the game board. SGFs not matching this are discarded with a warning", type=int, default=19)  # noqa: E501
    parser.add_argument("--x33_pattern_file", "-x33", help="3x3 pattern file", default=None)
    parser.add_argument("--d12_pattern_file", "-d12", help="12 point diamond pattern file", default=None)
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    converter = GameRolloutConverter(args.size, args.x33_pattern_file, args.d12_pattern_file)

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
        return [os.path.join(path, f) for f in files if _is_sgf(f)]

    # get an iterator of SGF files according to command line args
    if args.directory:
        if args.recurse:
            files = _walk_all_sgfs(args.directory)
        else:
            files = _list_sgfs(args.directory)
    else:
        files = (f.strip() for f in sys.stdin if _is_sgf(f))

    converter.sgfs_to_hdf5(files, args.outfile, verbose=args.verbose)


if __name__ == '__main__':
    run_game_converter()

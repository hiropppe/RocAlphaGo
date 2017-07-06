#!/usr/bin/env python
import numpy as np
from AlphaGo.preprocessing.preprocessing import Preprocess
from AlphaGo.util import sgf_iter_states, TooFewMove, TooManyMove
import AlphaGo.go as go
import os
import warnings
import sgf
import sys
import traceback
import h5py as h5
import itertools
import dill
import time

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor


class SizeMismatchError(Exception):
    pass


class GameConverter:

    def __init__(self, features):
        self.feature_processor = Preprocess(features)
        self.n_features = self.feature_processor.output_dim
        self.feature_calculation_speeds = []

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
                s = time.time()
                nn_input = self.feature_processor.state_to_tensor(state)
                self.feature_calculation_speeds.append(time.time()-s)
                yield (nn_input, move)

    def sgfs_to_hdf5(self, sgf_files, n_sgf_files, hdf5_file, bd_size=19, ignore_errors=True, verbose=False):
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

        # make a hidden temporary file in case of a crash.
        # on success, this is renamed to hdf5_file
        tmp_file = os.path.join(os.path.dirname(hdf5_file), ".tmp." + os.path.basename(hdf5_file))
        h5f = h5.File(tmp_file, 'w')

        try:
            # see http://docs.h5py.org/en/latest/high/group.html#Group.create_dataset
            states = h5f.require_dataset(
                'states',
                dtype=np.uint8,
                shape=(1, self.n_features, bd_size, bd_size),
                maxshape=(None, self.n_features, bd_size, bd_size),  # 'None' == arbitrary size
                exact=False,  # allow non-uint8 datasets to be loaded, coerced to uint8
                chunks=(64, self.n_features, bd_size, bd_size),  # approximately 1MB chunks
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

            # Store comma-separated list of feature planes in the scalar field 'features'. The
            # string can be retrieved using h5py's scalar indexing: h5f['features'][()]
            h5f['features'] = np.string_(','.join(self.feature_processor.feature_list))

            if verbose:
                print("created HDF5 dataset in {}".format(tmp_file))

            next_idx = 0
            n_parse_error = 0
            n_not19 = 0
            n_too_few_move = 0
            n_too_many_move = 0
            pbar = tqdm(total=n_sgf_files)
            for file_name in sgf_files:
                if verbose:
                    print(file_name)
                # count number of state/action pairs yielded by this game
                n_pairs = 0
                file_start_idx = next_idx
                try:
                    for state, move in self.convert_game(file_name, bd_size):
                        if next_idx >= len(states):
                            states.resize((next_idx + 1, self.n_features, bd_size, bd_size))
                            actions.resize((next_idx + 1, 2))
                        states[next_idx] = state
                        actions[next_idx] = move
                        n_pairs += 1
                        next_idx += 1
                except go.IllegalMove:
                    warnings.warn("Illegal Move encountered in %s\n"
                                  "\tdropping the remainder of the game" % file_name)
                except sgf.ParseException:
                    n_parse_error += 1
                    warnings.warn("Could not parse %s\n\tdropping game" % file_name)
                    if verbose:
                        err, msg, _ = sys.exc_info()
                        sys.stderr.write("{} {}\n".format(err, msg))
                        sys.stderr.write(traceback.format_exc())
                except SizeMismatchError:
                    n_not19 += 1
                    warnings.warn("Skipping %s; wrong board size" % file_name)
                except TooFewMove as e:
                    n_too_few_move += 1
                    warnings.warn('Too few move. {:d} less than 50. {:s}'.format(e.n_moves, file_name))
                except TooManyMove as e:
                    n_too_many_move += 1
                    warnings.warn('Too many move. {:d} more than 500. {:s}'.format(e.n_moves, file_name))
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    # catch everything else
                    if ignore_errors:
                        warnings.warn("Unkown exception with file %s\n\t%s" % (file_name, e),
                                      stacklevel=2)
                    else:
                        raise e
                finally:
                    pbar.update(1)
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
            os.remove(tmp_file)
            raise e

        if verbose:
            print("finished. renaming %s to %s" % (tmp_file, hdf5_file))

        print('Feature Calculation Speed: Avg. {:3f} us'.format(np.mean(self.feature_calculation_speeds)*1000*1000))

        # processing complete; rename tmp_file to hdf5_file
        h5f.close()
        os.rename(tmp_file, hdf5_file)


class ParallelGameConverter:

    def __init__(self, features, nb_workers=2):
        self.features = features
        self.feature_processor = Preprocess(features)
        self.n_features = self.feature_processor.output_dim
        self.nb_workers = nb_workers

    def sgfs_to_hdf5(self, sgf_files, hdf5_file, bd_size, ignore_errors=True, verbose=False):
        executor = ProcessPoolExecutor(max_workers=self.nb_workers)
        sgf_files_copy = itertools.tee(sgf_files, self.nb_workers)
        worker_hdf5_files = list()
        for worker_idx in range(self.nb_workers):
            worker_sgf_files = [sgf_file
                                for i, sgf_file
                                in enumerate(sgf_files_copy[worker_idx])
                                if i % self.nb_workers == worker_idx]
            worker_hdf5_file = hdf5_file + '.' + str(worker_idx)
            worker_hdf5_files.append(worker_hdf5_file)
            executor.submit(*pack_function(self.__sgfs_to_hdf5,
                                           worker_sgf_files,
                                           worker_hdf5_file,
                                           bd_size,
                                           ignore_errors,
                                           verbose))

        try:
            executor.shutdown()
        except:
            executor.shutdown(wait=False)

        self.__merge_hdf5(hdf5_file, worker_hdf5_files, bd_size)

    def __sgfs_to_hdf5(self, sgf_files, hdf5_file, bd_size, ignore_errors, verbose):
        converter = GameConverter(self.features)
        converter.sgfs_to_hdf5(sgf_files, hdf5_file, bd_size, ignore_errors, verbose)

    def __merge_hdf5(self, hdf5_file, worker_hdf5_files, bd_size):
        h5f = h5.File(hdf5_file, 'w')

        states = h5f.require_dataset(
            'states',
            dtype=np.uint8,
            shape=(1, self.n_features, bd_size, bd_size),
            maxshape=(None, self.n_features, bd_size, bd_size),
            exact=False,
            chunks=(64, self.n_features, bd_size, bd_size),
            compression="lzf")
        actions = h5f.require_dataset(
            'actions',
            dtype=np.uint8,
            shape=(1, 2),
            maxshape=(None, 2),
            exact=False,
            chunks=(1024, 2),
            compression="lzf")

        h5f['features'] = np.string_(','.join(self.feature_processor.feature_list))

        next_idx = 0
        for worker_hdf5_file in worker_hdf5_files:
            worker_h5f = h5.File(worker_hdf5_file, 'r')

            assert worker_h5f['states'].len() == worker_h5f['actions'].len()

            data_len = worker_h5f['states'].len()
            for data_idx in range(data_len):
                if next_idx >= len(states):
                    states.resize((next_idx + 1, self.n_features, bd_size, bd_size))
                    actions.resize((next_idx + 1, 2))
                states[next_idx] = worker_h5f['states'][data_idx]
                actions[next_idx] = worker_h5f['actions'][data_idx]
                next_idx += 1

            worker_h5f.close()

        h5f.close()


def apply_packed_function((dumped_function, item, args, kwargs),):
    target_function = dill.loads(dumped_function)
    res = target_function(item, *args, **kwargs)
    return res


def pack_function(target_function, item, *args, **kwargs):
    dumped_function = dill.dumps(target_function)
    return apply_packed_function, (dumped_function, item, args, kwargs)


def run_game_converter(cmd_line_args=None):
    """Run conversions. command-line args may be passed in as a list
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Prepare SGF Go game files for training the neural network model.',
        epilog="Available features are: board, ones, turns_since, liberties,\
        capture_size, self_atari_size, liberties_after, sensibleness, and zeros.\
        Ladder features are not currently implemented")
    parser.add_argument("--features", "-f", help="Comma-separated list of features to compute and store or 'all'", default='all')  # noqa: E501
    parser.add_argument("--outfile", "-o", help="Destination to write data (hdf5 file)", required=True)  # noqa: E501
    parser.add_argument("--recurse", "-R", help="Set to recurse through directories searching for SGF files", default=False, action="store_true")  # noqa: E501
    parser.add_argument("--directory", "-d", help="Directory containing SGF files to process. if not present, expects files from stdin", default=None)  # noqa: E501
    parser.add_argument("--size", "-s", help="Size of the game board. SGFs not matching this are discarded with a warning", type=int, default=19)  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    parser.add_argument("--workers", "-w", help="Number of workers to process SGF files", type=int, default=0)  # noqa: E501

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    if args.features.lower() == 'all':
        feature_list = [
            "board",
            "ones",
            "turns_since",
            "liberties",
            "capture_size",
            "self_atari_size",
            "liberties_after",
            "ladder_capture",
            "ladder_escape",
            "sensibleness",
            "zeros"]
    else:
        feature_list = args.features.split(",")

    if args.verbose:
        print("using features", feature_list)

    if args.workers:
        converter = ParallelGameConverter(feature_list, args.workers)
    else:
        converter = GameConverter(feature_list)

    def _is_sgf(fname):
        return fname.strip()[-4:] == ".sgf"

    def _count_all_sgfs(root):
        count = 0
        for (dirpath, dirname, files) in os.walk(root):
            for filename in files:
                if _is_sgf(filename):
                    count += 1
        return count

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
        n_files = _count_all_sgfs(args.directory)
        if args.recurse:
            files = _walk_all_sgfs(args.directory)
        else:
            files = _list_sgfs(args.directory)
    else:
        n_files = 1
        files = (f.strip() for f in sys.stdin if _is_sgf(f))

    converter.sgfs_to_hdf5(files, n_files, args.outfile, bd_size=args.size, verbose=args.verbose)


if __name__ == '__main__':
    run_game_converter()

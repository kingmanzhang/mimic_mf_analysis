import pickle
import argparse
import os.path
from mutual_information.mf_random import MutualInfoRandomizer
import logging.config
import numpy as np

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'log_config.conf')
logging.config.fileConfig(log_file_path)
logger = logging.getLogger(__name__)


def main():
    HOME_DIR = os.path.expanduser('~')
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    simulate_parser = subparser.add_parser('simulate',
                                           help='create empirical distributions')
    simulate_parser.add_argument('-i', '--input', help='input file path',
                        action='store', dest='input_path')
    simulate_parser.add_argument('-o', '--out', help='output directory',
                        action='store', dest='out_dir', default=HOME_DIR)
    simulate_parser.add_argument('-n', '--n_per_simulation',
                        help='sample size per simulation. ignore if to use '
                             'the same size in original data',
                        dest='n_per_run', type=int, default=None)
    simulate_parser.add_argument('-N', '--N_SIMULATIONS', help='total simulations',
                        dest='N_SIMULATIONS', type=int, default=1000)
    simulate_parser.add_argument('-v', '--verbose', help='print messages',
                                 dest='verbose', action='store_true',
                                 default=False)
    # job_id is used as the seed for random number generator
    simulate_parser.add_argument('-job_id', help='job array id (PBS_ARRAYID)',
                        dest='job_id', type=int, default=None)
    simulate_parser.add_argument('-cpu', help='specify the number of available cpu',
                        default=None, type=int, dest='cpu')
    simulate_parser.add_argument('-disease', help='specify if only to analyze such disease',
                                 default=[], dest='disease_of_interest',
                                 type=str)
    simulate_parser.set_defaults(func=simulate)

    estimate_parser = subparser.add_parser('estimate',
           help='estimate p value from empirical distributions')
    estimate_parser.add_argument('-i', '--input', help='input file path',
                                action='store', dest='input_path')
    estimate_parser.add_argument('-dist', '--dist_path',
                                 help='empirical distribution path',
                                 action='store', dest='dist_path')
    estimate_parser.add_argument('-o', '--ouput', help='output dir',
                                 action='store', dest='out_dir')
    estimate_parser.add_argument('-disease', help='specify if only to analyze such disease',
                                 default=[], dest='disease_of_interest',
                                 type=str)
    estimate_parser.set_defaults(func=estimate)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


def simulate(args):
    input_path = args.input_path
    per_simulation = args.n_per_run
    simulations = args.N_SIMULATIONS
    verbose = args.verbose
    dir = args.out_dir
    cpu = args.cpu
    job_id = args.job_id
    disease_of_interest = args.disease_of_interest

    with open(input_path, 'rb') as in_file:
        disease_synergy_map = pickle.load(in_file)
        logger.info('number of diseases to run simulations for {}'.format(
            len(disease_synergy_map)))

    # make sure output directory exists, including intermediate directories
    os.makedirs(dir, exist_ok=True)

    if job_id is None:
        job_suffix = ''
    else:
        job_suffix = '_' + str(job_id)

    for disease, synergy in disease_synergy_map.items():
        if disease_of_interest is not None and \
                        disease not in disease_of_interest:
            continue
        randmizer = MutualInfoRandomizer(synergy)
        if verbose:
            print('start calculating p values for {}'.format(disease))
        randmizer.simulate(per_simulation, simulations, cpu, job_id)
        # p = randmizer.p_value()
        # p_filepath = os.path.join(dir, disease + '_p_value_.obj')
        # with open(p_filepath, 'wb') as f:
        #     pickle.dump(p, file=f, protocol=2)

        distribution_file_path = os.path.join(dir, disease + job_suffix +
                                              '_distribution.obj')
        with open(distribution_file_path, 'wb') as f2:
            pickle.dump(randmizer.empirical_distribution, file=f2, protocol=2)

        if verbose:
            print('saved current batch of simulations {} for {}'.format(
                job_id, disease))


def estimate(args):
    input_path = args.input_path
    dist_path = args.dist_path
    out_path = args.out_dir
    disease_of_interest = args.disease_of_interest

    print(args)
    with open(input_path, 'rb') as in_file:
        mf_map = pickle.load(in_file)
        logger.info('number of diseases to run simulations for {}'.format(
            len(mf_map)))

    for disease, summary_statistics in mf_map.items():
        if disease_of_interest is not None and \
                        disease not in disease_of_interest:
            continue
        randmizer = MutualInfoRandomizer(summary_statistics)
        empirical_distribution = load_distribution(dist_path, disease)
        # serialize_empirical_distributions(empirical_distribution['synergy'],
        #      os.path.join(out_path, disease +
        #                   '_empirical_distribution_subset.obj'))
        randmizer.empirical_distribution = empirical_distribution
        p = randmizer.p_values()

    with open(out_path, 'wb') as f:
        pickle.dump(p, f, protocol=2)
    return p


def load_distribution(dir, disease_prefix):
    """
    Collect individual distribution profiles
    :param dir:
    :param disease:
    :return:
    """
    simulations = []
    for i in np.arange(5000):
        path = os.path.join(dir, disease_prefix + '_' + str(i) +
                            '_distribution.obj')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                try:
                    simulation = pickle.load(f)
                    simulations.append(simulation)
                except:
                    pass

    empirical_distributions = dict()
    empirical_distributions['mf_XY_omit_z'] = \
        np.concatenate([res['mf_XY_omit_z'] for res in simulations], axis=-1)
    empirical_distributions['mf_Xz'] = \
        np.concatenate([res['mf_Xz'] for res in simulations], axis=-1)
    empirical_distributions['mf_Yz'] = \
        np.concatenate([res['mf_Yz'] for res in simulations], axis=-1)
    empirical_distributions['mf_XY_z'] = \
        np.concatenate([res['mf_XY_z'] for res in simulations], axis=-1)
    empirical_distributions['mf_XY_given_z'] = \
        np.concatenate([res['mf_XY_given_z'] for res in simulations], axis=-1)
    empirical_distributions['synergy'] = \
        np.concatenate([res['synergy'] for res in simulations], axis=-1)

    return empirical_distributions


def serialize_empirical_distributions(distribution, path):
    M1 = distribution.shape[0]
    M2 = distribution.shape[1]
    N = distribution.shape[2]
    sampling_1d_size = min([M1, M2, 5])
    i_index = np.random.choice(np.arange(M1), sampling_1d_size, replace=False)
    j_index = np.random.choice(np.arange(M2), sampling_1d_size, replace=False)
    sampled_empirical_distributions = np.zeros([sampling_1d_size,
                                                sampling_1d_size, N])
    for i in np.arange(sampling_1d_size):
        for j in np.arange(sampling_1d_size):
            sampled_empirical_distributions[i, j, :] = \
                distribution[i_index[i],j_index[j], :]

    with open(path, 'wb') as f:
        pickle.dump(sampled_empirical_distributions, file=f, protocol=2)


if __name__=='__main__':
    main()

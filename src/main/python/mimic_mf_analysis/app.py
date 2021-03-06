import click
import numpy as np
import pandas as pd
from mutual_information.mf_random import MutualInfoRandomizer

from mimic_mf_analysis import mydb
import mimic_mf_analysis.analysis as analysis
import logging
from mutual_information.synergy_tree import SynergyTree
import pathlib
import yaml
import pickle
import os
import glob


logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


def parse_yaml(analysis_config_yaml_path):
    if not analysis_config_yaml_path:
        analysis_config_yaml_path = pathlib.Path(__name__).parent.joinpath("resource", "analysisConfig.yaml")
    with open(analysis_config_yaml_path, 'r') as f:
        analysis_config = yaml.load(f)
    logger.info(analysis_config)
    return analysis_config


@click.command()
@click.option("--analysis_config_yaml_path", help="analysis configuration file")
@click.option("--debug", is_flag=True, help="run in debug mode")
@click.option("--out", help="output directory")
def regardless_diagnosis(analysis_config_yaml_path, debug, out):
    """
    Generate the joint distribution of HPO pairs regardless of diseases.
    Terms of HPO pairs can be 1) one from rad and one from lab 2) both from rad or 3) both from lab
    """
    # how to run this
    analysis_config = parse_yaml(analysis_config_yaml_path=analysis_config_yaml_path)

    if debug:
        analysis_params = analysis_config['analysis-test']['regardless_of_diseases']
        logger.warning("running in debug mode")
    else:
        analysis_params = analysis_config['analysis-prod']['regardless_of_diseases']
        logger.info("running in prod mode")

    textHpo_occurrance_min, labHpo_occurrance_min = analysis_params['textHpo_occurrance_min'], analysis_params[
        'labHpo_occurrance_min']
    textHpo_threshold_min, textHpo_threshold_max = analysis_params['textHpo_threshold_min'], analysis_params[
        'textHpo_threshold_max']
    labHpo_threshold_min, labHpo_threshold_max = analysis_params['labHpo_threshold_min'], analysis_params[
        'labHpo_threshold_max']

    analysis.initTables(debug=debug)
    analysis.rankHpoFromText('', hpo_min_occurrence_per_encounter=textHpo_occurrance_min)
    analysis.rankHpoFromLab('', hpo_min_occurrence_per_encounter=labHpo_occurrance_min)

    batch_size = 11 if debug else 100

    summary_rad_lab, summary_rad_rad, summary_lab_lab = analysis.summary_textHpo_labHpo(batch_size,
                                                                               textHpo_occurrance_min,
                                                                               labHpo_occurrance_min,
                                                                               textHpo_threshold_min,
                                                                               textHpo_threshold_max,
                                                                               labHpo_threshold_min,
                                                                               labHpo_threshold_max)

    if out:
        out_dir = pathlib.Path(out)
    else:
        print("out directory not specified. default to mimic_analysis in home directory")
        out_dir = pathlib.Path().home().joinpath('mimic_analysis')
        if not out_dir.exists():
            out_dir.mkdir()

    with open(out_dir.joinpath("summary_rad_lab.obj"), 'wb') as f:
        pickle.dump(summary_rad_lab, f, protocol=2)
    with open(out_dir.joinpath("summary_rad_rad.obj"), 'wb') as f:
        pickle.dump(summary_rad_rad, f, protocol=2)
    with open(out_dir.joinpath("summary_lab_lab.obj"), 'wb') as f:
        pickle.dump(summary_lab_lab, f, protocol=2)


@click.command()
@click.option("--analysis_config_yaml_path", help="analysis configuration file")
@click.option("--debug", is_flag=True, help="run in debug mode")
@click.option("--out", help="output directory")
def regarding_diagnosis(analysis_config_yaml_path, debug, out):
    """
    Count the joint distribution of HPO pairs conditioned on a disease.
    Terms in the HPO pair could be 1) one from rad and one from lab, 2) both from rad or 3) both from lab.
    """
    # how to run this
    analysis_config = parse_yaml(analysis_config_yaml_path=analysis_config_yaml_path)

    if debug:
        analysis_params = analysis_config['analysis-test']['regarding_diagnosis']
        logger.warning("running in debug mode")
    else:
        analysis_params = analysis_config['analysis-prod']['regarding_diagnosis']
        logger.info("running in prod mode")

    primary_diagnosis_only = analysis_params['primary_diagnosis_only']
    diagnosis_threshold_min = analysis_params['diagnosis_threshold_min']
    textHpo_occurrance_min, labHpo_occurrance_min = analysis_params['textHpo_occurrance_min'], analysis_params['labHpo_occurrance_min']
    textHpo_threshold_min, textHpo_threshold_max = analysis_params['textHpo_threshold_min'], analysis_params['textHpo_threshold_max']
    labHpo_threshold_min, labHpo_threshold_max = analysis_params['labHpo_threshold_min'], analysis_params['labHpo_threshold_max']
    disease_of_interest = analysis_params['disease_of_interest']

    # 1. build the temp tables for Lab converted HPO, Text convert HPO
    # Read the comments within the method!
    analysis.initTables(debug=debug)

    # 2. iterate throw the dataset
    summaries_diag_textHpo_labHpo, summaries_diag_textHpo_textHpo, summaries_diag_labHpo_labHpo = analysis.summarize_diagnosis_textHpo_labHpo(
        primary_diagnosis_only, textHpo_occurrance_min, labHpo_occurrance_min, diagnosis_threshold_min,
        textHpo_threshold_min, textHpo_threshold_max, labHpo_threshold_min, labHpo_threshold_max, disease_of_interest,
        logger)

    if out:
        out_dir = pathlib.Path(out)
    else:
        print("out directory not specified. default to mimic_analysis in home directory")
        out_dir = pathlib.Path().home().joinpath('mimic_analysis')
        if not out_dir.exists():
            out_dir.mkdir()

    with open(out_dir.joinpath("summaries_diag_rad_lab.obj"), 'wb') as f:
        print("write summaries_diag_rad_lab.obj ")
        pickle.dump(summaries_diag_textHpo_labHpo, f, protocol=2)
    with open(out_dir.joinpath("summaries_diag_rad_rad.obj"), 'wb') as f:
        pickle.dump(summaries_diag_textHpo_textHpo, f, protocol=2)
    with open(out_dir.joinpath("summaries_diag_lab_lab.obj"), 'wb') as f:
        pickle.dump(summaries_diag_labHpo_labHpo, f, protocol=2)


@click.command()
@click.option("--analysis_config_yaml_path", help="analysis configuration file")
@click.option("--debug", is_flag=True, help="run in debug mode")
@click.option("--out", help="output directory")
def build_synergy_tree(analysis_config_yaml_path, debug, out):
    # how to run this
    analysis_config = parse_yaml(analysis_config_yaml_path=analysis_config_yaml_path)

    if debug:
        analysis_params = analysis_config['analysis-test']['synergy_tree']
        logger.warning("running in debug mode")
    else:
        analysis_params = analysis_config['analysis-prod']['synergy_tree']
        logger.info("running in prod mode")

    diagnosis = analysis_params['disease_of_interest']

    textHpo_occurrance_min, labHpo_occurrance_min = analysis_params['textHpo_occurrance_min'], analysis_params[
        'labHpo_occurrance_min']
    textHpo_threshold_min, textHpo_threshold_max = analysis_params['textHpo_threshold_min'], analysis_params[
        'textHpo_threshold_max']
    labHpo_threshold_min, labHpo_threshold_max = analysis_params['labHpo_threshold_min'], analysis_params[
        'labHpo_threshold_max']

    primary_diagnosis_only = analysis_params['primary_diagnosis_only']

    analysis.initTables(debug=debug)

    analysis.rankHpoFromText(diagnosis, textHpo_occurrance_min)
    analysis.rankHpoFromLab(diagnosis, labHpo_occurrance_min)
    # logger.info("..............diagnosis values found")

    textHpoOfInterest = pd.read_sql_query(
        "SELECT * FROM JAX_textHpoFrequencyRank WHERE N BETWEEN {} AND {}".format(textHpo_threshold_min,
                                                                                  textHpo_threshold_max),
        mydb).MAP_TO.values
    labHpoOfInterest = pd.read_sql_query(
        "SELECT * FROM JAX_labHpoFrequencyRank WHERE N BETWEEN {} AND {}".format(labHpo_threshold_min,
                                                                                 labHpo_threshold_max),
        mydb).MAP_TO.values
    # manually trim phenotypes TODO: further filter them
    # there is probably not a good way to automate this
    print(labHpoOfInterest)
    print(textHpoOfInterest)
    textHpoOfInterest = ['HP:0001877', 'HP:0020058', 'HP:0010927', 'HP:0001871', 'HP:0010929']
    labHpoOfInterest = ['HP:0002202', 'HP:0011032', 'HP:0100750']
    print("filtered phenotypes:")
    print(labHpoOfInterest)
    print(textHpoOfInterest)

    mydb.cursor().execute("""drop table if exists Jax_multivariant_synergy_table""")

    analysis.add_diag_columns(diagnosis, primary_diagnosis_only)
    var_dict = analysis.add_phenotype_columns(labHpos=labHpoOfInterest, \
                                     textHpos=textHpoOfInterest, \
                                     labHpo_threshold_min=labHpo_occurrance_min, \
                                     textHpo_threshold_min=textHpo_occurrance_min)
    mf_dict, summary_dict = analysis.precompute_mf_dict(var_dict.keys())
    syntree_038 = SynergyTree(var_dict.keys(), var_dict, mf_dict)
    syntree_038.synergy_tree().show()
    print(var_dict)


@click.command()
@click.option("--joint_distributions_path", help="HPO pair * disease joint distributions (output from running previous command)")
@click.option("--disease_of_interest", required=True, help="specify a disease to run simulations for")
@click.option("--out_dir", help="specify output directory")
@click.option("--verbose", is_flag=True, help="print more log info in verbose mode")
@click.option("--per_simulation", default=8, help="for every simulation, how many encounters to simulate (roughly equal to observed encounters)")
@click.option("--simulations", default=500, help="how many simulations to repeat")
@click.option("--cpu", default=8, help="number of CPU to use")
@click.option("--job_id", default=1, help="pass job id")
def simulate(joint_distributions_path, disease_of_interest, out_dir, verbose, per_simulation, simulations, cpu, job_id):
    """
    Provide the joint distributions of disease*HPO_pair, and run simulations
    """
    with open(joint_distributions_path, 'rb') as in_file:
        joint_distributions = pickle.load(in_file)
        logger.info('number of diseases in input file for joint distributions {}'.format(len(joint_distributions)))

    if out_dir is None:
        out_dir = pathlib.Path(joint_distributions_path).parent

    if job_id is None:
        job_suffix = ''
    else:
        job_suffix = '_' + str(job_id)

    joint_distribution = joint_distributions.get(disease_of_interest)
    if joint_distribution is None:
        raise RuntimeError("specified disease not included in the joint_distribution file. exit without simulation.")
    else:
        randmizer = MutualInfoRandomizer(joint_distribution)
        if verbose:
            print('start calculating p values for {}'.format(disease_of_interest))
        randmizer.simulate(per_simulation, simulations, cpu, job_id)

        distribution_file_path = os.path.join(out_dir, disease_of_interest + job_suffix + '_distribution.obj')
        with open(distribution_file_path, 'wb') as f2:
            pickle.dump(randmizer.empirical_distribution, file=f2, protocol=2)

        if verbose:
            print('saved current batch of simulations {} for {}'.format(job_id, disease_of_interest))


@click.command()
@click.option("--joint_distributions_path", help="HPO pair * disease joint distributions (output from running previous command)")
@click.option("--dist_path", help="directory path for simulation results")
@click.option("--out_path", help="output path, return a binary file of Python map")
@click.option("--disease_of_interest", help="specify a disease name")
def estimate(joint_distributions_path, dist_path, out_path, disease_of_interest):
    with open(joint_distributions_path, 'rb') as in_file:
        joint_distributions = pickle.load(in_file)
        logger.info('number of diseases in input file for joint distributions {}'.format(len(joint_distributions)))

    for disease, summary_statistics in joint_distributions.items():
        if disease_of_interest is not None and disease not in disease_of_interest:
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
    """
    simulations = []
    filename_pattern = f'{dir}/{disease_prefix}_*_distribution.obj'
    for path in glob.glob(filename_pattern):
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


cli.add_command(regardless_diagnosis)
cli.add_command(regarding_diagnosis)
cli.add_command(build_synergy_tree)
cli.add_command(simulate)
cli.add_command(estimate)


if __name__=='__main__':
    cli()
import click
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
import re


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
def build_synergy_tree():
    diagnosis = '038'
    textHpo_occurrance_min = 1
    labHpo_occurrance_min = 3
    textHpo_threshold_min = 7
    textHpo_threshold_max = 7
    labHpo_threshold_min = 8
    labHpo_threshold_max = 8
    primary_diagnosis_only = True

    analysis.initTables(debug=True)

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
@click.option("--diseases_of_interest", help="specify diseases to run simulations for, separated by comma")
@click.option("--out_dir", help="specify output directory")
@click.option("--verbose", is_flag=True, help="print more log info in verbose mode")
@click.option("--per_simulation", default=8, help="for every simulation, how many times to randomly sample")
@click.option("--simulations", default=500, help="how many simulations")
@click.option("--cpu", default=8, help="number of CPU to use")
@click.option("--job_id", default=1, help="pass job id")
def simulate(joint_distributions_path, diseases_of_interest, out_dir, verbose, per_simulation, simulations, cpu, job_id):
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

    diseases_of_interest = re.split(',\\s*', diseases_of_interest)

    # TODO: the following works as if disease_of_interest could be multiple diseases
    for disease, joint_distribution in joint_distributions.items():
        if diseases_of_interest is not None and disease not in diseases_of_interest:
            continue
        randmizer = MutualInfoRandomizer(joint_distribution)
        if verbose:
            print('start calculating p values for {}'.format(disease))
        randmizer.simulate(per_simulation, simulations, cpu, job_id)

        distribution_file_path = os.path.join(out_dir, disease + job_suffix + '_distribution.obj')
        with open(distribution_file_path, 'wb') as f2:
            pickle.dump(randmizer.empirical_distribution, file=f2, protocol=2)

        if verbose:
            print('saved current batch of simulations {} for {}'.format(job_id, disease))


@click.command()
def estimate():
    print("doing estimation")


cli.add_command(regardless_diagnosis)
cli.add_command(regarding_diagnosis)
cli.add_command(build_synergy_tree)
cli.add_command(simulate)
cli.add_command(estimate)


if __name__=='__main__':
    cli()
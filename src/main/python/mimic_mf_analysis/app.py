import click
import pandas as pd
from mimic_mf_analysis import mydb
import mimic_mf_analysis.preparation as preparation
import mimic_mf_analysis.analysis as analysis
import logging
from mutual_information.mf import MutualInfoXYz
from mutual_information.synergy_tree import SynergyTree
import pathlib
import yaml


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
@click.option("--debug", default=False, help="run in debug mode")
def mutual_info(analysis_config_yaml_path, debug):
    # how to run this
    analysis_config = parse_yaml(analysis_config_yaml_path=analysis_config_yaml_path)

    if debug:
        analysis_params = analysis_config['analysis-test']['regarding_diagnosis']
        logger.warning("running in debug mode")
    else:
        analysis_params = analysis_config['analysis-test']['regarding_diagnosis']
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
    print(summaries_diag_textHpo_labHpo)
    print(summaries_diag_textHpo_labHpo.get('038'))
    mf_XYz = MutualInfoXYz(summaries_diag_textHpo_labHpo.get('038'))
    print(mf_XYz.mutual_info_Xz())


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
def simulate():
    print(pd.read_sql('select * from labevents limit 4', mydb))
    preparation.encounterOfInterest(debug=True)


@click.command()
def estimate():
    print("doing estimation")


cli.add_command(mutual_info)
cli.add_command(build_synergy_tree)
cli.add_command(simulate)
cli.add_command(estimate)


if __name__=='__main__':
    cli()
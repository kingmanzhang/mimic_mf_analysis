import yaml
import mysql.connector
from logging import getLogger
from logging.config import fileConfig
from obonetx.ontology import Ontology
import pathlib

resource_dir = pathlib.Path(__file__).parent.joinpath('resource')
log_config_path = resource_dir.joinpath('logging_config.ini')
fileConfig(log_config_path, disable_existing_loggers=False)
logger = getLogger(__name__)

# load yaml configuration file
with open(resource_dir.joinpath('analysisConfig.yaml'), 'r') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

base_dir = config['base_dir']
hpo_obo_path = config['hp.obo.path']
hpo = Ontology(hpo_obo_path)

# set up MySql connection
host = config['database']['host']
user = config['database']['user']
password = config['database']['password']
database = config['database']['database']

mydb = mysql.connector.connect(host=host,
                               user=user,
                               passwd=password,
                               database=database,
                               auth_plugin='mysql_native_password')
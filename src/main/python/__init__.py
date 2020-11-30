import yaml
import mysql.connector
import logging
from obonetx.ontology import Ontology

# load yaml configuration file
with open('analysisConfig.yaml', 'r') as yaml_file:
    config = yaml.load(yaml_file)

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
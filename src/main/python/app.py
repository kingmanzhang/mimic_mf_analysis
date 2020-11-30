import click
import pandas as pd
from mysql.connector import connect


def connection():
    config = {'user': 'mimicuser',
              'password': 'mimic',
              'host': 'localhost',
              'database': 'mimiciiiv13',
              'auth_plugin': 'mysql_native_password'}
    return connect(**config)


@click.group()
def cli():
    pass


@click.command()
def mutual_info():
    print("calculate mutual information")


@click.command()
def simulate():
    dbcon = connection()
    print(pd.read_sql('select * from labevents limit 4', dbcon))


@click.command()
def estimate():
    print("doing estimation")


cli.add_command(mutual_info)
cli.add_command(simulate)
cli.add_command(estimate)


if __name__=='__main__':
    cli()
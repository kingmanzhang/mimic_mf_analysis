import numpy as np
import pandas as pd
import math
import mutual_information.mf as mf
from .preparation import encounterOfInterest, indexEncounterOfInterest, diagnosisProfile, rankICD, rankHpoFromLab, rankHpoFromText,
from tqdm import tqdm

from . import mydb

cursor = mydb.cursor(buffered=True)


# mutual information between radiology and lab

def createDiagnosisTable(diagnosis, primary_diagnosis_only):
    """
    Create a temporary table JAX_mf_diag. For encounters of interest, assign 0 or 1 to each encouter whether a diagnosis is observed.
    @param diagnosis: diagnosis code. An encounter is considered to be 1 if same or more detailed code is called.
    @prarm primary_diagnosis_only: an encounter may be associated with one primary diagnosis and many secondary ones.
    if value is set true, only primary diagnosis counts.
    """
    cursor.execute('DROP TEMPORARY TABLE IF EXISTS JAX_mf_diag')
    if primary_diagnosis_only:
        limit = 'AND SEQ_NUM=1'
    else:
        limit = ''
    cursor.execute('''
                CREATE TEMPORARY TABLE IF NOT EXISTS JAX_mf_diag 
                WITH 
                    d AS (
                        SELECT 
                            DISTINCT SUBJECT_ID, HADM_ID, '1' AS DIAGNOSIS
                        FROM 
                            JAX_diagnosisProfile 
                        WHERE ICD9_CODE LIKE '{}%' {})
                    -- This is encounters with positive diagnosis

                SELECT 
                    DISTINCT a.SUBJECT_ID, a.HADM_ID, IF(d.DIAGNOSIS IS NULL, '0', '1') AS DIAGNOSIS
                FROM 
                    JAX_encounterOfInterest AS a
                LEFT JOIN
                    d ON a.SUBJECT_ID = d.SUBJECT_ID AND a.HADM_ID = d.HADM_ID       
                /* -- This is the first join for diagnosis (0, or 1) */    
                '''.format(diagnosis, limit))
    cursor.execute('CREATE INDEX JAX_mf_diag_idx01 ON JAX_mf_diag (SUBJECT_ID, HADM_ID)')


def diagnosisTextHpo(phenotype):
    """
    Assign 0 or 1 to each encounter whether a phenotype is observed from radiology reports
    @phenotype: an HPO term id
    """
    cursor.execute('DROP TEMPORARY TABLE IF EXISTS JAX_mf_diag_textHpo')
    """
    cursor.execute('''
        CREATE TEMPORARY TABLE JAX_mf_diag_textHpo
        SELECT 
            L.*, IF(R.MAP_TO IS NULL, '0', '1') AS PHEN_TXT
        FROM JAX_mf_diag AS L 
        LEFT JOIN 
            (SELECT * 
            FROM JAX_textHpoProfile 
            WHERE JAX_textHpoProfile.MAP_TO = '{}') AS R 
        ON L.SUBJECT_ID = R.SUBJECT_ID AND L.HADM_ID = R.HADM_ID 
    '''.format(phenotype))
    """
    cursor.execute('''
        CREATE TEMPORARY TABLE JAX_mf_diag_textHpo
        WITH L AS (SELECT JAX_mf_diag.*, '{}' AS PHEN_TXT FROM JAX_mf_diag)
        SELECT 
            L.*, IF(R.dummy IS NULL, '0', '1') AS PHEN_TXT_VALUE
        FROM L 
        LEFT JOIN 
            JAX_textHpoProfile AS R
        ON L.SUBJECT_ID = R.SUBJECT_ID AND L.HADM_ID = R.HADM_ID AND L.PHEN_TXT = R.MAP_TO
    '''.format(phenotype))
    cursor.execute('CREATE INDEX JAX_mf_diag_textHpo_idx01 ON JAX_mf_diag_textHpo (SUBJECT_ID, HADM_ID)')


def diagnosisAllTextHpo(threshold_min, threshold_max):
    """
    For phenotypes of interest, defined with two parameters, assign 0 or 1 to each encounter whether a phenotype is observated from text data
    @param threshold_min: minimum threshold of encounter count for a phenotype to be interesting.
    @param threshold_max: maximum threshold of encounter count for a phenotype to be interesting.
    """
    cursor.execute('DROP TEMPORARY TABLE IF EXISTS JAX_mf_diag_allTextHpo')
    cursor.execute('''
        CREATE TEMPORARY TABLE JAX_mf_diag_allTextHpo
        WITH 
            P AS (SELECT MAP_TO AS PHEN_TXT FROM JAX_textHpoFrequencyRank WHERE N BETWEEN {} AND {}),
            L AS (SELECT * FROM JAX_mf_diag JOIN P)
        SELECT 
            L.*, IF(R.dummy IS NULL, '0', '1') AS PHEN_TXT_VALUE
        FROM L 
        LEFT JOIN 
            JAX_textHpoProfile AS R
        ON L.SUBJECT_ID = R.SUBJECT_ID AND L.HADM_ID = R.HADM_ID AND L.PHEN_TXT = R.MAP_TO
    '''.format(threshold_min, threshold_max))
    cursor.execute(
        'CREATE INDEX JAX_mf_diag_allTextHpo_idx01 ON JAX_mf_diag_allTextHpo (SUBJECT_ID, HADM_ID, PHEN_TXT)')


def diagnosisLabHpo(phenotype):
    """
    Assign 0 or 1 to each encounter whether a phenotype is observed from lab tests
    @phenotype: an HPO term id
    """
    cursor.execute('DROP TEMPORARY TABLE IF EXISTS JAX_mf_diag_labHpo')
    cursor.execute('''
        CREATE TEMPORARY TABLE JAX_mf_diag_labHpo
        WITH L AS (SELECT JAX_mf_diag.*, '{}' AS PHEN_LAB FROM JAX_mf_diag)
        SELECT 
            L.*, IF(R.dummy IS NULL, '0', '1') AS PHEN_LAB_VALUE
        FROM L 
        LEFT JOIN 
             JAX_labHpoProfile AS R 
        ON L.SUBJECT_ID = R.SUBJECT_ID AND L.HADM_ID = R.HADM_ID AND L.PHEN_LAB = R.MAP_TO
    '''.format(phenotype))
    cursor.execute('CREATE INDEX JAX_mf_diag_labHpo_idx01 ON JAX_mf_diag_labHpo (SUBJECT_ID, HADM_ID)')


def diagnosisAllLabHpo(threshold_min, threshold_max):
    """
    For phenotypes of interest, defined with two parameters, assign 0 or 1 to each encounter whether a phenotype is observated from lab tests
    @param threshold_min: minimum threshold of encounter count for a phenotype to be interesting.
    @param threshold_max: maximum threshold of encounter count for a phenotype to be interesting.
    """
    cursor.execute('DROP TEMPORARY TABLE IF EXISTS JAX_mf_diag_allLabHpo')
    cursor.execute('''
        CREATE TEMPORARY TABLE JAX_mf_diag_allLabHpo
        WITH 
            P AS (SELECT MAP_TO AS PHEN_LAB FROM JAX_labHpoFrequencyRank WHERE N BETWEEN {} AND {}),
            L AS (SELECT * FROM JAX_mf_diag JOIN P)
        SELECT 
            L.*, IF(R.dummy IS NULL, '0', '1') AS PHEN_LAB_VALUE
        FROM L 
        LEFT JOIN 
             JAX_labHpoProfile AS R 
        ON L.SUBJECT_ID = R.SUBJECT_ID AND L.HADM_ID = R.HADM_ID AND L.PHEN_LAB = R.MAP_TO
    '''.format(threshold_min, threshold_max))
    cursor.execute('CREATE INDEX JAX_mf_diag_allLabHpo_idx01 ON JAX_mf_diag_allLabHpo (SUBJECT_ID, HADM_ID)')


def diagnosisTextLab(phenotype):
    """
    Merge temporary tables to create one in which each encounter is assigned with 0 or 1 for diagnosis code, phenotype from text data and phenotype from lab tests.
    """
    cursor.execute('DROP TEMPORARY TABLE IF EXISTS JAX_mf_diag_txtHpo_labHpo')
    result = cursor.execute('''
        CREATE TEMPORARY TABLE JAX_mf_diag_txtHpo_labHpo 
        WITH L AS (SELECT JAX_mf_diag_textHpo.*, '{}' AS PHEN_LAB FROM JAX_mf_diag_textHpo)
        SELECT L.*, IF(R.dummy IS NULL, '0', '1') AS PHEN_LAB_VALUE
        FROM L 
        LEFT JOIN 
            JAX_labHpoProfile AS R 
        ON L.SUBJECT_ID = R.SUBJECT_ID AND L.HADM_ID = R.HADM_ID AND L.PHEN_LAB = R.MAP_TO
    '''.format(phenotype))


def diagnosisAllTextAllLab():
    """
    Merge temporary tables to create one in which each encounter is assigned with 0 or 1 for diagnosis code, all phenotypes of interest from text data, and all phenotypes from lab tests
    """
    cursor.execute('DROP TEMPORARY TABLE IF EXISTS JAX_mf_diag_allTxtHpo_allLabHpo')
    cursor.execute('''
        CREATE TEMPORARY TABLE JAX_mf_diag_allTxtHpo_allLabHpo 
        SELECT L.SUBJECT_ID, L.HADM_ID, L.DIAGNOSIS, L.PHEN_TXT, L.PHEN_TXT_VALUE, R.PHEN_LAB, R.PHEN_LAB_VALUE 
        FROM JAX_mf_diag_allTextHpo AS L 
        JOIN JAX_mf_diag_allLabHpo AS R
        ON L.SUBJECT_ID = R.SUBJECT_ID AND L.HADM_ID = R.HADM_ID
    ''')


def initSummaryStatisticTables():
    """
    Init summary statistics tables.
    """
    # define empty columns to store summary statistics
    summary_statistics1_radiology = pd.DataFrame(data={'DIAGNOSIS_CODE': [],
                                                       'PHENOTYPE': [],
                                                       'DIAGNOSIS_VALUE': [],
                                                       'PHENOTYPE_VALUE': [],
                                                       'N': []},
                                                 columns=['DIAGNOSIS_CODE', 'PHENOTYPE', 'DIAGNOSIS_VALUE',
                                                          'PHENOTYPE_VALUE', 'N'])

    summary_statistics1_lab = pd.DataFrame(data={'DIAGNOSIS_CODE': [],
                                                 'PHENOTYPE': [],
                                                 'DIAGNOSIS_VALUE': [],
                                                 'PHENOTYPE_VALUE': [],
                                                 'N': []},
                                           columns=['DIAGNOSIS_CODE', 'PHENOTYPE', 'DIAGNOSIS_VALUE', 'PHENOTYPE_VALUE',
                                                    'N'])

    summary_statistics2 = pd.DataFrame(data={'DIAGNOSIS_CODE': [],
                                             'PHEN_TXT': [],
                                             'PHEN_LAB': [],
                                             'DIAGNOSIS_VALUE': [],
                                             'PHEN_TXT_VALUE': [],
                                             'PHEN_LAB_VALUE': [],
                                             'N': []},
                                       columns=['DIAGNOSIS_CODE', 'PHEN_TXT', 'PHEN_LAB', 'DIAGNOSIS_VALUE',
                                                'PHEN_TXT_VALUE', 'PHEN_LAB_VALUE', 'N'])

    return summary_statistics1_radiology, summary_statistics1_lab, summary_statistics2


def initTables(debug=False):
    """
    This combines LabHpo and Inferred_LabHpo, and combines TextHpo and Inferred_TextHpo.
    Only need to run once. For efficiency consideration, the tables can also be created as perminent.
    It is time-consuming, so call it with caution.
    """
    # init textHpoProfile and index it
    # I create perminant tables to save time; other users should enable them
    # textHpoProfile(include_inferred=True, threshold=1)
    # indexTextHpoProfile()
    # init labHpoProfile and index it
    # labHpoProfile(threshold=1, include_inferred=True, force_update=True)
    # indexLabHpoProfile()

    # define encounters to analyze
    encounterOfInterest(debug)
    indexEncounterOfInterest()
    # init diagnosisProfile
    diagnosisProfile()


def iterate(primary_diagnosis_only, diagnosis_threshold_min, textHpo_threshold_min, labHpo_threshold_min, logger):
    logger.info('starting iterating...................................')
    N = pd.read_sql_query("SELECT count(*) FROM JAX_encounterOfInterest", mydb)
    # init empty tables to hold summary statistics
    summary_statistics1_radiology, summary_statistics1_lab, summary_statistics2 = initSummaryStatisticTables()

    # define a set of diseases that we want to analyze
    rankICD()

    diseaseOfInterest = pd.read_sql_query(
        "SELECT * FROM JAX_diagFrequencyRank WHERE N > {}".format(diagnosis_threshold_min), mydb).ICD9_CODE.values
    diseaseOfInterest = ['428']
    # define encounters to analyze
    logger.info('diseases of interest established: {}'.format(len(diseaseOfInterest)))
    for diagnosis in diseaseOfInterest:
        logger.info("start analyzing disease {}".format(diagnosis))

        # assign each encounter whether a diagnosis code is observed
        # create a table j1 (joint 1)
        createDiagnosisTable(diagnosis, primary_diagnosis_only)
        # for every diagnosis, find phenotypes of interest to look at from radiology reports
        # for every diagnosis, find phenotypes of interest to look at from laboratory tests
        rankHpoFromText(diagnosis)
        rankHpoFromLab(diagnosis)

        textHpoOfInterest = pd.read_sql_query(
            "SELECT * FROM JAX_textHpoFrequencyRank WHERE N > {}".format(textHpo_threshold_min), mydb).MAP_TO.values
        labHpoOfInterest = pd.read_sql_query(
            "SELECT * FROM JAX_labHpoFrequencyRank WHERE N > {}".format(labHpo_threshold_min), mydb).MAP_TO.values
        logger.info("TextHpo of interest established, size: {}".format(len(textHpoOfInterest)))
        logger.info("LabHpo of interest established, size: {}".format(len(labHpoOfInterest)))
        for textHpo in textHpoOfInterest:
            logger.info("iteration: TextHpo--{}".format(textHpo))
            # assign each encounter whether a phenotype is observed from radiology reports
            diagnosisTextHpo(textHpo)
            result1_text = pd.read_sql_query('''
                SELECT 
                    '{}' AS DIAGNOSIS_CODE, '{}' AS PHENOTYPE, DIAGNOSIS AS DIAGNOSIS_VALUE, PHEN_TXT_VALUE AS PHENOTYPE_VALUE, COUNT(*) AS N 
                FROM JAX_mf_diag_textHpo 
                GROUP BY 
                    DIAGNOSIS, PHEN_TXT_VALUE
            '''.format(diagnosis, textHpo), mydb)
            summary_statistics1_radiology = summary_statistics1_radiology.append(result1_text)
            # summary statistics for p1
            # calculate I(p1;D)
            for labHpo in labHpoOfInterest:
                logger.info(".........LabHpo--{}".format(labHpo))
                diagnosisLabHpo(labHpo)
                result1_lab = pd.read_sql_query('''
                    SELECT 
                        '{}' AS DIAGNOSIS_CODE, '{}' AS PHENOTYPE, DIAGNOSIS AS DIAGNOSIS_VALUE, PHEN_LAB_VALUE AS PHENOTYPE_VALUE, COUNT(*) AS N 
                    FROM 
                        JAX_mf_diag_labHpo 
                    GROUP BY DIAGNOSIS, PHEN_LAB_VALUE
                '''.format(diagnosis, labHpo), mydb)
                summary_statistics1_lab = summary_statistics1_lab.append(result1_lab)

                # assign each encounter whether a phenotype is observed from lab tests
                diagnosisTextLab(labHpo)
                result2 = pd.read_sql_query('''
                    SELECT 
                        '{}' AS DIAGNOSIS_CODE, 
                        '{}' AS PHEN_TXT, 
                        '{}' AS PHEN_LAB,  
                        DIAGNOSIS AS DIAGNOSIS_VALUE, 
                        PHEN_TXT_VALUE, 
                        PHEN_LAB_VALUE, 
                        COUNT(*) AS N
                    FROM JAX_mf_diag_txtHpo_labHpo 
                    GROUP BY DIAGNOSIS, PHEN_TXT_VALUE, PHEN_LAB_VALUE
                '''.format(diagnosis, textHpo, labHpo), mydb)
                summary_statistics2 = summary_statistics2.append(result2)
    logger.info('end iterating.....................................')
    return N, summary_statistics1_radiology, summary_statistics1_lab, summary_statistics2


def iterate_batch(primary_diagnosis_only, diagnosis_threshold_min, textHpo_threshold_min, textHpo_threshold_max,
                  labHpo_threshold_min, labHpo_threshold_max, logger):
    logger.info('starting iterating...................................')
    N = pd.read_sql_query("SELECT count(*) FROM JAX_encounterOfInterest", mydb)
    # init empty tables to hold summary statistics
    summary_statistics1_radiology, summary_statistics1_lab, summary_statistics2 = initSummaryStatisticTables()

    # define a set of diseases that we want to analyze
    rankICD()

    diseaseOfInterest = pd.read_sql_query(
        "SELECT * FROM JAX_diagFrequencyRank WHERE N > {}".format(diagnosis_threshold_min), mydb).ICD9_CODE.values
    diseaseOfInterest = ['428']
    logger.info('diseases of interest established: {}'.format(len(diseaseOfInterest)))

    for diagnosis in diseaseOfInterest:
        logger.info("start analyzing disease {}".format(diagnosis))

        logger.info(".......assigning values of diagnosis")
        # assign each encounter whether a diagnosis code is observed
        # create a table j1 (joint 1)
        createDiagnosisTable(diagnosis, primary_diagnosis_only)
        # for every diagnosis, find phenotypes of interest to look at from radiology reports
        # for every diagnosis, find phenotypes of interest to look at from laboratory tests
        rankHpoFromText(diagnosis)
        rankHpoFromLab(diagnosis)
        logger.info("..............diagnosis values found")

        logger.info(".......assigning values of TextHpo")
        diagnosisAllTextHpo(textHpo_threshold_min, textHpo_threshold_max)
        result1_text = pd.read_sql_query("""
            SELECT '{}' AS DIAGNOSIS_CODE, 
                PHEN_TXT AS PHENOTYPE, 
                DIAGNOSIS AS DIAGNOSIS_VALUE, 
                PHEN_TXT_VALUE AS PHENOTYPE_VALUE, 
                COUNT(*) AS N 
            FROM JAX_mf_diag_allTextHpo 
            GROUP BY DIAGNOSIS, PHEN_TXT, PHEN_TXT_VALUE
        """.format(diagnosis), mydb)
        logger.info("..............TextHpo values found")
        summary_statistics1_radiology = summary_statistics1_radiology.append(result1_text)

        logger.info(".......assigning values of LabHpo")
        diagnosisAllLabHpo(labHpo_threshold_min, labHpo_threshold_max)
        result1_lab = pd.read_sql_query("""
            SELECT 
                '{}' AS DIAGNOSIS_CODE, 
                PHEN_LAB AS PHENOTYPE, 
                DIAGNOSIS AS DIAGNOSIS_VALUE, 
                PHEN_LAB_VALUE AS PHENOTYPE_VALUE, 
                COUNT(*) AS N 
            FROM JAX_mf_diag_allLabHpo 
            GROUP BY DIAGNOSIS, PHEN_LAB, PHEN_LAB_VALUE
        """.format(diagnosis), mydb)
        logger.info("..............LabHpo values found")
        summary_statistics1_lab = summary_statistics1_lab.append(result1_lab)

        logger.info(".......building diagnosis-TextHpo-LabHpo joint distribution")
        diagnosisAllTextAllLab()
        result2 = pd.read_sql_query("""
            SELECT 
                '{}' AS DIAGNOSIS_CODE, 
                PHEN_TXT, 
                PHEN_LAB, 
                DIAGNOSIS AS DIAGNOSIS_VALUE,
                PHEN_TXT_VALUE, 
                PHEN_LAB_VALUE, 
                COUNT(*) AS N 
            FROM JAX_mf_diag_allTxtHpo_allLabHpo 
            GROUP BY DIAGNOSIS, PHEN_LAB, PHEN_LAB_VALUE, PHEN_TXT, PHEN_TXT_VALUE
        """.format(diagnosis), mydb)
        logger.info("..............diagnosis-TextHpo-LabHpo joint distribution built")
        summary_statistics2 = summary_statistics2.append(result2)

    logger.info('end iterating.....................................')
    return N, summary_statistics1_radiology, summary_statistics1_lab, summary_statistics2


def indexDiagnosisTable():
    cursor.execute("ALTER TABLE JAX_mf_diag ADD COLUMN ROW_ID INT AUTO_INCREMENT PRIMARY KEY;")


def batch_query(start_index, end_index, textHpo_occurrance_min, labHpo_occurrance_min, textHpo_threshold_min,
                textHpo_threshold_max, labHpo_threshold_min, labHpo_threshold_max):
    """
    Queries databases in small batches, return diagnosis values, phenotypes from text data and phenotypes from lab data.
    @param start_index: minimum row_id
    @param end_index: maximum row_id
    @param textHpo_occurrance_min: minimum occurrances of a phenotype from text data for it to be called in one encounter
    @param labHpo_occurrance_max: maximum occurrances of a phenotype from lab tests for it to be called in one encounter
    @param textHpo_threshold_min: minimum number of encounters of a phenotypes from text data for it to be analyzed
    @param textHpo_threshold_max: maximum number of encounters of a phenotypes from text data for it to be analyzed
    @param labHpo_threshold_min: minimum number of encounters of a phenotype from lab tests for it to be analyzed
    @param labHpo_threshold_max: maximum number of encounters of a phenotype from lab tests for it to be analyzed
    """
    diagnosisVector = pd.read_sql_query('''
        SELECT * FROM JAX_mf_diag WHERE ROW_ID BETWEEN {} AND {}
    '''.format(start_index, end_index), mydb)

    textHpoFlat = pd.read_sql_query('''
        WITH encounters AS (
            SELECT SUBJECT_ID, HADM_ID
            FROM JAX_mf_diag 
            WHERE ROW_ID BETWEEN {} AND {}
        ), 
        textHpoOfInterest AS (
            SELECT MAP_TO 
            FROM JAX_textHpoFrequencyRank 
            WHERE N BETWEEN {} AND {}
        ), 
        joint as (
            SELECT *
            FROM encounters 
            JOIN textHpoOfInterest),
        JAX_textHpoProfile_filtered AS (
            SELECT * 
            FROM JAX_textHpoProfile 
            WHERE OCCURRANCE >= {}
        )

        SELECT L.SUBJECT_ID, L.HADM_ID, L.MAP_TO, IF(R.dummy IS NULL, 0, 1) AS VALUE
        FROM joint as L
        LEFT JOIN 
        JAX_textHpoProfile_filtered AS R
        ON L.SUBJECT_ID = R.SUBJECT_ID AND L.HADM_ID = R.HADM_ID AND L.MAP_TO = R.MAP_TO  
    '''.format(start_index, end_index, textHpo_threshold_min, textHpo_threshold_max, textHpo_occurrance_min), mydb)

    labHpoFlat = pd.read_sql_query('''
        WITH encounters AS (
            SELECT SUBJECT_ID, HADM_ID
            FROM JAX_mf_diag 
            WHERE ROW_ID BETWEEN {} AND {}
        ), 
        labHpoOfInterest AS (
            SELECT MAP_TO 
            FROM JAX_labHpoFrequencyRank 
            WHERE N BETWEEN {} AND {}
        ), 
        joint as (
            SELECT *
            FROM encounters 
            JOIN labHpoOfInterest),
        JAX_labHpoProfile_filtered AS (
            SELECT * 
            FROM JAX_labHpoProfile 
            WHERE OCCURRANCE >= {}
        )

        SELECT L.SUBJECT_ID, L.HADM_ID, L.MAP_TO, IF(R.dummy IS NULL, 0, 1) AS VALUE
        FROM joint as L
        LEFT JOIN 
        JAX_labHpoProfile_filtered AS R
        ON L.SUBJECT_ID = R.SUBJECT_ID AND L.HADM_ID = R.HADM_ID AND L.MAP_TO = R.MAP_TO
    '''.format(start_index, end_index, labHpo_threshold_min, labHpo_threshold_max, labHpo_occurrance_min), mydb)

    return diagnosisVector, textHpoFlat, labHpoFlat


def summarize_diagnosis_textHpo_labHpo(primary_diagnosis_only, textHpo_occurrance_min, labHpo_occurrance_min,
                                       diagnosis_threshold_min, textHpo_threshold_min, textHpo_threshold_max,
                                       labHpo_threshold_min, labHpo_threshold_max, logger):
    """
    Iterate database to get summary statistics.

    @param primary_diagnosis_only: only primary diagnosis is analyzed
    @param textHpo_occurrance_min: minimum occurrances of a phenotype from text data for it to be called in one encounter
    @param labHpo_occurrance_max: maximum occurrances of a phenotype from lab tests for it to be called in one encounter
    @param textHpo_threshold_min: minimum number of encounters of a phenotypes from text data for it to be analyzed
    @param textHpo_threshold_max: maximum number of encounters of a phenotypes from text data for it to be analyzed
    @param labHpo_threshold_min: minimum number of encounters of a phenotype from lab tests for it to be analyzed
    @param labHpo_threshold_max: maximum number of encounters of a phenotype from lab tests for it to be analyzed
    @param logger: logger for logging

    """
    logger.info('starting iterate_in_batch()')
    batch_size = 100

    # define a set of diseases that we want to analyze
    rankICD()

    diseaseOfInterest = pd.read_sql_query(
        "SELECT * FROM JAX_diagFrequencyRank WHERE N > {}".format(diagnosis_threshold_min), mydb).ICD9_CODE.values
    # disable the following line to analyze all diseases of interest
    diseaseOfInterest = ['428', '584', '038']
    logger.info('diagnosis of interest: {}'.format(len(diseaseOfInterest)))

    summaries_diag_textHpo_labHpo = {}
    summaries_diag_textHpo_textHpo = {}
    summaries_diag_labHpo_labHpo = {}

    pbar = tqdm(total=len(diseaseOfInterest))
    for diagnosis in diseaseOfInterest:
        logger.info("start analyzing disease {}".format(diagnosis))

        logger.info(".......assigning values of diagnosis")
        # assign each encounter whether a diagnosis code is observed
        # create a table j1 (joint 1)
        createDiagnosisTable(diagnosis, primary_diagnosis_only)
        indexDiagnosisTable()
        # for every diagnosis, find phenotypes of interest to look at from radiology reports
        # for every diagnosis, find phenotypes of interest to look at from laboratory tests
        rankHpoFromText(diagnosis, textHpo_occurrance_min)
        rankHpoFromLab(diagnosis, labHpo_occurrance_min)
        logger.info("..............diagnosis values found")

        textHpoOfInterest = pd.read_sql_query(
            "SELECT * FROM JAX_textHpoFrequencyRank WHERE N BETWEEN {} AND {}".format(textHpo_threshold_min,
                                                                                      textHpo_threshold_max),
            mydb).MAP_TO.values
        labHpoOfInterest = pd.read_sql_query(
            "SELECT * FROM JAX_labHpoFrequencyRank WHERE N BETWEEN {} AND {}".format(labHpo_threshold_min,
                                                                                     labHpo_threshold_max),
            mydb).MAP_TO.values
        logger.info("TextHpo of interest established, size: {}".format(len(textHpoOfInterest)))
        logger.info("LabHpo of interest established, size: {}".format(len(labHpoOfInterest)))

        ## find the start and end ROW_ID for patient*encounter
        ADM_ID_START, ADM_ID_END = \
        pd.read_sql_query('SELECT MIN(ROW_ID) AS min, MAX(ROW_ID) AS max FROM JAX_mf_diag', mydb).iloc[0]
        batch_N = ADM_ID_END - ADM_ID_START + 1
        TOTAL_BATCH = math.ceil(batch_N / batch_size)  # total number of batches

        summaries_diag_textHpo_labHpo[diagnosis] = mf.SummaryXYz(textHpoOfInterest, labHpoOfInterest, diagnosis)
        summaries_diag_textHpo_textHpo[diagnosis] = mf.SummaryXYz(textHpoOfInterest, textHpoOfInterest, diagnosis)
        summaries_diag_labHpo_labHpo[diagnosis] = mf.SummaryXYz(labHpoOfInterest, labHpoOfInterest, diagnosis)

        logger.info('starting batch queries for {}'.format(diagnosis))
        for i in np.arange(TOTAL_BATCH):
            start_index = i * batch_size + ADM_ID_START
            if i < TOTAL_BATCH - 1:
                end_index = start_index + batch_size - 1
            else:
                end_index = batch_N

            diagnosisFlat, textHpoFlat, labHpoFlat = batch_query(start_index, end_index, textHpo_occurrance_min,
                                                                 labHpo_occurrance_min, textHpo_threshold_min,
                                                                 textHpo_threshold_max, labHpo_threshold_min,
                                                                 labHpo_threshold_max)

            batch_size_actual = len(diagnosisFlat)
            textHpoOfInterest_size = len(textHpoOfInterest)
            labHpoOfInterest_size = len(labHpoOfInterest)
            # print('len(textHpoFlat)= {}, batch_size_actual={}, textHpoOfInterest_size={}'.format(len(textHpoFlat), batch_size_actual, textHpoOfInterest_size))
            assert (len(textHpoFlat) == batch_size_actual * textHpoOfInterest_size)
            assert (len(labHpoFlat) == batch_size_actual * labHpoOfInterest_size)

            if batch_size_actual > 0:
                diagnosisVector = diagnosisFlat.DIAGNOSIS.values.astype(int)
                # reformat the flat vector into N x M matrix, N is batch size, i.e. number of encounters, M is the length of HPO terms
                textHpoMatrix = textHpoFlat.VALUE.values.astype(int).reshape(
                    [batch_size_actual, textHpoOfInterest_size], order='F')
                labHpoMatrix = labHpoFlat.VALUE.values.astype(int).reshape([batch_size_actual, labHpoOfInterest_size],
                                                                           order='F')
                # check the matrix formatting is correct
                # disable the following 4 lines to speed things up
                textHpoLabelsMatrix = textHpoFlat.MAP_TO.values.reshape([batch_size_actual, textHpoOfInterest_size],
                                                                        order='F')
                labHpoLabelsMatrix = labHpoFlat.MAP_TO.values.reshape([batch_size_actual, labHpoOfInterest_size],
                                                                      order='F')
                assert (textHpoLabelsMatrix[0, :] == textHpoOfInterest).all()
                assert (labHpoLabelsMatrix[0, :] == labHpoOfInterest).all()
                if i % 100 == 0:
                    logger.info(
                        'new batch: start_index={}, end_index={}, batch_size= {}, textHpo_size = {}, labHpo_size = {}'.format(
                            start_index, end_index, batch_size_actual, textHpoMatrix.shape[1], labHpoMatrix.shape[1]))
                summaries_diag_textHpo_labHpo[diagnosis].add_batch(textHpoMatrix, labHpoMatrix, diagnosisVector)
                summaries_diag_textHpo_textHpo[diagnosis].add_batch(textHpoMatrix, textHpoMatrix, diagnosisVector)
                summaries_diag_labHpo_labHpo[diagnosis].add_batch(labHpoMatrix, labHpoMatrix, diagnosisVector)

        pbar.update(1)

    pbar.close()

    return summaries_diag_textHpo_labHpo, summaries_diag_textHpo_textHpo, summaries_diag_labHpo_labHpo
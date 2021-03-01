import numpy as np
import pandas as pd
import math
import mutual_information.mf as mf
import mutual_information.synergy_tree as synergy_tree
from mimic_mf_analysis.preparation import encounterOfInterest, indexEncounterOfInterest, diagnosisProfile, rankICD, rankHpoFromLab, rankHpoFromText
from tqdm import tqdm

from mimic_mf_analysis import mydb

cursor = mydb.cursor(buffered=True)


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


def indexDiagnosisTable():
    cursor.execute("ALTER TABLE JAX_mf_diag ADD COLUMN ROW_ID INT AUTO_INCREMENT PRIMARY KEY;")


def batch_query(start_index,
                end_index,
                textHpo_occurrance_min,
                labHpo_occurrance_min,
                textHpo_threshold_min,
                textHpo_threshold_max,
                labHpo_threshold_min,
                labHpo_threshold_max):
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


def summarize_diagnosis_textHpo_labHpo(primary_diagnosis_only,
                                       textHpo_occurrance_min,
                                       labHpo_occurrance_min,
                                       diagnosis_threshold_min,
                                       textHpo_threshold_min,
                                       textHpo_threshold_max,
                                       labHpo_threshold_min,
                                       labHpo_threshold_max,
                                       disease_of_interest,
                                       logger):
    """
    Iterate database to get summary statistics. For each disease of interest, automatically determine a list of phenotypes derived from labs (labHpo) and a list of phenotypes from text mining (textHpo). For each pair of phenotypes, count the number of encounters according to whether the phenotypes and diagnosis are observated.
    @param primary_diagnosis_only: only primary diagnosis is analyzed
    @param textHpo_occurrance_min: minimum occurrances of a phenotype from text data for it to be called in one encounter
    @param labHpo_occurrance_max: maximum occurrances of a phenotype from lab tests for it to be called in one encounter
    @param textHpo_threshold_min: minimum number of encounters of a phenotypes from text data for it to be analyzed
    @param textHpo_threshold_max: maximum number of encounters of a phenotypes from text data for it to be analyzed
    @param labHpo_threshold_min: minimum number of encounters of a phenotype from lab tests for it to be analyzed
    @param labHpo_threshold_max: maximum number of encounters of a phenotype from lab tests for it to be analyzed
    @param disease_of_interest: either set to "calculated", or a list of ICD-9 codes (get all possible codes from temp table JAX_diagFrequencyRank)
    @param logger: logger for logging

    :return: three dictionaries of summary statistics, of which the keys are diagnosis codes and the values are instances of the SummaryXYz class.
    First dictionary, X (a list of phenotype variables) are from textHpo and Y are from labHpo;
    Secondary dictionary, all terms in X, Y are from textHpo;
    Third dictionary, all terms in X, Y are all from labHpo. Note that terms in X and Y are calculated separately for each diagnosis and may be different.
    """
    logger.info('starting iterate_in_batch()')
    batch_size = 100

    # define a set of diseases that we want to analyze
    rankICD()

    if disease_of_interest == 'calculated':
        diseaseOfInterest = pd.read_sql_query(
            "SELECT * FROM JAX_diagFrequencyRank WHERE N > {}".format(diagnosis_threshold_min), mydb).ICD9_CODE.values
    elif isinstance(disease_of_interest, list) and len(disease_of_interest) > 0:
        # disable the following line to analyze all diseases of interest
        # diseaseOfInterest = ['428', '584', '038', '493']
        diseaseOfInterest = disease_of_interest
    else:
        raise RuntimeError
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


def add_diag_columns(diagnosis, primary_diagnosis_only):
    createDiagnosisTable(diagnosis, primary_diagnosis_only)
    # copy into a new table Jax_multivariant_synergy_table(SUBJECT_ID, HADM_ID, DIAGNOSIS)
    cursor.execute("""
        CREATE TEMPORARY TABLE IF NOT EXISTS Jax_multivariant_synergy_table AS (
            SELECT * 
            FROM JAX_mf_diag
        )""")
    cursor.execute('CREATE INDEX Jax_multivariant_synergy_table_idx01 ON JAX_mf_diag (SUBJECT_ID, HADM_ID)')


def add_phenotype_columns(labHpos, textHpos, labHpo_threshold_min, textHpo_threshold_min):
    # save the variable transformation for later use
    var_dict = {}
    i = 0
    for labHpo in labHpos:
        i = i + 1
        colName = 'V' + str(i)
        var_dict[colName] = ('LabHpo', labHpo)
        cursor.execute("""
            ALTER TABLE Jax_multivariant_synergy_table ADD COLUMN {} INT DEFAULT 0""".format(colName))
        cursor.execute("""
            UPDATE Jax_multivariant_synergy_table 
            LEFT JOIN JAX_labHpoProfile 
            ON Jax_multivariant_synergy_table.SUBJECT_ID = JAX_labHpoProfile.SUBJECT_ID AND 
            Jax_multivariant_synergy_table.HADM_ID = JAX_labHpoProfile.HADM_ID
            SET {} = IF(JAX_labHpoProfile.OCCURRANCE > {}, 1, 0)
            WHERE JAX_labHpoProfile.MAP_TO = '{}'
        """.format(colName, labHpo_threshold_min, labHpo))

    for textHpo in textHpos:
        i = i + 1
        colName = 'V' + str(i)
        var_dict[colName] = ('TextHpo', textHpo)
        cursor.execute("""
            ALTER TABLE Jax_multivariant_synergy_table ADD COLUMN {} INT DEFAULT 0""".format(colName))
        cursor.execute("""
            UPDATE Jax_multivariant_synergy_table 
            LEFT JOIN JAX_textHpoProfile 
            ON Jax_multivariant_synergy_table.SUBJECT_ID = JAX_textHpoProfile.SUBJECT_ID AND 
            Jax_multivariant_synergy_table.HADM_ID = JAX_textHpoProfile.HADM_ID
            SET {} = IF(JAX_textHpoProfile.OCCURRANCE > {}, 1, 0)
            WHERE JAX_textHpoProfile.MAP_TO = '{}'
        """.format(colName, textHpo_threshold_min, textHpo))
    return var_dict


def precompute_mf(variables):
    """
    Compute the mutual information between the joint distribution of all the variables and the medical outcome
    """
    summary_counts = pd.read_sql_query("""
        WITH summary AS (
        SELECT {}, DIAGNOSIS, COUNT(*) AS N
        FROM Jax_multivariant_synergy_table
        GROUP BY {}, DIAGNOSIS)
        SELECT *, SUM(N) OVER (PARTITION BY {}) AS V, SUM(N) OVER (PARTITION BY DIAGNOSIS) AS D
        FROM summary
    """.format(','.join(variables), ','.join(variables), ','.join(variables)), mydb)
    total = np.sum(summary_counts.N)
    p = summary_counts.N / total
    p_V = summary_counts.V / total
    p_D = summary_counts.D / total
    mf = np.sum(p * np.log2(p / (p_V * p_D)))
    return mf, summary_counts


def precompute_mf_dict(var_ids):
    var_subsets = synergy_tree.subsets(var_ids, include_self=True)
    mf_dict = {}
    summary_dict = {}
    pbar = tqdm(total=len(var_subsets))
    for var_subset in var_subsets:
        mf, summary_count = precompute_mf(var_subset)
        mf_dict[var_subset] = mf
        summary_dict[var_subset] = summary_count
        pbar.update(1)
    pbar.close()

    return mf_dict, summary_dict
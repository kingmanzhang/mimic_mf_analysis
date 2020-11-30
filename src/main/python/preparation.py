from . import mydb

cursor = mydb.cursor(buffered=True)


def encounterOfInterest(debug=False, N=100):
    """
    Define encounters of interest. The method is not finalized yet. Currently, it will use all encounters in our database.
    @param debug: set to True to select a small subset for testing
    @param N: limit the number of encounters when debug is set to True. If debug is set to False, N is ignored.
    """
    cursor.execute('DROP TEMPORARY TABLE IF EXISTS JAX_encounterOfInterest')
    if debug:
        limit = 'LIMIT {}'.format(N)
    else:
        limit = ''
    # This is admissions that we want to analyze, 'LIMIT 100' in debug mode
    cursor.execute('''
                CREATE TEMPORARY TABLE IF NOT EXISTS JAX_encounterOfInterest(
                    ROW_ID MEDIUMINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY)

                SELECT 
                    DISTINCT SUBJECT_ID, HADM_ID 
                FROM admissions
                {}
                '''.format(limit))


def indexEncounterOfInterest():
    """
    Create index on encounters table.
    """
    cursor.execute('CREATE INDEX JAX_encounterOfInterest_idx01 ON JAX_encounterOfInterest (SUBJECT_ID, HADM_ID)')


def diagnosisProfile():
    """
    For encounters of interest, find all of their diagnosis codes
    """
    cursor.execute('DROP TEMPORARY TABLE IF EXISTS JAX_diagnosisProfile')
    cursor.execute('''
                CREATE TEMPORARY TABLE IF NOT EXISTS JAX_diagnosisProfile
                SELECT 
                    DIAGNOSES_ICD.SUBJECT_ID, DIAGNOSES_ICD.HADM_ID, DIAGNOSES_ICD.ICD9_CODE, DIAGNOSES_ICD.SEQ_NUM
                FROM
                    DIAGNOSES_ICD
                RIGHT JOIN
                    JAX_encounterOfInterest
                ON 
                    DIAGNOSES_ICD.SUBJECT_ID = JAX_encounterOfInterest.SUBJECT_ID 
                    AND 
                    DIAGNOSES_ICD.HADM_ID = JAX_encounterOfInterest.HADM_ID
                ''')


def textHpoProfile(include_inferred=True):
    """
    Set up a table for patient phenotypes from text mining. By default, merge directly mapped HPO terms and inferred terms.
    It is currently defined as a temporary table. But in reality, it is created as a perminent table as it takes a long time to init, and it is going to be used multiple times.
    """
    if include_inferred:
        cursor.execute('''
                    CREATE TEMPORARY TABLE IF NOT EXISTS JAX_textHpoProfile
                    WITH abnorm AS (
                        SELECT
                            NOTEEVENTS.SUBJECT_ID, NOTEEVENTS.HADM_ID, NoteHpoClinPhen.MAP_TO
                        FROM 
                            NOTEEVENTS 
                        JOIN NoteHpoClinPhen on NOTEEVENTS.ROW_ID = NoteHpoClinPhen.NOTES_ROW_ID

                        UNION ALL

                        SELECT
                            NOTEEVENTS.SUBJECT_ID, NOTEEVENTS.HADM_ID, Inferred_NoteHpo.INFERRED_TO AS MAP_TO
                        FROM 
                            NOTEEVENTS 
                        JOIN Inferred_NoteHpo on NOTEEVENTS.ROW_ID = Inferred_NoteHpo.NOTEEVENT_ROW_ID
                        )
                    SELECT SUBJECT_ID, HADM_ID, MAP_TO, COUNT(*) AS OCCURRANCE, 1 AS dummy
                    FROM abnorm 
                    GROUP BY SUBJECT_ID, HADM_ID, MAP_TO
                ''')

    else:
        cursor.execute('''
                    CREATE TEMPORARY TABLE IF NOT EXISTS JAX_p_text
                    WITH abnorm AS (
                        SELECT
                            NOTEEVENTS.SUBJECT_ID, NOTEEVENTS.HADM_ID, NoteHpoClinPhen.MAP_TO
                        FROM 
                            NOTEEVENTS 
                        JOIN NoteHpoClinPhen on NOTEEVENTS.ROW_ID = NoteHpoClinPhen.NOTES_ROW_ID)
                    SELECT SUBJECT_ID, HADM_ID, MAP_TO
                    FROM abnorm 
                    GROUP BY SUBJECT_ID, HADM_ID, MAP_TO, COUNT(*) AS OCCURRANCE, 1 AS dummy
                ''')


def indexTextHpoProfile():
    """
    Create indeces to speed up query
    """
    # _idx01 is unnecessary if _idx3 exists
    # cursor.execute('CREATE INDEX JAX_textHpoProfile_idx01 ON JAX_textHpoProfile (SUBJECT_ID, HADM_ID)')
    cursor.execute('CREATE INDEX JAX_textHpoProfile_idx02 ON JAX_textHpoProfile (MAP_TO);')
    cursor.execute('CREATE INDEX JAX_textHpoProfile_idx03 ON JAX_textHpoProfile (SUBJECT_ID, HADM_ID, MAP_TO)')
    cursor.execute('CREATE INDEX JAX_textHpoProfile_idx04 ON JAX_textHpoProfile (OCCURRANCE)')


def labHpoProfile(include_inferred=True):
    """
    Set up a table for lab tests-derived phenotypes. By default, also include phenotypes that are inferred from direct mapping.
    Similar to textHpoProfile, this could be created as a perminent table.
    """
    cursor.execute('''DROP TEMPORARY TABLE IF EXISTS JAX_labHpoProfile''')
    if include_inferred:
        cursor.execute('''
                    CREATE TEMPORARY TABLE IF NOT EXISTS JAX_labHpoProfile
                    WITH abnorm AS (
                        SELECT
                            LABEVENTS.SUBJECT_ID, LABEVENTS.HADM_ID, LabHpo.MAP_TO
                        FROM 
                            LABEVENTS 
                        JOIN LabHpo on LABEVENTS.ROW_ID = LabHpo.ROW_ID
                        WHERE LabHpo.NEGATED = 'F'

                        UNION ALL

                        SELECT 
                            LABEVENTS.SUBJECT_ID, LABEVENTS.HADM_ID, INFERRED_LABHPO.INFERRED_TO AS MAP_TO 
                        FROM 
                            INFERRED_LABHPO 
                        JOIN 
                            LABEVENTS ON INFERRED_LABHPO.LABEVENT_ROW_ID = LABEVENTS.ROW_ID
                        )
                    SELECT SUBJECT_ID, HADM_ID, MAP_TO, COUNT(*) AS OCCURRANCE, 1 AS dummy
                    FROM abnorm 
                    GROUP BY SUBJECT_ID, HADM_ID, MAP_TO
                ''')
    else:
        cursor.execute('''
                    CREATE TEMPORARY TABLE IF NOT EXISTS JAX_labHpoProfile
                    WITH abnorm AS (
                        SELECT
                            LABEVENTS.SUBJECT_ID, LABEVENTS.HADM_ID, LabHpo.MAP_TO
                        FROM 
                            LABEVENTS 
                        JOIN LabHpo on LABEVENTS.ROW_ID = LabHpo.ROW_ID
                        WHERE LabHpo.NEGATED = 'F')
                    SELECT SUBJECT_ID, HADM_ID, MAP_TO, COUNT(*) AS OCCURRANCE, 1 AS dummy
                    FROM abnorm 
                    GROUP BY SUBJECT_ID, HADM_ID, MAP_TO
                ''')


def indexLabHpoProfile():
    # _idx01 is not necessary if _idx3 exists
    # cursor.execute('CREATE INDEX JAX_labHpoProfile_idx01 ON JAX_labHpoProfile (SUBJECT_ID, HADM_ID)')
    cursor.execute('CREATE INDEX JAX_labHpoProfile_idx02 ON JAX_labHpoProfile (MAP_TO);')
    cursor.execute('CREATE INDEX JAX_labHpoProfile_idx03 ON JAX_labHpoProfile (SUBJECT_ID, HADM_ID, MAP_TO)')
    cursor.execute('CREATE INDEX JAX_labHpoProfile_idx04 ON JAX_labHpoProfile (OCCURRANCE)')


def rankICD():
    """
    Rank frequently seen ICD-9 codes (first three or four digits) among encounters of interest.
    """
    cursor.execute('DROP TEMPORARY TABLE IF EXISTS JAX_diagFrequencyRank')
    cursor.execute("""
        CREATE TEMPORARY TABLE IF NOT EXISTS JAX_diagFrequencyRank
        WITH JAX_temp_diag AS (
            SELECT DISTINCT SUBJECT_ID, HADM_ID, 
                CASE 
                    WHEN(ICD9_CODE LIKE 'V%') THEN SUBSTRING(ICD9_CODE, 1, 3) 
                    WHEN(ICD9_CODE LIKE 'E%') THEN SUBSTRING(ICD9_CODE, 1, 4) 
                ELSE 
                    SUBSTRING(ICD9_CODE, 1, 3) END AS ICD9_CODE 
            FROM JAX_diagnosisProfile)
        SELECT 
            ICD9_CODE, COUNT(*) AS N
        FROM
            JAX_temp_diag
        GROUP BY 
            ICD9_CODE
        ORDER BY N
        DESC
        """)


def rankHpoFromText(diagnosis, hpo_min_occurrence_per_encounter):
    """
    Rank frequently seen phenotypes (HPO term) from text mining among encounters of interest.
    An encounter may have multiple occurrances of a phenotype term. A phenotype is called if its occurrance
    meets a minimum threshold.
    @param hpo_min_occurrence_per_encounter: threshold for a phenotype abnormality to be called. Usually use 1.
    """
    cursor.execute('DROP TEMPORARY TABLE IF EXISTS JAX_textHpoFrequencyRank')
    cursor.execute('''
            CREATE TEMPORARY TABLE JAX_textHpoFrequencyRank            
            WITH pd AS(
                SELECT 
                    JAX_textHpoProfile.*
                FROM 
                    JAX_textHpoProfile 
                JOIN (
                    SELECT 
                        DISTINCT SUBJECT_ID, HADM_ID
                    FROM 
                        JAX_diagnosisProfile 
                    WHERE 
                        ICD9_CODE LIKE '{}%') AS d
                ON 
                    JAX_textHpoProfile.SUBJECT_ID = d.SUBJECT_ID AND JAX_textHpoProfile.HADM_ID = d.HADM_ID
                WHERE 
                    OCCURRANCE >= {})
            SELECT 
                MAP_TO, COUNT(*) AS N, 1 AS PHENOTYPE
            FROM pd
            GROUP BY MAP_TO
            ORDER BY N DESC'''.format(diagnosis, hpo_min_occurrence_per_encounter))


def rankHpoFromLab(diagnosis, hpo_min_occurrence_per_encounter):
    """
    Rank frequently seen phenotypes (HPO term) from lab texts among encounters of interest.
    An encounter may have multiple occurrances of a phenotype term, such as from lab tests that are frequently ordered.
    A phenotype is called if its occurrance meets a minimum threshold.
    @param hpo_min_occurrence_per_encounter: threshold for a phenotype abnormality to be called.
    For example, if the parameter is set to 3, HP:0002153 Hyperkalemia is assigned iff three or more lab tests return higher than normal values for blood potassium concentrations
    """
    cursor.execute('DROP TEMPORARY TABLE IF EXISTS JAX_labHpoFrequencyRank')
    cursor.execute('''
            CREATE TEMPORARY TABLE JAX_labHpoFrequencyRank            
            WITH pd AS(
                SELECT 
                    JAX_labHpoProfile.*
                FROM 
                    JAX_labHpoProfile 
                JOIN (
                    SELECT 
                        DISTINCT SUBJECT_ID, HADM_ID
                    FROM 
                        JAX_diagnosisProfile 
                    WHERE 
                        ICD9_CODE LIKE '{}%') AS d
                ON 
                    JAX_labHpoProfile.SUBJECT_ID = d.SUBJECT_ID AND JAX_labHpoProfile.HADM_ID = d.HADM_ID
                WHERE
                    OCCURRANCE >= {})
            SELECT 
                MAP_TO, COUNT(*) AS N, 1 AS PHENOTYPE
            FROM pd
            GROUP BY MAP_TO
            ORDER BY N DESC'''.format(diagnosis, hpo_min_occurrence_per_encounter))

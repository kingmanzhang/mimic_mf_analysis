from mimic_mf_analysis import mydb
import pandas as pd


def get_labHpo_for_textHpo(textHpo):
    with mydb.cursor() as cursor:
        cursor.execute(f'''
            CREATE TEMPORARY TABLE textHpoRecords AS
            select *
            FROM JAX_textHpoProfile
            WHERE MAP_TO = '{textHpo}';
        ''')

        cursor.execute('''
            CREATE TEMPORARY TABLE labHpoForTextHpo AS
            SELECT JAX_labHpoProfile.*
            FROM JAX_labHpoProfile
            JOIN textHpoRecords on JAX_labHpoProfile.SUBJECT_ID = textHpoRecords.SUBJECT_ID AND
            JAX_labHpoProfile.HADM_ID = textHpoRecords.HADM_ID;
        ''')

        results = pd.read_sql('select * from labHpoForTextHpo', mydb)
        return results

# def get_labHpo_for_textHpo(textHpo, leadtime=0):
#     """
#     Given a phenotype recorded by a textHpo, find labHpo terms observed before the specified lead time.
#     Algorithm:
#     1. identify which patient has been observed with the query textHpo, and record the time of observation
#     2. query the labHpo and find all record before the date when the textHpo is observed.
#     :param textHpo: a phenotype observed from texts
#     :param leadtime: if not None, find lab-derived phenotypes with a leadtime before the text-derived phenotype
#     :return:
#     """
#     with mydb.cursor() as cursor:
#         cursor.execute(f'''
#             CREATE TEMPORARY TABLE labHpoForTextHpo
#             WITH textHpo_of_interest AS (
#             SELECT
#                 NOTEEVENTS.SUBJECT_ID, NOTEEVENTS.HADM_ID, NoteHpoClinPhen.MAP_TO, NOTEEVENTS.CHARTTIME
#             FROM
#                 NOTEEVENTS
#             JOIN NoteHpoClinPhen on NOTEEVENTS.ROW_ID = NoteHpoClinPhen.NOTES_ROW_ID
#             WHERE NoteHpoClinPhen.MAP_TO = '{textHpo}'
#             UNION ALL
#             SELECT
#                 NOTEEVENTS.SUBJECT_ID, NOTEEVENTS.HADM_ID, Inferred_NoteHpo.INFERRED_TO AS MAP_TO, NOTEEVENTS.CHARTTIME
#             FROM
#                 NOTEEVENTS
#             JOIN Inferred_NoteHpo on NOTEEVENTS.ROW_ID = Inferred_NoteHpo.NOTEEVENT_ROW_ID
#             WHERE Inferred_NoteHpo.INFERRED_TO = '{textHpo}'
#             ),
#             patientOfInterest AS (
#                 SELECT SUBJECT_ID, MIN(CHARTTIME) as earliest_observation_date
#                 FROM textHpo_of_interest
#                 GROUP BY SUBJECT_ID),
#             raw AS (
#                 select
#                     LABEVENTS.SUBJECT_ID, LABEVENTS.HADM_ID, LabHpo.MAP_TO, LABEVENTS.CHARTTIME
#                 FROM LabHpo
#                 JOIN LABEVENTS on LABEVENTS.ROW_ID = LabHpo.ROW_ID
#                 JOIN patientOfInterest on LABEVENTS.SUBJECT_ID = patientOfInterest.SUBJECT_ID
#                 WHERE LabHpo.NEGATED = 'F')
#             SELECT
#                 LABEVENTS.SUBJECT_ID, LABEVENTS.HADM_ID, INFERRED_LABHPO.INFERRED_TO AS MAP_TO, LABEVENTS.CHARTTIME
#             FROM
#                 INFERRED_LABHPO
#             JOIN
#                 LABEVENTS ON INFERRED_LABHPO.LABEVENT_ROW_ID = LABEVENTS.ROW_ID
#             JOIN patientOfInterest on LABEVENTS.SUBJECT_ID = patientOfInterest.SUBJECT_ID
#             UNION ALL
#             SELECT *
#             FROM raw
#         ''')
#
#         labHpo = pd.read_sql('''
#             SELECT * FROM labHpoForTextHpo;
#         ''', mydb)
#         print(labHpo.head())


if __name__=='__main__':
    # Thromboembolism
    get_labHpo_for_textHpo(textHpo='HP:0001907')
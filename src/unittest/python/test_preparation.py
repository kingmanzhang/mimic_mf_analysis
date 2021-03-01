import unittest
from mimic_mf_analysis import mydb
from mimic_mf_analysis.preparation import encounterOfInterest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        encounterOfInterest(debug=True)
        cursor = mydb.cursor()
        cursor.execute("select * from JAX_encounterOfInterest")
        data = cursor.fetchall()
        self.assertEqual(len(data), 100)


if __name__ == '__main__':
    unittest.main()

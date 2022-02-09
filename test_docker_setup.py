import unittest
from docker_setup import *

class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.yi = dockerSetUp()

    def test_add_table_columns(self):
        self.assertEqual(self.yi.check_table(),['CandidateID', 'Full_Name', 'Gender', 'DoB', 'Email', 'Full_Address', 'University', 'Degree', 'Invited_By', 'Geo_Flex', 'Course_Interest'])

    def test_all_tables_upload(self):
        my_list = ['BENCHMARK', 'Candidate_Course_J', 'Candidates', 'Course', 'Interview', 'Interview_Quality_J', 'Quality', 'sparta_day', 'sparta_day_results', 'Tech_Skill', 'Tech_Skill_Score_J', 'Trainer']
        assert_results = list(map(lambda x: x.upper(), my_list))
        self.assertEqual(self.yi.all_tables_upload(), assert_results)
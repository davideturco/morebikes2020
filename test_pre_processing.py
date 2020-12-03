from pre_processing import *
import pandas as pd


def test_day_transform():
    days = pd.DataFrame({'Days': ['Monday', 'Sunday', 'Sunday', 'Thursday']})
    days_num = pd.DataFrame({'Days': [1, 7, 7, 4]})

    pd.testing.assert_frame_equal(day_transform(days), days_num)

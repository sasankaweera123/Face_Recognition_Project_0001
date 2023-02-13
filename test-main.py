import unittest
import attendance


class TestAttendance(unittest.TestCase):

    def get_classnames(self):
        result = attendance.get_classnames()
        self.assertEqual(result, 16)

    def find_name(self):
        result = attendance.find_name(index=0, image=0)
        self.assertEqual(result, 16)


if __name__ == '__main__':
    unittest.main()


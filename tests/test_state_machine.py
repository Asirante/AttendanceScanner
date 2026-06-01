"""상태 머신 단위 테스트 (체크리스트: 5가지 전이 케이스 검증)."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import AttendanceDB

class TestStateMachine(unittest.TestCase):
    def setUp(self):

        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db = AttendanceDB(db_path=self.path)

        self.db.conn.execute(
            "INSERT OR REPLACE INTO Employees (emp_id, name, embedding) VALUES (?,?,?)",
            ("E001", "테스트", b"\x00" * 4),
        )
        self.db.conn.commit()

    def tearDown(self):
        self.db.close()
        os.remove(self.path)

    def test_1_first_visit_only_checkin(self):

        self.assertEqual(self.db.get_available_actions("E001"), ["출근"])

    def test_2_after_checkin(self):
        self.db.log_action("E001", "출근")
        self.assertEqual(self.db.get_available_actions("E001"), ["외출", "퇴근"])

    def test_3_after_step_out(self):
        self.db.log_action("E001", "출근")
        self.db.log_action("E001", "외출")
        self.assertEqual(self.db.get_available_actions("E001"), ["복귀"])

    def test_4_after_return(self):
        self.db.log_action("E001", "출근")
        self.db.log_action("E001", "외출")
        self.db.log_action("E001", "복귀")
        self.assertEqual(self.db.get_available_actions("E001"), ["외출", "퇴근"])

    def test_5_after_checkout_terminal(self):
        self.db.log_action("E001", "출근")
        self.db.log_action("E001", "퇴근")
        self.assertEqual(self.db.get_available_actions("E001"), [])

    def test_6_invalid_transition_raises(self):

        with self.assertRaises(ValueError):
            self.db.log_action("E001", "퇴근")

if __name__ == "__main__":
    unittest.main(verbosity=2)

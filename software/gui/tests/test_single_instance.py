import unittest
import uuid

from single_instance import SingleInstanceGuard


class SingleInstanceGuardTests(unittest.TestCase):
    def test_lock_is_atomic_and_released_on_close(self):
        name = f"WheeledLegRobot.GUI.Test.{uuid.uuid4()}"
        first = SingleInstanceGuard(name)
        self.assertTrue(first.acquired)
        try:
            second = SingleInstanceGuard(name)
            self.assertFalse(second.acquired)
            second.close()
        finally:
            first.close()

        third = SingleInstanceGuard(name)
        try:
            self.assertTrue(third.acquired)
        finally:
            third.close()


if __name__ == "__main__":
    unittest.main()

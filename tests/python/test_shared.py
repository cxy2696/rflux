import multiprocessing
import unittest
import uuid

import numpy as np

from rflux.shared import SharedArray


def _child_write(name, shape):
    with SharedArray.attach(name, shape) as shared:
        shared.array[:] = np.arange(shared.array.size) + 100.0


class TestSharedArray(unittest.TestCase):
    def test_create_attach_and_mutation(self):
        with SharedArray.create((4, 3), unlink_on_exit=True) as owner:
            owner.array[:] = np.arange(12).reshape(4, 3)
            with SharedArray.attach(owner.name, (4, 3)) as attached:
                attached.array[0, 0] = 99.0
                self.assertEqual(owner.array[0, 0], 99.0)
            self.assertEqual(owner.shape, (4, 3))
            self.assertEqual(owner.dtype, np.dtype(np.float64))

    def test_cross_process_visibility(self):
        name = f"rflux-test-{uuid.uuid4().hex[:12]}"
        with SharedArray.create((8,), name=name, unlink_on_exit=True) as owner:
            context = multiprocessing.get_context("spawn")
            process = context.Process(
                target=_child_write, args=(owner.name, (8,))
            )
            process.start()
            process.join(timeout=20)
            self.assertEqual(process.exitcode, 0)
            np.testing.assert_allclose(
                owner.array, np.arange(8.0) + 100.0
            )

    def test_view_and_lifecycle(self):
        with SharedArray.create((2, 2), unlink_on_exit=True) as shared:
            self.assertFalse(shared.closed)
            raw = shared.view(np.float32)
            self.assertEqual(raw.shape, (2, 4))
            shared.close()
            self.assertTrue(shared.closed)
            with self.assertRaises(RuntimeError):
                _ = shared.array
            shared.close()

    def test_rejects_invalid_attach(self):
        with SharedArray.create((2,), unlink_on_exit=True) as shared:
            with self.assertRaises(ValueError):
                SharedArray.attach(shared.name, (100_000,))

    def test_pickle_transfers_descriptor(self):
        import pickle

        with SharedArray.create((3,), unlink_on_exit=True) as owner:
            restored = pickle.loads(pickle.dumps(owner))
            try:
                self.assertEqual(restored.name, owner.name)
                np.testing.assert_array_equal(restored.array, owner.array)
            finally:
                restored.close()


if __name__ == "__main__":
    unittest.main()

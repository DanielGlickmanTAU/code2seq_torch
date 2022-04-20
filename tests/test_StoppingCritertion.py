from unittest import TestCase

from code2seq.utils.StoppingCritertion import StoppingCriterion


class TestStoppingCriterion(TestCase):
    def test_stop_when_converging_on_accuracy_thershold(self):
        critertion = StoppingCriterion(higher_is_better=True, threshold=1., desired_stable_evaluations=2)

        critertion(0.9)
        should_stop = critertion(0.9)
        assert not should_stop

        assert not critertion(0.9)
        assert not critertion(1.1)
        assert not critertion(0.9)
        assert not critertion(1.1)
        assert not critertion(0.9)
        assert not critertion(1.1)

        critertion(1.)
        assert critertion(1.)

    def test_early_stop(self):
        critertion = StoppingCriterion(higher_is_better=True, patience=1)
        assert not critertion(0.1)
        assert not critertion(0.2)
        assert not critertion(0.3)

        # not updating first time
        assert not critertion(0.3)
        # now updating should reset counter
        assert not critertion(0.4)
        # not updating again
        assert not critertion(0.4)
        # second time with no improvement, should stop
        assert critertion(0.4)

"""Tests for leakage boundaries and score-fusion controls."""
import unittest
import torch
from attribute_fusion_experiment import standardized, stratified_folds, select_weight, changes


class FusionTests(unittest.TestCase):
    def test_normalization_and_zero_weight(self):
        torch.manual_seed(18)
        scores = torch.randn(500, 75)
        torch.testing.assert_close(standardized(scores), standardized(scores*7+3), atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.equal(scores.argmax(-1), standardized(scores).argmax(-1)))
        self.assertTrue(torch.isfinite(standardized(torch.ones(2, 75))).all())

    def test_fold_selection_and_tie(self):
        labels = torch.repeat_interleave(torch.arange(75), torch.tensor([5]*50+[10]*25))
        folds = stratified_folds(labels, 5, 20260929)
        self.assertTrue(torch.equal(folds, stratified_folds(labels, 5, 20260929)))
        for fold in range(5):
            held = folds == fold
            self.assertEqual(int(held.sum()), 100)
            self.assertTrue(torch.equal(torch.bincount(labels[held]), torch.tensor([1]*50+[2]*25)))
        # A candidate perfect only on held-out rows cannot win fitting selection.
        held = folds == 0
        pred = torch.stack([labels.clone(), (labels+1) % 75])[:, None].repeat(1, 3, 1)
        pred[0, :, held] = (labels[held]+1) % 75
        pred[1, :, held] = labels[held]
        self.assertEqual(select_weight(pred[:, :, ~held], labels[~held], [0, .1])[0], 0)
        pred[1] = pred[0]
        self.assertEqual(select_weight(pred[:, :, ~held], labels[~held], [0, .1])[0], 0)

    def test_error_accounting(self):
        labels = torch.tensor([0, 1, 50, 51])
        native = torch.tensor([0, 2, 50, 52])
        fused = torch.tensor([3, 1, 50, 53])
        result = changes(fused, native, labels)
        self.assertEqual(result['all']['corrected'], 1)
        self.assertEqual(result['all']['newly_wrong'], 1)
        self.assertEqual(result['all']['net_correct'], 0)
        self.assertEqual(result['unseen']['wrong_to_different_wrong'], 1)


if __name__ == '__main__':
    unittest.main()

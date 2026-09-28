import unittest
from report_expanded import paired_statistics,holm


class PairedStatisticsTests(unittest.TestCase):
    def test_identical_predictions_have_no_gain(self):
        result=paired_statistics([0,1,0,1],[0,1,0,1],[0,1,1,1])
        self.assertEqual(result['mcnemar_exact_two_sided_p'],1)
        self.assertEqual(result['paired_stratified_bootstrap_95ci_pp'],[0,0])

    def test_exact_discordant_pair_probability_and_correction(self):
        # Six paired corrections and no regressions: 2 / 2**6.
        r=paired_statistics([1]*6,[0]*6,[0]*6)
        self.assertEqual(r['mcnemar_exact_two_sided_p'],.03125)
        self.assertEqual(r['improvement_percentage_points'],100)
        records=[r,dict(mcnemar_exact_two_sided_p=.5,improvement_percentage_points=-5)]
        holm(records)
        self.assertEqual(records[0]['holm_adjusted_p'],.0625)
        self.assertFalse(records[0]['significant_positive_gain'])


if __name__=='__main__':unittest.main()

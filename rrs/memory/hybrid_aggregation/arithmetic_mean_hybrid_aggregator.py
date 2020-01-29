from .hybrid_aggregator import HybridAggregator


class ArithmeticMeanHybridAggregator(HybridAggregator):

    def aggregate_scores(self, scores):
        num = 0.0
        for s in scores:
            num += float(s)

        den = float(len(scores))

        am = num / den
        return am

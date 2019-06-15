from .hybrid_aggregator import HybridAggregator


class ArithmeticMeanHybridAggregator(HybridAggregator):

    def aggregate_scores(self):
        num = 0.0
        for s in self.scores:
            num += float(s)

        den = float(len(s))

        am = num / den
        return am

from overrides import overrides

from experiments.analysis.ngrams.ngram_extractor_base import NgramsExtractor
from experiments.analysis.utils.lf_utils import TOKENS_TO_IGNORE


class NgramsExtractorSurfaceLevel(NgramsExtractor):
    def __init__(self, config):
        super().__init__(config)

    @overrides
    def _get_ngrams_from_target(self, target_tokens, freqs=False):
        tokens = [t for t in target_tokens if t not in TOKENS_TO_IGNORE]
        return set(tuple(tokens[i:i + 2]) for i in range(len(tokens) - 1))

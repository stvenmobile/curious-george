"""
The combined curiosity score - the ranking function Phase 1 and the
interest pipeline were always building toward.

Three axes, one score - but not combined as equal partners. Phase 1's
own real, replicated result (notebooks/curious_george.ipynb) is the
reason for the shape: mastery and resonance alone can't distinguish
genuine learnable structure from noise - `known` and `noise` both
showed negative transfer despite very different baseline loss and would
have had very different resonance too. Only transferability
(deep-scoring's generalization signal) actually predicted whether
studying something was worthwhile. So transferability is the dominant
term; mastery only matters as a hard stop once something is fully
learned (no room left to gain, regardless of how well it transfers);
resonance is a smaller preference/orientation boost on top of genuine
substance, not a substitute for it - a high-resonance candidate that's
actually noise should still score low.

Uses generalization_mean (RAW), not generalization_ratio_mean. Phase
1's own "known" result found the ratio's stdev is dominated by dividing
by a tiny, unstable memorization denominator when comparing topics with
very different memorization magnitudes - exactly the situation here
(arbitrary real candidates with very different content and lengths).
Raw generalization_progress was the recommended, more trustworthy
comparison across topics; see that notebook result for the numbers.

RESONANCE_WEIGHT and MASTERY_PENALTY are simple, round starting values,
not calibrated against enough real data to justify anything more
precise yet - explicitly provisional, worth adjusting once there's a
real track record to tune against.
"""

from typing import Dict, List, Optional, Tuple

from curious_george.memory_store import DeepScoreRecord, MemoryStore, PipelineStatus

RESONANCE_WEIGHT = 1.0
MASTERY_PENALTY = 10.0


def curiosity_score(mastery: Optional[str], resonance: Optional[float],
                     deep_score: Optional[DeepScoreRecord]) -> Optional[float]:
    """None if the candidate has never been deep-scored - a candidate
    can't be meaningfully judged for curiosity without that; mastery and
    resonance alone are exactly what Phase 1 showed isn't sufficient."""
    if deep_score is None or deep_score.generalization_mean is None:
        return None

    score = deep_score.generalization_mean
    if resonance is not None:
        score += RESONANCE_WEIGHT * resonance
    if mastery == "master":
        score -= MASTERY_PENALTY
    return score


def score_topic(store: MemoryStore, topic: str,
                 interest_embeddings: Optional[Dict] = None,
                 embedding_method: Optional[str] = None) -> Optional[float]:
    """Convenience wrapper pulling mastery/resonance/deep-score directly
    from the store for one topic, rather than requiring the caller to
    already have them in hand. Resonance is skipped (not an error) if
    interest_embeddings/embedding_method aren't both given - matches
    pipeline.add_candidate's own pattern for the same tradeoff."""
    mastery = store.mastery_level(topic)
    resonance = None
    if interest_embeddings and embedding_method:
        resonance = store.resonance(topic, embedding_method, interest_embeddings)
    deep_score = store.latest_deep_score(topic)
    return curiosity_score(mastery, resonance, deep_score)


def rank_candidates(store: MemoryStore, topics: Optional[List[str]] = None,
                     interest_embeddings: Optional[Dict] = None,
                     embedding_method: Optional[str] = None) -> List[Tuple[str, float]]:
    """Scores and ranks every requested topic (defaults to every ACTIVE
    item - the ones that survived deep-scoring), highest curiosity_score
    first. Topics that can't be scored (never deep-scored) are silently
    excluded, not returned as None entries - "what should I be curious
    about right now" only makes sense over things that CAN be judged."""
    if topics is None:
        topics = [item.topic for item in store.items if item.status == PipelineStatus.ACTIVE]

    scored = []
    for topic in topics:
        score = score_topic(store, topic, interest_embeddings, embedding_method)
        if score is not None:
            scored.append((topic, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored

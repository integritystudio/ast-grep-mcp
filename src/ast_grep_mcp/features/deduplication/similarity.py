"""
MinHash-based similarity calculation with hybrid two/three-stage pipeline.

Stage 1: Fast MinHash filter (O(n)) — eliminates dissimilar pairs quickly.
Stage 2: AST structural verification for Stage 1 candidates.
Stage 3: Optional CodeBERT semantic similarity for Type-4 clone detection.
"""

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple

from datasketch import MinHash, MinHashLSH

from ...constants import (
    ASTFingerprintDefaults,
    ConversionFactors,
    DeduplicationDefaults,
    FormattingDefaults,
    HybridSimilarityDefaults,
    IndentationDefaults,
    LogBucketThresholds,
    MinHashDefaults,
    SemanticSimilarityDefaults,
)
from ...core.logging import get_logger
from .scoring_scales import SimilarityDiscreteBand

COSINE_UNIT_INTERVAL_DIVISOR = 2.0


@dataclass
class SimilarityConfig:
    """Configuration for MinHash similarity calculation."""

    num_permutations: int = MinHashDefaults.NUM_PERMUTATIONS
    """Number of hash permutations. Higher = more accurate but slower."""

    shingle_size: int = MinHashDefaults.SHINGLE_SIZE
    """Token n-gram size. 3 is optimal for code."""

    similarity_threshold: float = DeduplicationDefaults.MIN_SIMILARITY
    """Minimum Jaccard similarity for LSH candidate retrieval."""

    use_token_shingles: bool = True
    """Use token-based shingles (True) vs character-based (False)."""

    small_code_token_threshold: int = MinHashDefaults.SMALL_CODE_TOKEN_THRESHOLD
    """Code with fewer tokens than this is considered 'small'."""

    lsh_recall_margin: float = MinHashDefaults.LSH_RECALL_MARGIN
    """LSH threshold offset below min_similarity to improve recall."""

    enable_small_code_fallback: bool = True
    """Fall back to all-pairs comparison for small datasets with no LSH candidates."""

    max_fallback_items: int = MinHashDefaults.MAX_FALLBACK_ITEMS
    """Maximum items for all-pairs fallback before O(n²) becomes too expensive."""

    small_code_threshold: int = MinHashDefaults.SEQUENCEMATCHER_TOKEN_THRESHOLD
    """Tokens below this use SequenceMatcher; MinHash is unreliable for small code."""

    use_small_code_fallback: bool = True
    """Use SequenceMatcher for code below small_code_threshold tokens."""


@dataclass
class SimilarityResult:
    """Result of a similarity calculation."""

    similarity: float
    """Estimated Jaccard similarity (0.0 to 1.0)."""

    method: str
    """Method used: 'minhash', 'exact', or 'hybrid'."""

    verified: bool = False
    candidate_count: int = 0


SimilarityMethod = Literal["minhash", "ast", "hybrid", "sequence_matcher"]


@dataclass
class HybridSimilarityConfig:
    """Configuration for the hybrid two/three-stage similarity pipeline."""

    minhash_early_exit_threshold: float = HybridSimilarityDefaults.MINHASH_EARLY_EXIT_THRESHOLD
    """MinHash similarity below this triggers early exit (skips Stage 2)."""

    minhash_weight: float = HybridSimilarityDefaults.MINHASH_WEIGHT
    ast_weight: float = HybridSimilarityDefaults.AST_WEIGHT
    min_tokens_for_ast: int = HybridSimilarityDefaults.MIN_TOKENS_FOR_AST
    max_lines_for_full_ast: int = HybridSimilarityDefaults.MAX_LINES_FOR_FULL_AST
    use_sequence_matcher_for_ast: bool = True

    # Stage 3: CodeBERT semantic similarity (requires transformers + torch)
    enable_semantic: bool = SemanticSimilarityDefaults.ENABLE_SEMANTIC
    semantic_weight: float = SemanticSimilarityDefaults.SEMANTIC_WEIGHT
    semantic_stage_threshold: float = SemanticSimilarityDefaults.SEMANTIC_STAGE_THRESHOLD
    """Minimum AST similarity to proceed to Stage 3."""
    semantic_model_name: str = SemanticSimilarityDefaults.MODEL_NAME
    semantic_device: str = SemanticSimilarityDefaults.DEFAULT_DEVICE

    def __post_init__(self) -> None:
        self._apply_semantic_rebalance_if_needed()
        self._validate_weight_bounds()
        self._validate_weight_sum()

    def _apply_semantic_rebalance_if_needed(self) -> None:
        """Rebalance to semantic weight profile when enabling semantic on legacy defaults."""
        uses_legacy_two_stage_defaults = (
            abs(self.minhash_weight - HybridSimilarityDefaults.MINHASH_WEIGHT) <= HybridSimilarityDefaults.WEIGHT_SUM_TOLERANCE
            and abs(self.ast_weight - HybridSimilarityDefaults.AST_WEIGHT) <= HybridSimilarityDefaults.WEIGHT_SUM_TOLERANCE
            and abs(self.semantic_weight - SemanticSimilarityDefaults.SEMANTIC_WEIGHT) <= HybridSimilarityDefaults.WEIGHT_SUM_TOLERANCE
        )
        if self.enable_semantic and uses_legacy_two_stage_defaults:
            self.minhash_weight = SemanticSimilarityDefaults.MINHASH_WEIGHT_WITH_SEMANTIC
            self.ast_weight = SemanticSimilarityDefaults.AST_WEIGHT_WITH_SEMANTIC

    def _validate_weight_bounds(self) -> None:
        """Raise ValueError if any weight is outside [0.0, 1.0]."""
        self._check_bound("minhash_early_exit_threshold", self.minhash_early_exit_threshold)
        self._check_bound("minhash_weight", self.minhash_weight)
        self._check_bound("ast_weight", self.ast_weight)
        self._check_bound("semantic_weight", self.semantic_weight)
        self._check_bound("semantic_stage_threshold", self.semantic_stage_threshold)

    @staticmethod
    def _check_bound(name: str, value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be between 0.0 and 1.0")

    @staticmethod
    def _check_weight_sum(total: float, msg: str) -> None:
        if abs(total - HybridSimilarityDefaults.WEIGHT_SUM_TARGET) > HybridSimilarityDefaults.WEIGHT_SUM_TOLERANCE:
            raise ValueError(msg)

    def _validate_weight_sum(self) -> None:
        """Raise ValueError if active weights do not sum to 1.0."""
        if self.enable_semantic:
            total = self.minhash_weight + self.ast_weight + self.semantic_weight
            self._check_weight_sum(
                total,
                f"minhash_weight + ast_weight + semantic_weight must equal 1.0 when semantic is enabled, got {total}",
            )
        else:
            total = self.minhash_weight + self.ast_weight
            self._check_weight_sum(total, f"minhash_weight + ast_weight must equal 1.0, got {total}")


@dataclass
class HybridSimilarityResult:
    """Result of hybrid two/three-stage similarity calculation."""

    similarity: float
    """Final combined similarity score (0.0 to 1.0)."""

    method: SimilarityMethod
    """'minhash' (early exit) or 'hybrid' (full pipeline)."""

    verified: bool
    minhash_similarity: float
    ast_similarity: Optional[float] = None
    semantic_similarity: Optional[float] = None
    stage1_passed: bool = False
    stage2_passed: bool = False
    early_exit: bool = False
    semantic_skipped: bool = False
    token_count: int = 0
    semantic_model: Optional[str] = None

    @property
    def combined_similarity(self) -> float:
        """Backward-compatible alias for final similarity score."""
        return self.similarity

    def to_dict(self) -> Dict[str, object]:
        """Convert result to dictionary for serialization."""
        return {
            "similarity": round(self.similarity, FormattingDefaults.SIMILARITY_PRECISION),
            "method": self.method,
            "verified": self.verified,
            "minhash_similarity": round(self.minhash_similarity, FormattingDefaults.SIMILARITY_PRECISION),
            "ast_similarity": round(self.ast_similarity, FormattingDefaults.SIMILARITY_PRECISION)
            if self.ast_similarity is not None
            else None,
            "semantic_similarity": round(self.semantic_similarity, FormattingDefaults.SIMILARITY_PRECISION)
            if self.semantic_similarity is not None
            else None,
            "stage1_passed": self.stage1_passed,
            "stage2_passed": self.stage2_passed,
            "early_exit": self.early_exit,
            "semantic_skipped": self.semantic_skipped,
            "token_count": self.token_count,
            "semantic_model": self.semantic_model,
        }


class MinHashSimilarity:
    """
    MinHash-based similarity calculator for code clone detection.

    Estimates Jaccard similarity in O(n) using token shingles and MinHash
    signatures, versus SequenceMatcher's O(n²).
    """

    def __init__(self, config: Optional[SimilarityConfig] = None) -> None:
        self.config = config or SimilarityConfig()
        self.logger = get_logger("deduplication.similarity")
        self._signature_cache: OrderedDict[str, MinHash] = OrderedDict()
        self._lsh_index: Optional[MinHashLSH] = None
        self._lsh_keys: Dict[str, int] = {}

    def create_minhash(self, code: str) -> MinHash:
        """Create a MinHash signature from code.

        Signatures are cached keyed on the exact code string (collision-safe,
        unlike ``hash()``) with an LRU bound of
        ``MinHashDefaults.SIGNATURE_CACHE_MAX_SIZE`` entries.
        """
        cached = self._signature_cache.get(code)
        if cached is not None:
            self._signature_cache.move_to_end(code)
            return cached

        m = MinHash(num_perm=self.config.num_permutations)

        if self.config.use_token_shingles:
            tokens = self._tokenize(code)
            shingles = self._create_token_shingles(tokens, self.config.shingle_size)
        else:
            shingles = self._create_char_shingles(code, self.config.shingle_size)

        for shingle in shingles:
            m.update(shingle.encode("utf8"))

        self._signature_cache[code] = m
        if len(self._signature_cache) > MinHashDefaults.SIGNATURE_CACHE_MAX_SIZE:
            self._signature_cache.popitem(last=False)
        return m

    def estimate_similarity(self, code1: str, code2: str) -> float:
        """Estimate similarity using MinHash, with SequenceMatcher fallback for small code."""
        if not code1 or not code2:
            return 0.0

        tokens1 = self._tokenize(code1)
        tokens2 = self._tokenize(code2)
        min_tokens = min(len(tokens1), len(tokens2))

        if self.config.use_small_code_fallback and min_tokens < self.config.small_code_threshold:
            self.logger.debug(
                "small_code_fallback",
                token_count=min_tokens,
                threshold=self.config.small_code_threshold,
                method="SequenceMatcher",
            )
            return SequenceMatcher(None, tokens1, tokens2).ratio()

        self.logger.debug(
            "minhash_standard_path",
            token_count=min_tokens,
            threshold=self.config.small_code_threshold,
        )
        m1 = self.create_minhash(code1)
        m2 = self.create_minhash(code2)
        return float(m1.jaccard(m2))

    def calculate_similarity(self, code1: str, code2: str) -> SimilarityResult:
        """Calculate similarity with full result details."""
        if not code1 or not code2:
            return SimilarityResult(similarity=0.0, method="exact", verified=True)

        similarity = self.estimate_similarity(code1, code2)
        return SimilarityResult(similarity=similarity, method="minhash", verified=False)

    def build_lsh_index(
        self,
        code_items: List[Tuple[str, str]],
        threshold: Optional[float] = None,
    ) -> None:
        """Build LSH index for fast near-duplicate queries."""
        effective_threshold = threshold if threshold is not None else self.config.similarity_threshold

        self._lsh_index = MinHashLSH(
            threshold=effective_threshold,
            num_perm=self.config.num_permutations,
        )
        self._lsh_keys = {}

        for key, code in code_items:
            m = self.create_minhash(code)
            self._insert_lsh_key(key, code, m)

        self.logger.info(
            "lsh_index_built",
            total_items=len(code_items),
            indexed_items=len(self._lsh_keys),
            threshold=effective_threshold,
        )

    def _insert_lsh_key(self, key: str, code: str, m: MinHash) -> None:
        try:
            self._lsh_index.insert(key, m)  # type: ignore[union-attr]
            self._lsh_keys[key] = hash(code)
        except ValueError:
            self.logger.debug("lsh_key_duplicate", key=key)

    def query_similar(self, code: str) -> List[str]:
        """Query the LSH index for similar code snippets."""
        if self._lsh_index is None:
            self.logger.warning("lsh_index_not_built")
            return []

        m = self.create_minhash(code)
        return list(self._lsh_index.query(m))

    def find_all_similar_pairs(
        self,
        code_items: List[Tuple[str, str]],
        min_similarity: float = DeduplicationDefaults.MIN_SIMILARITY,
        scorer: Optional[Callable[[str, str], float]] = None,
    ) -> List[Tuple[str, str, float]]:
        """Find all pairs of similar code snippets using LSH.

        Args:
            code_items: ``(key, code)`` pairs to compare.
            min_similarity: Threshold a pair must meet to be returned.
            scorer: Final similarity function applied to each LSH candidate.
                Defaults to this class's MinHash estimate. Callers with a more
                precise (and more expensive) measure should pass it here —
                MinHash under-estimates relative to AST/semantic scoring, so
                verifying with the estimate discards true positives that
                ``lsh_recall_margin`` deliberately admitted as candidates
                (TSD-07).
        """
        if not code_items:
            return []

        small_code_count = self._count_small_code_items(code_items)
        lsh_threshold = self._calculate_adaptive_threshold(min_similarity)
        self.build_lsh_index(code_items, threshold=lsh_threshold)
        candidates = self._find_lsh_candidates(code_items, lsh_threshold)

        use_fallback = self._should_use_fallback(candidates, code_items, small_code_count)
        if use_fallback:
            candidates = self._generate_all_pairs(code_items)

        similar_pairs = self._verify_candidates(candidates, code_items, min_similarity, scorer)
        self.logger.info(
            "similar_pairs_verified",
            candidates_checked=len(candidates),
            pairs_above_threshold=len(similar_pairs),
            min_similarity=min_similarity,
            used_fallback=use_fallback,
        )
        return similar_pairs

    def _count_small_code_items(self, code_items: List[Tuple[str, str]]) -> int:
        """Count items below the small code token threshold."""
        small_code_count = sum(1 for _, code in code_items if len(self._tokenize(code)) < self.config.small_code_token_threshold)

        if small_code_count > 0:
            self.logger.info(
                "small_code_detected",
                small_code_count=small_code_count,
                total_items=len(code_items),
                token_threshold=self.config.small_code_token_threshold,
            )

        return small_code_count

    def _calculate_adaptive_threshold(self, min_similarity: float) -> float:
        """Lower LSH threshold below min_similarity to improve recall."""
        return max(HybridSimilarityDefaults.LSH_THRESHOLD_FLOOR, min_similarity - self.config.lsh_recall_margin)

    def _find_lsh_candidates(
        self,
        code_items: List[Tuple[str, str]],
        lsh_threshold: float,
    ) -> Set[Tuple[str, str]]:
        """Find candidate pairs using the LSH index."""
        candidates: Set[Tuple[str, str]] = set()

        for key, code in code_items:
            pairs = {(min(key, sk), max(key, sk)) for sk in self.query_similar(code) if sk != key}
            candidates.update(pairs)

        self.logger.info(
            "lsh_candidates_found",
            total_items=len(code_items),
            candidate_pairs=len(candidates),
            lsh_threshold=lsh_threshold,
        )

        return candidates

    def _should_use_fallback(
        self,
        candidates: Set[Tuple[str, str]],
        code_items: List[Tuple[str, str]],
        small_code_count: int,
    ) -> bool:
        """Return True if all-pairs fallback should supplement or replace LSH results.

        TSD-08: the original guard returned ``False`` as soon as
        ``len(candidates) > 0``, which permanently suppressed the safety net
        even on tiny corpora where brute force is essentially free.  A
        partially populated candidate set never self-corrected.

        The updated logic retains the fallback for any corpus that is within
        the cheap O(n²) budget (≤ ``max_fallback_items``), regardless of
        whether LSH already found something.  On larger corpora the guard is
        unchanged: LSH is the only viable strategy and the fallback stays off.
        """
        if not self.config.enable_small_code_fallback:
            return False
        if len(code_items) > self.config.max_fallback_items:
            return False

        # On a corpus small enough for brute force, always run all-pairs so
        # LSH recall gaps on small inputs are automatically covered.
        use_fallback = small_code_count > 0 or len(code_items) <= 10

        if use_fallback:
            reason = "no_lsh_candidates" if len(candidates) == 0 else "small_corpus_safety_net"
            self.logger.info(
                "using_all_pairs_fallback",
                reason=reason,
                item_count=len(code_items),
                small_code_count=small_code_count,
                lsh_candidates=len(candidates),
            )

        return use_fallback

    def _verify_candidates(
        self,
        candidates: Set[Tuple[str, str]],
        code_items: List[Tuple[str, str]],
        min_similarity: float,
        scorer: Optional[Callable[[str, str], float]] = None,
    ) -> List[Tuple[str, str, float]]:
        """Filter candidate pairs to those meeting the minimum similarity threshold."""
        code_map = dict(code_items)
        score = scorer or self.estimate_similarity
        return [
            (k1, k2, sim)
            for k1, k2 in candidates
            for sim in (score(code_map[k1], code_map[k2]),)
            if sim >= min_similarity
        ]

    def _generate_all_pairs(
        self,
        code_items: List[Tuple[str, str]],
    ) -> Set[Tuple[str, str]]:
        """Generate all unique pairs for brute-force O(n²) fallback on small datasets."""
        pairs: Set[Tuple[str, str]] = set()
        keys = [key for key, _ in code_items]

        for i, key1 in enumerate(keys):
            for key2 in keys[i + 1 :]:
                pairs.add((min(key1, key2), max(key1, key2)))

        return pairs

    def clear_cache(self) -> None:
        """Clear the signature cache and LSH index."""
        self._signature_cache.clear()
        self._lsh_index = None
        self._lsh_keys.clear()

    def _tokenize(self, code: str) -> List[str]:
        """Tokenize code into identifiers, numbers, operators, and punctuation."""
        code = re.sub(r"\s+", " ", code)
        return re.findall(
            r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[+\-*/=<>!&|%^~]+|[(){}\[\],;:.]",
            code,
        )

    def _create_token_shingles(self, tokens: List[str], size: int) -> List[str]:
        """Create token n-gram shingles."""
        if len(tokens) < size:
            return [" ".join(tokens)] if tokens else []
        return [" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]

    def _create_char_shingles(self, text: str, size: int) -> List[str]:
        """Create character n-gram shingles."""
        if len(text) < size:
            return [text] if text else []
        return [text[i : i + size] for i in range(len(text) - size + 1)]


class HybridSimilarity:
    """
    Hybrid two/three-stage similarity calculator combining MinHash, AST, and CodeBERT.

    Stage 1: Fast MinHash filter (O(n)) — eliminates dissimilar pairs.
    Stage 2: Structural verification via SequenceMatcher on normalized code.
    Stage 3: CodeBERT semantic similarity (optional, requires transformers + torch).
    """

    def __init__(
        self,
        minhash_config: Optional[SimilarityConfig] = None,
        hybrid_config: Optional[HybridSimilarityConfig] = None,
    ) -> None:
        # Backward compatibility: some callers pass HybridSimilarityConfig as first arg.
        if isinstance(minhash_config, HybridSimilarityConfig) and hybrid_config is None:
            hybrid_config = minhash_config
            minhash_config = None

        self.minhash_config = minhash_config or SimilarityConfig()
        self.hybrid_config = hybrid_config or HybridSimilarityConfig()
        self.logger = get_logger("deduplication.hybrid_similarity")
        self._minhash = MinHashSimilarity(self.minhash_config)
        self._semantic: Optional["SemanticSimilarity"] = None
        self._semantic_available: Optional[bool] = None

    def _try_init_semantic(self) -> Optional["SemanticSimilarity"]:
        try:
            semantic_config = SemanticSimilarityConfig(
                model_name=self.hybrid_config.semantic_model_name,
                device=self.hybrid_config.semantic_device,
            )
            self._semantic = SemanticSimilarity(semantic_config)
            self._semantic_available = True
            return self._semantic
        except Exception as e:
            self._semantic_available = False
            self.logger.warning("semantic_similarity_init_failed", error=str(e))
            return None

    def _get_semantic_calculator(self) -> Optional["SemanticSimilarity"]:
        """Get or lazily create the semantic similarity calculator."""
        if not self.hybrid_config.enable_semantic:
            return None
        if self._semantic_available is False:
            return None
        if self._semantic is not None:
            return self._semantic
        if not _check_transformers_available():
            self._semantic_available = False
            self.logger.info("semantic_similarity_unavailable", reason="transformers or torch not installed")
            return None
        return self._try_init_semantic()

    def _build_empty_similarity_result(self) -> HybridSimilarityResult:
        return HybridSimilarityResult(
            similarity=0.0,
            method="minhash",
            verified=False,
            minhash_similarity=0.0,
            ast_similarity=None,
            semantic_similarity=None,
            stage1_passed=False,
            stage2_passed=False,
            early_exit=True,
            semantic_skipped=True,
            token_count=0,
            semantic_model=None,
        )

    def _run_stage1_filter(
        self,
        code1: str,
        code2: str,
        tokens1: List[str],
        tokens2: List[str],
        avg_token_count: int,
    ) -> Tuple[float, Optional[HybridSimilarityResult]]:
        """Run Stage 1 MinHash filter. Returns (score, early_exit_result or None)."""
        minhash_sim = self._minhash.estimate_similarity(code1, code2)

        below_threshold = minhash_sim < self.hybrid_config.minhash_early_exit_threshold
        if below_threshold and self.hybrid_config.enable_semantic and self.minhash_config.use_small_code_fallback:
            minhash_sim = max(minhash_sim, SequenceMatcher(None, tokens1, tokens2).ratio())

        if minhash_sim >= self.hybrid_config.minhash_early_exit_threshold:
            return minhash_sim, None

        self.logger.debug(
            "hybrid_early_exit",
            minhash_similarity=round(minhash_sim, FormattingDefaults.SIMILARITY_PRECISION),
            threshold=self.hybrid_config.minhash_early_exit_threshold,
        )
        return minhash_sim, HybridSimilarityResult(
            similarity=minhash_sim,
            method="minhash",
            verified=False,
            minhash_similarity=minhash_sim,
            ast_similarity=None,
            semantic_similarity=None,
            stage1_passed=False,
            stage2_passed=False,
            early_exit=True,
            semantic_skipped=True,
            token_count=avg_token_count,
            semantic_model=None,
        )

    def _build_two_stage_result(self, minhash_sim: float, ast_sim: float, token_count: int) -> HybridSimilarityResult:
        combined = self.hybrid_config.minhash_weight * minhash_sim + self.hybrid_config.ast_weight * ast_sim
        stage2_passed = ast_sim >= self.hybrid_config.semantic_stage_threshold
        self.logger.debug(
            "hybrid_verification_complete",
            minhash_similarity=round(minhash_sim, FormattingDefaults.SIMILARITY_PRECISION),
            ast_similarity=round(ast_sim, FormattingDefaults.SIMILARITY_PRECISION),
            combined_similarity=round(combined, FormattingDefaults.SIMILARITY_PRECISION),
            weights=f"{self.hybrid_config.minhash_weight}/{self.hybrid_config.ast_weight}",
            semantic_enabled=self.hybrid_config.enable_semantic,
            semantic_skipped=True,
        )
        return HybridSimilarityResult(
            similarity=combined,
            method="hybrid",
            verified=True,
            minhash_similarity=minhash_sim,
            ast_similarity=ast_sim,
            semantic_similarity=None,
            stage1_passed=True,
            stage2_passed=stage2_passed,
            early_exit=False,
            semantic_skipped=True,
            token_count=token_count,
            semantic_model=None,
        )

    def calculate_hybrid_similarity(
        self,
        code1: str,
        code2: str,
    ) -> HybridSimilarityResult:
        """Calculate similarity using the hybrid two/three-stage pipeline."""
        if not code1 or not code2:
            return self._build_empty_similarity_result()

        tokens1 = self._minhash._tokenize(code1)
        tokens2 = self._minhash._tokenize(code2)
        avg_token_count = (len(tokens1) + len(tokens2)) // 2

        minhash_sim, early_exit_result = self._run_stage1_filter(code1, code2, tokens1, tokens2, avg_token_count)
        if early_exit_result is not None:
            return early_exit_result

        ast_sim = self._calculate_ast_similarity(code1, code2)
        semantic_calc = self._get_semantic_calculator()

        if semantic_calc is not None and ast_sim >= self.hybrid_config.semantic_stage_threshold:
            return self._calculate_with_semantic(code1, code2, minhash_sim, ast_sim, avg_token_count, semantic_calc)

        return self._build_two_stage_result(minhash_sim, ast_sim, avg_token_count)

    def _calculate_with_semantic(
        self,
        code1: str,
        code2: str,
        minhash_sim: float,
        ast_sim: float,
        token_count: int,
        semantic_calc: "SemanticSimilarity",
    ) -> HybridSimilarityResult:
        """Run Stage 3 semantic similarity, falling back to two-stage on error."""
        try:
            semantic_sim = semantic_calc.calculate_similarity(code1, code2)
            return self._build_three_stage_result(minhash_sim, ast_sim, semantic_sim, token_count, semantic_calc)
        except Exception as e:
            self.logger.warning("semantic_similarity_failed", error=str(e), fallback="two_stage")
            combined = self.hybrid_config.minhash_weight * minhash_sim + self.hybrid_config.ast_weight * ast_sim
            return HybridSimilarityResult(
                similarity=combined,
                method="hybrid",
                verified=True,
                minhash_similarity=minhash_sim,
                ast_similarity=ast_sim,
                semantic_similarity=None,
                stage1_passed=True,
                stage2_passed=True,
                early_exit=False,
                semantic_skipped=True,
                token_count=token_count,
                semantic_model=None,
            )

    def _build_three_stage_result(
        self,
        minhash_sim: float,
        ast_sim: float,
        semantic_sim: float,
        token_count: int,
        semantic_calc: "SemanticSimilarity",
    ) -> HybridSimilarityResult:
        combined = (
            self.hybrid_config.minhash_weight * minhash_sim
            + self.hybrid_config.ast_weight * ast_sim
            + self.hybrid_config.semantic_weight * semantic_sim
        )
        self.logger.debug(
            "hybrid_semantic_complete",
            minhash_similarity=round(minhash_sim, FormattingDefaults.SIMILARITY_PRECISION),
            ast_similarity=round(ast_sim, FormattingDefaults.SIMILARITY_PRECISION),
            semantic_similarity=round(semantic_sim, FormattingDefaults.SIMILARITY_PRECISION),
            combined_similarity=round(combined, FormattingDefaults.SIMILARITY_PRECISION),
            weights=f"{self.hybrid_config.minhash_weight}/{self.hybrid_config.ast_weight}/{self.hybrid_config.semantic_weight}",
        )
        return HybridSimilarityResult(
            similarity=combined,
            method="hybrid",
            verified=True,
            minhash_similarity=minhash_sim,
            ast_similarity=ast_sim,
            semantic_similarity=semantic_sim,
            stage1_passed=True,
            stage2_passed=True,
            early_exit=False,
            semantic_skipped=False,
            token_count=token_count,
            semantic_model=semantic_calc.config.model_name,
        )

    def _calculate_ast_similarity(self, code1: str, code2: str) -> float:
        """Calculate structural similarity via SequenceMatcher on normalized code."""
        lines1 = len(code1.split("\n"))
        lines2 = len(code2.split("\n"))

        if max(lines1, lines2) > self.hybrid_config.max_lines_for_full_ast:
            return self._calculate_simplified_ast_similarity(code1, code2)

        norm1 = self._normalize_for_ast(code1)
        norm2 = self._normalize_for_ast(code2)
        return SequenceMatcher(None, norm1, norm2).ratio()

    def _normalize_for_ast(self, code: str) -> str:
        """Strip comments, blank lines, and normalize indentation for structural comparison."""
        lines = []
        for line in code.split("\n"):
            line = line.rstrip()
            stripped = line.strip()
            if not stripped or self._is_comment_line(stripped):
                continue
            line = self._strip_inline_comments(line)
            normalized_line = self._normalize_indentation(line)
            if normalized_line.strip():
                lines.append(normalized_line)
        return "\n".join(lines)

    @staticmethod
    def _is_comment_line(stripped: str) -> bool:
        """Return True if the line is a Python # or JS // comment."""
        return stripped.startswith("#") or stripped.startswith("//")

    @staticmethod
    def _strip_hash_comment(line: str) -> str:
        if "#" not in line:
            return line
        quote_count = line.count('"') + line.count("'")
        hash_pos = line.find("#")
        if quote_count % 2 == 0 and hash_pos > 0:
            return line[:hash_pos].rstrip()
        return line

    @staticmethod
    def _strip_slash_comment(line: str) -> str:
        if "//" not in line:
            return line
        comment_pos = line.find("//")
        return line[:comment_pos].rstrip() if comment_pos > 0 else line

    @staticmethod
    def _strip_inline_comments(line: str) -> str:
        """Remove inline Python # and JS // comments from a line."""
        line = HybridSimilarity._strip_hash_comment(line)
        return HybridSimilarity._strip_slash_comment(line)

    @staticmethod
    def _normalize_indentation(line: str) -> str:
        """Canonicalize leading whitespace to 4-space units."""
        indent_count = len(line) - len(line.lstrip())
        divisor = (
            IndentationDefaults.SPACES_PER_LEVEL
            if indent_count % IndentationDefaults.SPACES_PER_LEVEL == 0
            else IndentationDefaults.ALT_SPACES_PER_LEVEL
        )
        indent_level = indent_count // divisor
        return "    " * indent_level + line.lstrip()

    def _calculate_simplified_ast_similarity(self, code1: str, code2: str) -> float:
        """Jaccard similarity of structural keyword patterns for large code."""
        patterns1 = self._extract_structural_patterns(code1)
        patterns2 = self._extract_structural_patterns(code2)

        if not patterns1 or not patterns2:
            return 0.0

        set1 = set(patterns1)
        set2 = set(patterns2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    _STRUCTURAL_KEYWORDS: frozenset[str] = frozenset(
        {
            "def",
            "class",
            "if",
            "elif",
            "else",
            "for",
            "while",
            "try",
            "except",
            "finally",
            "with",
            "return",
            "yield",
            "async",
            "await",
            "function",
            "const",
            "let",
            "var",
        }
    )

    def _extract_structural_patterns(self, code: str) -> List[str]:
        """Extract indented control-flow keyword sequences for structural comparison."""
        patterns = []
        prev_keyword = None

        for line in code.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            words = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", stripped)
            kw = next((w.lower() for w in words if w.lower() in self._STRUCTURAL_KEYWORDS), None)
            if kw is None:
                continue

            indent_level = (len(line) - len(stripped)) // 2
            pattern = f"L{indent_level}:{kw}"
            if prev_keyword:
                pattern = f"{prev_keyword}->{pattern}"
            patterns.append(pattern)
            prev_keyword = kw

        return patterns

    def estimate_similarity(self, code1: str, code2: str) -> float:
        """Convenience wrapper returning only the similarity score."""
        return self.calculate_hybrid_similarity(code1, code2).similarity

    def clear_cache(self) -> None:
        """Clear the MinHash signature cache."""
        self._minhash.clear_cache()


@dataclass
class SimilarityBucket:
    """A bucket of potentially similar code items."""

    structure_hash: int
    items: List[Tuple[str, str]] = field(default_factory=list)


class EnhancedStructureHash:
    """
    Structure hash using AST-like node sequence patterns for bucket distribution.

    Combines five factors: node sequence fingerprint, control flow complexity,
    call pattern signature, nesting depth, and logarithmic size bucket.
    """

    # Maps keywords/patterns to pseudo-AST node types
    NODE_TYPES: Dict[str, str] = {
        "def": "FN",
        "function": "FN",
        "async": "AS",
        "class": "CL",
        "if": "IF",
        "else": "EL",
        "elif": "EI",
        "switch": "SW",
        "case": "CA",
        "match": "MA",
        "for": "FR",
        "while": "WH",
        "do": "DO",
        "try": "TR",
        "except": "EX",
        "catch": "CT",
        "finally": "FI",
        "raise": "RA",
        "throw": "TH",
        "return": "RT",
        "yield": "YD",
        "with": "WT",
        "lambda": "LM",
        "const": "CN",
        "let": "LT",
        "var": "VR",
        "import": "IM",
        "from": "FM",
        "require": "RQ",
        "assert": "AS",
        "pass": "PS",
        "break": "BK",
        "continue": "CO",
    }

    CALL_PATTERN = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")

    def __init__(self) -> None:
        self.logger = get_logger("deduplication.structure_hash")

    def calculate(self, code: str) -> int:
        """Calculate structure hash combining node sequence, complexity, calls, depth, and size."""
        node_sequence = self._extract_node_sequence(code)

        node_fingerprint = "->".join(node_sequence[: ASTFingerprintDefaults.MAX_NODE_SEQUENCE_LENGTH])
        complexity = self._calculate_control_flow_complexity(node_sequence)
        call_signature = self._extract_call_signature(code)
        max_depth = self._estimate_nesting_depth(code)
        lines = [line for line in code.split("\n") if line.strip()]
        size_bucket = self._logarithmic_bucket(len(lines))

        fingerprint_parts = [
            f"N:{node_fingerprint}",
            f"X{min(complexity, ASTFingerprintDefaults.MAX_COMPLEXITY_HEX_VALUE):X}",
            f"C:{call_signature}",
            f"D{min(max_depth, ASTFingerprintDefaults.MAX_NESTING_DEPTH_DIGIT)}",
            f"S{size_bucket:02d}",
        ]

        fingerprint = "|".join(fingerprint_parts)
        struct_hash = hash(fingerprint) % ASTFingerprintDefaults.HASH_MODULO
        return struct_hash * ASTFingerprintDefaults.HASH_BUCKET_MULTIPLIER + size_bucket

    def _extract_node_sequence(self, code: str) -> List[str]:
        """Extract pseudo-AST node type sequence from code."""
        nodes: List[str] = []
        for line in code.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                continue
            tokens = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", stripped)
            node = next((self.NODE_TYPES[t.lower()] for t in tokens if t.lower() in self.NODE_TYPES), None)
            if node is not None:
                nodes.append(node)
        return nodes

    def _calculate_control_flow_complexity(self, nodes: List[str]) -> int:
        """Approximate cyclomatic complexity by counting decision-point nodes."""
        decision_nodes = {"IF", "EI", "FR", "WH", "TR", "EX", "CT", "CA", "MA"}
        return 1 + sum(1 for n in nodes if n in decision_nodes)

    _CALL_EXCLUDED: frozenset[str] = frozenset(
        {
            "if",
            "for",
            "while",
            "with",
            "elif",
            "match",
            "case",
            "except",
            "try",
            "catch",
            "finally",
            "print",
            "return",
            "assert",
            "raise",
            "yield",
            "pass",
            "def",
            "class",
            "lambda",
            "async",
            "len",
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "set",
            "tuple",
            "range",
            "enumerate",
            "zip",
            "map",
            "filter",
            "sorted",
            "reversed",
            "type",
            "isinstance",
            "hasattr",
            "getattr",
            "setattr",
            "open",
            "input",
            "super",
            "property",
            "staticmethod",
            "classmethod",
        }
    )

    _DEF_PATTERNS = (
        (r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)", "def "),
        (r"^class\s+([a-zA-Z_][a-zA-Z0-9_]*)", "class "),
        (r"^async\s+def\s+([a-zA-Z_][a-zA-Z0-9_]*)", "async def "),
    )

    def _defined_name_from_line(self, stripped: str) -> Optional[str]:
        for pattern, prefix in self._DEF_PATTERNS:
            if stripped.startswith(prefix):
                m = re.match(pattern, stripped)
                return m.group(1) if m else None
        return None

    def _collect_defined_names(self, code: str) -> set[str]:
        names = (self._defined_name_from_line(line.strip()) for line in code.split("\n"))
        return {n for n in names if n is not None}

    def _extract_call_signature(self, code: str) -> str:
        """4-character hex hash of called (non-builtin) function names."""
        calls = self.CALL_PATTERN.findall(code)
        defined_names = self._collect_defined_names(code)
        filtered_calls = [c for c in calls if c.lower() not in self._CALL_EXCLUDED and c not in defined_names]

        if not filtered_calls:
            return "0000"

        unique_calls = sorted(set(filtered_calls))[: ASTFingerprintDefaults.MAX_UNIQUE_CALLS]
        call_str = ",".join(unique_calls)
        return f"{hash(call_str) % ASTFingerprintDefaults.CALL_SIGNATURE_BITMASK:0{ASTFingerprintDefaults.CALL_SIGNATURE_HEX_WIDTH}X}"

    @staticmethod
    def _line_nesting_depth(line: str) -> int:
        indent = len(line) - len(line.lstrip())
        divisor = (
            IndentationDefaults.ALT_SPACES_PER_LEVEL
            if indent % IndentationDefaults.SPACES_PER_LEVEL != 0
            else IndentationDefaults.SPACES_PER_LEVEL
        )
        return indent // divisor

    def _estimate_nesting_depth(self, code: str) -> int:
        """Estimate maximum nesting depth from indentation."""
        return max(
            (self._line_nesting_depth(line) for line in code.split("\n") if line.strip()),
            default=0,
        )

    def _logarithmic_bucket(self, line_count: int) -> int:
        """Map line count to a logarithmic size bucket (0–9) for uniform distribution."""
        thresholds = (
            (LogBucketThresholds.TINY, 0),
            (LogBucketThresholds.SMALL, 1),
            (LogBucketThresholds.MEDIUM, 2),
            (LogBucketThresholds.LARGE, int(SimilarityDiscreteBand.BAND_3)),
            (LogBucketThresholds.VERY_LARGE, int(SimilarityDiscreteBand.BAND_4)),
            (LogBucketThresholds.HUGE, int(SimilarityDiscreteBand.BAND_5)),
            (LogBucketThresholds.MASSIVE, int(SimilarityDiscreteBand.BAND_6)),
        )
        for limit, bucket in thresholds:
            if line_count < limit:
                return bucket
        return min(
            LogBucketThresholds.OVERFLOW_BASE_BUCKET + (line_count - LogBucketThresholds.MASSIVE) // LogBucketThresholds.MASSIVE,
            LogBucketThresholds.MAX_BUCKET,
        )

    def create_buckets(
        self,
        code_items: List[Tuple[str, str]],
    ) -> Dict[int, SimilarityBucket]:
        """Create structure hash buckets for code items."""
        buckets: Dict[int, SimilarityBucket] = {}

        for key, code in code_items:
            structure_hash = self.calculate(code)
            if structure_hash not in buckets:
                buckets[structure_hash] = SimilarityBucket(structure_hash=structure_hash)
            buckets[structure_hash].items.append((key, code))

        self.logger.info(
            "structure_buckets_created",
            total_items=len(code_items),
            bucket_count=len(buckets),
            avg_bucket_size=len(code_items) / len(buckets) if buckets else 0,
        )

        return buckets


# Optional imports for semantic similarity (CodeBERT)
# Loaded lazily when SemanticSimilarity is instantiated.
_TRANSFORMERS_AVAILABLE: Optional[bool] = None
_torch: Any = None
_transformers: Any = None


def _check_transformers_available() -> bool:
    """Return True if both transformers and torch can be imported."""
    global _TRANSFORMERS_AVAILABLE, _torch, _transformers

    if _TRANSFORMERS_AVAILABLE is not None:
        return _TRANSFORMERS_AVAILABLE

    try:
        import torch
        import transformers

        _torch = torch
        _transformers = transformers
        _TRANSFORMERS_AVAILABLE = True
    except ImportError:
        _TRANSFORMERS_AVAILABLE = False

    return _TRANSFORMERS_AVAILABLE


@dataclass
class SemanticSimilarityConfig:
    """Configuration for CodeBERT-based semantic similarity (Type-4 clone detection)."""

    model_name: str = "microsoft/codebert-base"
    max_length: int = SemanticSimilarityDefaults.MAX_TOKEN_LENGTH
    device: str = "auto"
    """Device for inference: 'auto', 'cpu', 'cuda', or 'mps'."""
    batch_size: int = SemanticSimilarityDefaults.DEFAULT_BATCH_SIZE
    cache_embeddings: bool = True
    normalize_embeddings: bool = True
    """L2-normalize embeddings (recommended for cosine similarity)."""


@dataclass
class SemanticSimilarityResult:
    """Result of CodeBERT semantic similarity calculation."""

    similarity: float
    """Cosine similarity between code embeddings (0.0 to 1.0)."""

    model_used: str
    embedding_dim: int
    """Dimensionality of the embeddings (768 for CodeBERT)."""

    truncated: bool = False
    code1_tokens: int = 0
    code2_tokens: int = 0
    computation_time_ms: int = 0

    @property
    def model_name(self) -> str:
        """Backward-compatible alias for model identifier."""
        return self.model_used

    @property
    def embedding1_shape(self) -> Tuple[int]:
        """Backward-compatible embedding shape for first snippet."""
        return (self.embedding_dim,)

    @property
    def embedding2_shape(self) -> Tuple[int]:
        """Backward-compatible embedding shape for second snippet."""
        return (self.embedding_dim,)

    def to_dict(self) -> Dict[str, object]:
        """Convert result to dictionary for serialization."""
        return {
            "similarity": round(self.similarity, FormattingDefaults.SIMILARITY_PRECISION),
            "model_used": self.model_used,
            "model_name": self.model_used,
            "embedding_dim": self.embedding_dim,
            "embedding1_shape": self.embedding1_shape,
            "embedding2_shape": self.embedding2_shape,
            "truncated": self.truncated,
            "code1_tokens": self.code1_tokens,
            "code2_tokens": self.code2_tokens,
            "computation_time_ms": self.computation_time_ms,
        }


class SemanticSimilarity:
    """
    CodeBERT-based semantic similarity for Type-4 clone detection.

    Generates 768-dim embeddings capturing semantic meaning, enabling detection
    of functionally equivalent code with different implementations.
    Requires optional deps: `pip install transformers torch`
    """

    def __init__(self, config: Optional[SemanticSimilarityConfig] = None) -> None:
        """
        Raises ImportError if transformers or torch are not installed.
        """
        self.config = config or SemanticSimilarityConfig()
        self.logger = get_logger("deduplication.semantic_similarity")
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Optional[str] = None
        self._embedding_cache: OrderedDict[str, Any] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._initialized = False

    @staticmethod
    def is_available() -> bool:
        """Return True if transformers and torch are installed."""
        return _check_transformers_available()

    def _init_model_components(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)  # type: ignore[no-untyped-call,unused-ignore]
        self._model = AutoModel.from_pretrained(self.config.model_name)  # type: ignore[no-untyped-call,unused-ignore]
        self._device = self._select_device(torch_module=torch)
        self._model = self._model.to(self._device)
        self._model.eval()
        self._initialized = True
        self.logger.info("codebert_model_loaded", model_name=self.config.model_name, device=self._device)

    def _load_model(self) -> None:
        """Lazily load the CodeBERT model (~400MB) on first use."""
        if self._initialized:
            return
        if not _check_transformers_available():
            raise ImportError(
                "Semantic similarity requires 'transformers' and 'torch' packages. Install with: pip install transformers torch"
            )
        self.logger.info("loading_codebert_model", model_name=self.config.model_name)
        try:
            self._init_model_components()
        except Exception as e:
            self.logger.error("codebert_model_load_failed", model_name=self.config.model_name, error=str(e))
            raise RuntimeError(f"Failed to load CodeBERT model: {e}") from e

    def _select_device(self, torch_module: Any) -> str:
        """Select the best available device: CUDA > MPS > CPU."""
        if self.config.device != "auto":
            return self.config.device
        if torch_module.cuda.is_available():
            return "cuda"
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _run_model_inference(self, code: str) -> Any:
        import torch

        inputs = self._tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
        if self.config.normalize_embeddings:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
        return embedding

    def get_embedding(self, code: str) -> Any:
        """Generate a (768,) embedding tensor for code using CodeBERT.

        Embeddings are cached keyed on the exact code string (collision-safe,
        unlike ``hash()``) with an LRU bound of
        ``SemanticSimilarityDefaults.EMBEDDING_CACHE_MAX_SIZE`` entries.
        """
        self._load_model()
        if self.config.cache_embeddings:
            cached = self._embedding_cache.get(code)
            if cached is not None:
                self._cache_hits += 1
                self._embedding_cache.move_to_end(code)
                return cached
        self._cache_misses += 1
        embedding = self._run_model_inference(code)
        if self.config.cache_embeddings:
            self._embedding_cache[code] = embedding
            if len(self._embedding_cache) > SemanticSimilarityDefaults.EMBEDDING_CACHE_MAX_SIZE:
                self._embedding_cache.popitem(last=False)
        return embedding

    def calculate_similarity(self, code1: str, code2: str) -> float:
        """Cosine similarity between CodeBERT embeddings, blended with lexical overlap."""
        if not code1 or not code2:
            return 0.0

        import torch.nn.functional as functional

        emb1 = self.get_embedding(code1)
        emb2 = self.get_embedding(code2)

        embedding_cosine: float = functional.cosine_similarity(
            emb1.unsqueeze(0),
            emb2.unsqueeze(0),
            dim=1,
        ).item()

        # Normalize cosine to [0,1] and blend with lexical score.
        # CodeBERT cosine is anisotropic for short snippets; lexical signal improves separation.
        semantic_score = (embedding_cosine + 1.0) / COSINE_UNIT_INTERVAL_DIVISOR
        lexical_score = self._calculate_lexical_similarity(code1, code2)
        similarity = 0.5 * semantic_score + 0.5 * lexical_score
        return max(0.0, min(1.0, similarity))

    def _calculate_lexical_similarity(self, code1: str, code2: str) -> float:
        """Token-level lexical similarity for semantic calibration."""
        tokens1 = re.findall(
            r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[+\-*/=<>!&|%^~]+|[(){}\[\],;:.]",
            re.sub(r"\s+", " ", code1),
        )
        tokens2 = re.findall(
            r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[+\-*/=<>!&|%^~]+|[(){}\[\],;:.]",
            re.sub(r"\s+", " ", code2),
        )
        if not tokens1 or not tokens2:
            return 0.0
        return float(SequenceMatcher(None, tokens1, tokens2).ratio())

    def calculate_similarity_detailed(self, code1: str, code2: str) -> SemanticSimilarityResult:
        """Calculate semantic similarity with full result metadata."""
        if not code1 or not code2:
            return SemanticSimilarityResult(
                similarity=0.0,
                model_used=self.config.model_name,
                embedding_dim=SemanticSimilarityDefaults.EMBEDDING_DIM,
                truncated=False,
                code1_tokens=0,
                code2_tokens=0,
                computation_time_ms=0,
            )

        self._load_model()
        start_time = time.perf_counter()

        tokens1: int = self._tokenizer(code1, return_tensors="pt")["input_ids"].shape[1]
        tokens2: int = self._tokenizer(code2, return_tensors="pt")["input_ids"].shape[1]
        truncated = tokens1 > self.config.max_length or tokens2 > self.config.max_length

        similarity = self.calculate_similarity(code1, code2)
        elapsed_ms = int((time.perf_counter() - start_time) * ConversionFactors.MILLISECONDS_PER_SECOND)

        return SemanticSimilarityResult(
            similarity=similarity,
            model_used=self.config.model_name,
            embedding_dim=768,
            truncated=truncated,
            code1_tokens=tokens1,
            code2_tokens=tokens2,
            computation_time_ms=elapsed_ms,
        )

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self.logger.debug("embedding_cache_cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Return cache size, hits, and misses."""
        return {
            "cache_size": len(self._embedding_cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
        }

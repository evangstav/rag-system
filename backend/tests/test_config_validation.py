"""
Quick test to verify hybrid search weight validation.
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pydantic import ValidationError


def test_valid_weights():
    """Test that valid weights are accepted."""
    from app.config import Settings

    # Valid: both weights between 0 and 1
    settings = Settings(
        OPENAI_API_KEY="test-key",
        HYBRID_SEARCH_SEMANTIC_WEIGHT=0.5,
        HYBRID_SEARCH_KEYWORD_WEIGHT=0.5,
    )
    assert settings.hybrid_search_semantic_weight == 0.5
    assert settings.hybrid_search_keyword_weight == 0.5
    print("✓ Valid weights (0.5, 0.5) accepted")


def test_invalid_semantic_weight_negative():
    """Test that negative semantic weight is rejected."""
    from app.config import Settings

    try:
        Settings(
            OPENAI_API_KEY="test-key",
            HYBRID_SEARCH_SEMANTIC_WEIGHT=-0.1,
            HYBRID_SEARCH_KEYWORD_WEIGHT=0.5,
        )
        raise AssertionError("Should have raised ValidationError")
    except ValidationError as e:
        assert "hybrid_search_semantic_weight" in str(e)
        assert "must be between 0 and 1" in str(e)
        print("✓ Negative semantic weight rejected")


def test_invalid_semantic_weight_too_high():
    """Test that semantic weight > 1 is rejected."""
    from app.config import Settings

    try:
        Settings(
            OPENAI_API_KEY="test-key",
            HYBRID_SEARCH_SEMANTIC_WEIGHT=1.5,
            HYBRID_SEARCH_KEYWORD_WEIGHT=0.5,
        )
        raise AssertionError("Should have raised ValidationError")
    except ValidationError as e:
        assert "hybrid_search_semantic_weight" in str(e)
        assert "must be between 0 and 1" in str(e)
        print("✓ Semantic weight > 1 rejected")


def test_invalid_keyword_weight_negative():
    """Test that negative keyword weight is rejected."""
    from app.config import Settings

    try:
        Settings(
            OPENAI_API_KEY="test-key",
            HYBRID_SEARCH_SEMANTIC_WEIGHT=0.5,
            HYBRID_SEARCH_KEYWORD_WEIGHT=-0.2,
        )
        raise AssertionError("Should have raised ValidationError")
    except ValidationError as e:
        assert "hybrid_search_keyword_weight" in str(e)
        assert "must be between 0 and 1" in str(e)
        print("✓ Negative keyword weight rejected")


def test_invalid_keyword_weight_too_high():
    """Test that keyword weight > 1 is rejected."""
    from app.config import Settings

    try:
        Settings(
            OPENAI_API_KEY="test-key",
            HYBRID_SEARCH_SEMANTIC_WEIGHT=0.5,
            HYBRID_SEARCH_KEYWORD_WEIGHT=2.0,
        )
        raise AssertionError("Should have raised ValidationError")
    except ValidationError as e:
        assert "hybrid_search_keyword_weight" in str(e)
        assert "must be between 0 and 1" in str(e)
        print("✓ Keyword weight > 1 rejected")


def test_boundary_values():
    """Test that boundary values (0 and 1) are accepted."""
    from app.config import Settings

    # All semantic (keyword = 0)
    settings = Settings(
        OPENAI_API_KEY="test-key",
        HYBRID_SEARCH_SEMANTIC_WEIGHT=1.0,
        HYBRID_SEARCH_KEYWORD_WEIGHT=0.0,
    )
    assert settings.hybrid_search_semantic_weight == 1.0
    assert settings.hybrid_search_keyword_weight == 0.0
    print("✓ Boundary values (1.0, 0.0) accepted")

    # All keyword (semantic = 0)
    settings = Settings(
        OPENAI_API_KEY="test-key",
        HYBRID_SEARCH_SEMANTIC_WEIGHT=0.0,
        HYBRID_SEARCH_KEYWORD_WEIGHT=1.0,
    )
    assert settings.hybrid_search_semantic_weight == 0.0
    assert settings.hybrid_search_keyword_weight == 1.0
    print("✓ Boundary values (0.0, 1.0) accepted")


def test_weights_sum_warning():
    """Test that warning is issued when weights don't sum to 1.0."""
    from app.config import Settings
    import warnings

    # Weights that don't sum to 1.0 should warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        settings = Settings(
            OPENAI_API_KEY="test-key",
            HYBRID_SEARCH_SEMANTIC_WEIGHT=0.3,
            HYBRID_SEARCH_KEYWORD_WEIGHT=0.3,  # Sum = 0.6
        )

        # Check that a warning was issued
        assert len(w) == 1
        assert "sum to" in str(w[0].message)
        assert "0.60" in str(w[0].message)
        print("✓ Warning issued for weights summing to 0.6")


if __name__ == "__main__":
    print("\nTesting hybrid search weight validation...\n")

    test_valid_weights()
    test_invalid_semantic_weight_negative()
    test_invalid_semantic_weight_too_high()
    test_invalid_keyword_weight_negative()
    test_invalid_keyword_weight_too_high()
    test_boundary_values()
    test_weights_sum_warning()

    print("\n✅ All validation tests passed!\n")

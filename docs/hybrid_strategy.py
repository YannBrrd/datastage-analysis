"""
Hybrid Analysis Strategy - Maximize insights within $300 budget
"""

# PHASE 1: FREE - Local Analysis (0 tokens)
# ✅ Parse all 9000 jobs
# ✅ Extract fingerprints (stage types, connections, complexity)
# ✅ Structural clustering (hash-based, 0 cost)
# ✅ Semantic clustering (local embeddings, 0 cost)
# ✅ Pattern analysis (migration complexity scoring)
# Result: ~100-200 structural patterns identified

# PHASE 2: SELECTIVE LLM - Claude on Edge Cases Only (~$150)
# Strategy: Use LLM only where it adds unique value
edge_case_scenarios = {
    "cluster_validation": {
        "description": "Validate that semantic clusters are truly similar",
        "approach": "Compare 2-3 representative pairs per cluster",
        "jobs_to_compare": "~50 clusters × 3 pairs = 150 comparisons",
        "cost_estimate": "$1.50 per cluster × 50 = $75"
    },
    
    "ambiguous_migrations": {
        "description": "Analyze jobs with mixed complexity signals",
        "approach": "LLM review of jobs scoring 60-80 complexity",
        "jobs_to_compare": "~100 jobs (most interesting cases)",
        "cost_estimate": "$0.75 per job × 100 = $75"
    },
    
    "pattern_recommendations": {
        "description": "Generate migration templates for top 10 patterns",
        "approach": "Deep analysis of most common job architectures",
        "jobs_to_compare": "10 patterns × 3 examples = 30 jobs",
        "cost_estimate": "$2 per pattern × 10 = $20"
    }
}

# Total: ~280 LLM comparisons for $150-170
# Remaining budget: $120-130 for iterative refinement

# PHASE 3: ITERATIVE - Use Insights to Refine ($50-100)
# Based on Phase 2 findings:
# - Deep-dive on highest-risk migrations
# - Validate automated complexity scores
# - Generate code templates for common patterns

# EXPECTED OUTCOMES with $300 budget:
outcomes = {
    "coverage": "100% of jobs analyzed structurally, 3% with LLM review",
    "patterns_identified": "100-200 distinct patterns",
    "migration_templates": "10-15 reusable PySpark templates",
    "risk_assessment": "High-confidence scoring for all 9000 jobs",
    "effort_estimation": "Validated estimates with LLM spot-checks",
    "confidence": "85-90% (vs 95% with full LLM comparison)"
}

print("HYBRID STRATEGY SUMMARY")
print("="*60)
print(f"Budget: $300")
print(f"Jobs Analyzed Locally: 9,000 (100%)")
print(f"Jobs with LLM Review: ~280 (3% - strategic selection)")
print(f"Expected Patterns: 100-200")
print(f"Migration Templates: 10-15")
print(f"Confidence Level: 85-90%")
print("="*60)
print("\nKEY INSIGHT: LLM adds most value on:")
print("  1. Cluster validation (are groups really similar?)")
print("  2. Ambiguous cases (complexity scoring edge cases)")
print("  3. Template generation (convert patterns to PySpark)")
print("\nLocal analysis handles:")
print("  ✓ Structural similarity (hash-based)")
print("  ✓ Semantic similarity (embeddings)")
print("  ✓ Pattern extraction (stage analysis)")
print("  ✓ Complexity scoring (rule-based)")

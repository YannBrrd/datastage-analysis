"""
Token Budget Optimizer - Calculate and minimize Claude API costs
"""

import logging
from typing import Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TokenBudget:
    """Token usage breakdown and cost estimation."""
    input_tokens: int
    cached_tokens: int  # Tokens read from cache (90% cheaper)
    output_tokens: int
    total_cost_usd: float
    comparisons_count: int
    
    def savings_vs_naive(self, naive_cost: float) -> float:
        """Calculate savings percentage vs naive approach."""
        return ((naive_cost - self.total_cost_usd) / naive_cost) * 100


class TokenOptimizer:
    """
    Calculate optimal strategy for 9000 jobs to minimize Claude API costs.
    
    Pricing (Claude 3.5 Sonnet as of 2024):
    - Input: $3.00 per million tokens
    - Cached Input: $0.30 per million tokens (90% cheaper!)
    - Output: $15.00 per million tokens
    """
    
    # Pricing constants (per million tokens)
    INPUT_COST = 3.00
    CACHED_INPUT_COST = 0.30
    OUTPUT_COST = 15.00
    
    def estimate_full_comparison(self, n_jobs: int, representatives_pct: float = 0.10) -> Dict:
        """
        Estimate token usage for full dataset comparison.
        
        Strategy:
        1. Cluster 9000 jobs → select 10% representatives (~900 jobs)
        2. Compare representatives only (900 choose 2 = ~400K pairs)
        3. Use smart batching: 12 pairs per API call
        4. Use prompt caching: 90% of input tokens cached after first batch
        
        Args:
            n_jobs: Total number of jobs (e.g., 9000)
            representatives_pct: Percentage to compare (default 10%)
            
        Returns:
            Dict with detailed cost breakdown
        """
        n_representatives = int(n_jobs * representatives_pct)
        n_comparisons = (n_representatives * (n_representatives - 1)) // 2
        
        # Token estimates per comparison
        tokens_per_job_summary = 500  # Compressed job representation
        system_prompt_tokens = 1200  # Cached system context
        output_per_comparison = 300  # Analysis output
        
        # Batching: 12 comparisons per API call
        batch_size = 12
        n_batches = (n_comparisons + batch_size - 1) // batch_size
        
        # First batch: write system prompt to cache
        first_batch_input = system_prompt_tokens + (tokens_per_job_summary * 2 * batch_size)
        cache_write_tokens = system_prompt_tokens
        
        # Subsequent batches: read from cache
        subsequent_batches = n_batches - 1
        cached_reads = system_prompt_tokens * subsequent_batches
        fresh_input_per_batch = tokens_per_job_summary * 2 * batch_size
        subsequent_input = fresh_input_per_batch * subsequent_batches
        
        # Total tokens
        total_input_fresh = first_batch_input + subsequent_input
        total_input_cached = cached_reads
        total_output = output_per_comparison * n_comparisons
        
        # Calculate costs
        cost_fresh_input = (total_input_fresh / 1_000_000) * self.INPUT_COST
        cost_cached_input = (total_input_cached / 1_000_000) * self.CACHED_INPUT_COST
        cost_output = (total_output / 1_000_000) * self.OUTPUT_COST
        total_cost = cost_fresh_input + cost_cached_input + cost_output
        
        # Naive approach (no optimization)
        naive_input = (system_prompt_tokens + tokens_per_job_summary * 2) * n_comparisons
        naive_output = total_output
        naive_cost = ((naive_input / 1_000_000) * self.INPUT_COST + 
                     (naive_output / 1_000_000) * self.OUTPUT_COST)
        
        savings = ((naive_cost - total_cost) / naive_cost) * 100
        
        result = {
            "dataset": {
                "total_jobs": n_jobs,
                "representatives": n_representatives,
                "comparisons": n_comparisons,
                "api_calls": n_batches
            },
            "tokens": {
                "input_fresh": total_input_fresh,
                "input_cached": total_input_cached,
                "cache_write": cache_write_tokens,
                "output": total_output,
                "total": total_input_fresh + total_input_cached + total_output
            },
            "costs_usd": {
                "fresh_input": round(cost_fresh_input, 2),
                "cached_input": round(cost_cached_input, 2),
                "output": round(cost_output, 2),
                "total": round(total_cost, 2),
                "naive_approach": round(naive_cost, 2),
                "savings_pct": round(savings, 1)
            },
            "efficiency": {
                "cost_per_comparison": round(total_cost / n_comparisons, 6),
                "cost_per_job": round(total_cost / n_representatives, 4),
                "comparisons_per_dollar": int(n_comparisons / total_cost) if total_cost > 0 else 0
            }
        }
        
        logger.info(f"💰 Estimated cost for {n_jobs} jobs: ${total_cost:.2f} "
                   f"(vs ${naive_cost:.2f} naive = {savings:.1f}% savings)")
        
        return result
    
    def recommend_strategy(self, n_jobs: int, budget_usd: float = 300) -> Dict:
        """
        Recommend optimal strategy given budget constraints.
        
        Args:
            n_jobs: Total number of jobs
            budget_usd: Available budget in USD
            
        Returns:
            Dict with recommended parameters
        """
        # Try different representative percentages
        strategies = []
        for pct in [0.05, 0.10, 0.15, 0.20, 0.25]:
            estimate = self.estimate_full_comparison(n_jobs, pct)
            if estimate['costs_usd']['total'] <= budget_usd:
                strategies.append({
                    "representatives_pct": pct,
                    "representatives_count": estimate['dataset']['representatives'],
                    "comparisons": estimate['dataset']['comparisons'],
                    "cost": estimate['costs_usd']['total'],
                    "budget_used_pct": (estimate['costs_usd']['total'] / budget_usd) * 100
                })
        
        if not strategies:
            logger.warning(f"⚠️ Budget ${budget_usd} insufficient even for 5% representatives")
            return {"error": "Budget too low", "minimum_needed": 
                   self.estimate_full_comparison(n_jobs, 0.05)['costs_usd']['total']}
        
        # Recommend highest coverage within budget
        best = max(strategies, key=lambda s: s['representatives_pct'])
        
        logger.info(f"✅ Recommended: {best['representatives_pct']*100}% representatives "
                   f"({best['representatives_count']} jobs, {best['comparisons']} comparisons) "
                   f"for ${best['cost']:.2f} ({best['budget_used_pct']:.1f}% of budget)")
        
        return {
            "recommended": best,
            "all_options": strategies,
            "budget_usd": budget_usd
        }
    
    def print_comparison_table(self, n_jobs: int):
        """Print visual comparison of strategies."""
        print(f"\n{'='*80}")
        print(f"TOKEN OPTIMIZATION ANALYSIS - {n_jobs} DataStage Jobs")
        print(f"{'='*80}\n")
        
        print(f"{'Strategy':<25} {'Reps':<8} {'Comps':<12} {'Cost':<10} {'Savings':<10}")
        print(f"{'-'*80}")
        
        for pct in [0.05, 0.10, 0.15, 0.20, 0.25]:
            est = self.estimate_full_comparison(n_jobs, pct)
            strategy = f"{pct*100}% representatives"
            reps = est['dataset']['representatives']
            comps = f"{est['dataset']['comparisons']:,}"
            cost = f"${est['costs_usd']['total']:.2f}"
            savings = f"{est['costs_usd']['savings_pct']:.1f}%"
            print(f"{strategy:<25} {reps:<8} {comps:<12} {cost:<10} {savings:<10}")
        
        print(f"{'-'*80}\n")
        
        # Detailed breakdown for recommended strategy (10%)
        detailed = self.estimate_full_comparison(n_jobs, 0.10)
        print("RECOMMENDED: 10% Representatives Strategy")
        print(f"  • Jobs to compare: {detailed['dataset']['representatives']}")
        print(f"  • Total comparisons: {detailed['dataset']['comparisons']:,}")
        print(f"  • API calls needed: {detailed['dataset']['api_calls']:,}")
        print(f"  • Fresh input tokens: {detailed['tokens']['input_fresh']:,}")
        print(f"  • Cached input tokens: {detailed['tokens']['input_cached']:,} (90% cheaper!)")
        print(f"  • Output tokens: {detailed['tokens']['output']:,}")
        print(f"  • Total cost: ${detailed['costs_usd']['total']:.2f}")
        print(f"  • Cost per comparison: ${detailed['efficiency']['cost_per_comparison']:.6f}")
        print(f"  • Savings vs naive: {detailed['costs_usd']['savings_pct']:.1f}%")
        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    optimizer = TokenOptimizer()
    
    # For user's 9000 jobs dataset
    optimizer.print_comparison_table(9000)
    
    # Check what's possible with $300 budget
    recommendation = optimizer.recommend_strategy(9000, budget_usd=300)
    print(f"\n💡 With $300 budget, you can process "
          f"{recommendation['recommended']['representatives_count']} jobs "
          f"({recommendation['recommended']['comparisons']:,} comparisons)")

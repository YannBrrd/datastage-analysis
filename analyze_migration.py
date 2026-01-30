#!/usr/bin/env python3
"""
DataStage Migration Analyzer

Analyzes DataStage DSX files and classifies them for AWS Glue migration.
Outputs a detailed report with AUTO/SEMI-AUTO/MANUAL classification.

Usage:
    python analyze_migration.py /path/to/dsx/files
    python analyze_migration.py /path/to/dsx/files --output report.csv
    python analyze_migration.py /path/to/dsx/files --format json
"""

import argparse
import sys
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datastage_analysis.parsers.dsx_parser import DSXParser
from datastage_analysis.analysis.pattern_analyzer import PatternAnalyzer
from datastage_analysis.analysis.commonality_detector import CommonalityDetector
from datastage_analysis.prediction.migration_predictor import (
    MigrationPredictor,
    MigrationPrediction,
    MigrationCategory,
    MigrationRisk,
    MigrationPriorityRanker,
)
from datastage_analysis.config import get_config


class MigrationAnalyzer:
    """Analyzes DataStage jobs for AWS Glue migration."""

    def __init__(self, verbose: bool = False, debug: bool = False):
        self.parser = DSXParser()
        self.pattern_analyzer = PatternAnalyzer()
        self.predictor = MigrationPredictor()
        self.commonality_detector = CommonalityDetector()
        self.verbose = verbose
        self.debug = debug

        if debug:
            import logging
            logging.basicConfig(level=logging.DEBUG)
            logging.getLogger('datastage_analysis').setLevel(logging.DEBUG)

    def _parse_file(self, dsx_file: Path) -> Tuple[Optional[Any], Optional[Dict]]:
        """Parse a single DSX file. Thread-safe worker function."""
        try:
            parser = DSXParser()  # Create new parser instance per thread
            job = parser._parse_single_job(dsx_file)
            if job is None:
                return None, {"file": str(dsx_file), "error": "Failed to parse file"}
            return job, None
        except Exception as e:
            return None, {"file": str(dsx_file), "error": str(e)}

    def analyze_directory(self, directory: str) -> Dict[str, Any]:
        """
        Analyze all DSX files in a directory.

        Args:
            directory: Path to directory containing DSX files

        Returns:
            Analysis results with predictions and summary
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Find all DataStage files (.dsx, .dsx.gz, .xml, .xml.gz)
        dsx_files = list(dir_path.glob("**/*.dsx"))
        dsx_gz_files = list(dir_path.glob("**/*.dsx.gz"))
        xml_files = list(dir_path.glob("**/*.xml"))
        xml_gz_files = list(dir_path.glob("**/*.xml.gz"))
        all_files = dsx_files + dsx_gz_files + xml_files + xml_gz_files

        if not all_files:
            print(f"⚠️  No DataStage files found in {directory}")
            return {"error": "No DataStage files found", "predictions": [], "summary": {}}

        # Get config for parallel workers
        config = get_config()
        max_workers = config.get('parser', 'max_workers', default=4)

        print(f"📁 Found {len(dsx_files)} .dsx, {len(dsx_gz_files)} .dsx.gz, {len(xml_files)} .xml, {len(xml_gz_files)} .xml.gz (total: {len(all_files)})")
        print(f"🚀 Parsing with {max_workers} parallel workers")
        print("-" * 60)

        # Parse files in parallel
        predictions: List[MigrationPrediction] = []
        structures: Dict[str, Dict] = {}
        errors: List[Dict] = []
        total_files = len(all_files)
        completed = 0
        lock = threading.Lock()

        def update_progress(filename: str):
            nonlocal completed
            with lock:
                completed += 1
                progress_pct = int((completed / total_files) * 100)
                bar_width = 30
                filled = int(bar_width * completed / total_files)
                bar = "█" * filled + "░" * (bar_width - filled)
                print(f"\r⏳ [{bar}] {progress_pct:3d}% ({completed}/{total_files}) {filename[:30]:<30}", end="", flush=True)

        # Phase 1: Parallel file parsing
        parsed_jobs = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self._parse_file, f): f for f in all_files}

            for future in as_completed(future_to_file):
                dsx_file = future_to_file[future]
                update_progress(dsx_file.name)

                job, error = future.result()
                if error:
                    errors.append(error)
                    if self.debug:
                        print(f"\n  ❌ {error['file']}: {error['error']}")
                elif job:
                    parsed_jobs.append((dsx_file, job))
                    if self.debug:
                        stages = job.structure.get('stages', [])
                        jobs_in_struct = job.structure.get('jobs', [])
                        print(f"\n  📄 {dsx_file.name}: {len(stages)} stages, {len(jobs_in_struct)} sub-jobs")

        # Phase 2: Sequential analysis (fast, needs shared state)
        print(f"\r⏳ Analyzing {len(parsed_jobs)} parsed jobs..." + " " * 40, end="", flush=True)

        for dsx_file, job in parsed_jobs:
            job_name = job.name
            structure = job.structure
            structures[job_name] = structure

            # Handle multi-job DSX files
            jobs_in_file = structure.get('jobs', [])
            if jobs_in_file and len(jobs_in_file) > 1:
                for sub_job in jobs_in_file:
                    sub_name = sub_job.get('name', job_name)
                    sub_structure = {
                        'name': sub_name,
                        'stages': sub_job.get('stages', []),
                        'links': sub_job.get('links', []),
                    }
                    self._analyze_single_job(sub_name, sub_structure, predictions, structures)
            else:
                self._analyze_single_job(job_name, structure, predictions, structures)

        # Clear progress bar and show completion
        print(f"\r✅ Parsed {total_files} files, found {len(predictions)} jobs" + " " * 40)

        # Generate summary
        summary = self._generate_summary(predictions)

        # Rank jobs for migration priority
        ranker = MigrationPriorityRanker()
        ranked_jobs = ranker.rank_jobs(predictions)

        # Analyze commonalities (duplicates, patterns)
        commonality_report = self.commonality_detector.analyze(structures)

        return {
            "predictions": predictions,
            "structures": structures,  # Include structures for generation
            "summary": summary,
            "ranked_jobs": ranked_jobs,
            "commonality": commonality_report,
            "errors": errors,
            "analyzed_at": datetime.now().isoformat(),
        }

    def _analyze_single_job(self, job_name: str, structure: Dict,
                           predictions: List[MigrationPrediction],
                           structures: Dict[str, Dict]):
        """Analyze a single job and add prediction to list."""
        structures[job_name] = structure

        # Create a mock job dict for pattern analyzer
        job_dict = {
            'name': job_name,
            'structure': structure,
        }

        # Analyze pattern
        patterns = self.pattern_analyzer.analyze_jobs([job_dict])
        if patterns:
            pattern = patterns[0]

            # Predict migration outcome
            prediction = self.predictor.predict(pattern, structure)
            predictions.append(prediction)

            if self.verbose:
                self._print_prediction(prediction)

    def _print_prediction(self, pred: MigrationPrediction):
        """Print a single prediction to console."""
        icons = {
            MigrationCategory.AUTO: "🟢",
            MigrationCategory.SEMI_AUTO: "🟡",
            MigrationCategory.MANUAL: "🔴",
        }
        risk_icons = {
            MigrationRisk.LOW: "✅",
            MigrationRisk.MEDIUM: "⚠️",
            MigrationRisk.HIGH: "🔶",
            MigrationRisk.CRITICAL: "🚨",
        }

        icon = icons.get(pred.category, "⚪")
        risk_icon = risk_icons.get(pred.risk_level, "❓")

        print(f"  {icon} {pred.job_name}")
        print(f"     Category: {pred.category.value} | Confidence: {pred.confidence:.0%}")
        print(f"     Success: {pred.success_probability:.0%} | Effort: {pred.estimated_hours:.1f}h")
        print(f"     Risk: {risk_icon} {pred.risk_level.value}")
        if pred.automation_blockers:
            print(f"     Blockers: {', '.join(pred.automation_blockers[:2])}")
        print()

    def _generate_summary(self, predictions: List[MigrationPrediction]) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not predictions:
            return {}

        auto_count = sum(1 for p in predictions if p.category == MigrationCategory.AUTO)
        semi_count = sum(1 for p in predictions if p.category == MigrationCategory.SEMI_AUTO)
        manual_count = sum(1 for p in predictions if p.category == MigrationCategory.MANUAL)

        total = len(predictions)

        return {
            "total_jobs": total,
            "auto": {
                "count": auto_count,
                "percentage": round(auto_count / total * 100, 1) if total > 0 else 0,
            },
            "semi_auto": {
                "count": semi_count,
                "percentage": round(semi_count / total * 100, 1) if total > 0 else 0,
            },
            "manual": {
                "count": manual_count,
                "percentage": round(manual_count / total * 100, 1) if total > 0 else 0,
            },
            "avg_success_probability": round(
                sum(p.success_probability for p in predictions) / total * 100, 1
            ) if total > 0 else 0,
            "total_estimated_hours": round(
                sum(p.estimated_hours for p in predictions), 1
            ),
            "risk_distribution": {
                "low": sum(1 for p in predictions if p.risk_level == MigrationRisk.LOW),
                "medium": sum(1 for p in predictions if p.risk_level == MigrationRisk.MEDIUM),
                "high": sum(1 for p in predictions if p.risk_level == MigrationRisk.HIGH),
                "critical": sum(1 for p in predictions if p.risk_level == MigrationRisk.CRITICAL),
            },
            "high_risk_jobs": [
                p.job_name for p in predictions
                if p.risk_level in (MigrationRisk.HIGH, MigrationRisk.CRITICAL)
            ],
            "unknown_stages": self._collect_unknown_stages(predictions),
        }

    def _collect_unknown_stages(self, predictions: List[MigrationPrediction]) -> Dict[str, Any]:
        """Collect all unknown/unrecognized DataStage stage types."""
        unknown_counts = {}
        jobs_with_unknown = []

        for p in predictions:
            if hasattr(p, 'unknown_stages') and p.unknown_stages:
                jobs_with_unknown.append(p.job_name)
                for stage_type in p.unknown_stages:
                    unknown_counts[stage_type] = unknown_counts.get(stage_type, 0) + 1

        return {
            "types": dict(sorted(unknown_counts.items(), key=lambda x: -x[1])),
            "total_unique": len(unknown_counts),
            "jobs_affected": len(jobs_with_unknown),
            "job_names": jobs_with_unknown[:20],  # Limit to first 20
        }


def print_report(results: Dict[str, Any]):
    """Print formatted report to console."""
    summary = results.get("summary", {})
    predictions = results.get("predictions", [])

    print("\n" + "=" * 60)
    print("📊 MIGRATION ANALYSIS REPORT")
    print("=" * 60)

    if not summary:
        print("No jobs analyzed.")
        return

    # Summary section
    print(f"\n📈 SUMMARY")
    print(f"   Total Jobs Analyzed: {summary['total_jobs']}")
    print()

    # Category breakdown with visual bar
    print("   Migration Categories:")
    categories = [
        ("🟢 AUTO      ", summary['auto']['count'], summary['auto']['percentage']),
        ("🟡 SEMI-AUTO ", summary['semi_auto']['count'], summary['semi_auto']['percentage']),
        ("🔴 MANUAL    ", summary['manual']['count'], summary['manual']['percentage']),
    ]

    for label, count, pct in categories:
        bar_width = int(pct / 2)
        bar = "█" * bar_width + "░" * (50 - bar_width)
        print(f"   {label}: {count:4d} ({pct:5.1f}%) |{bar}|")

    print()
    print(f"   Average Success Probability: {summary['avg_success_probability']}%")
    print(f"   Total Estimated Effort: {summary['total_estimated_hours']} hours")

    # Risk distribution
    print("\n   Risk Distribution:")
    risk_dist = summary.get('risk_distribution', {})
    print(f"   ✅ Low: {risk_dist.get('low', 0)} | ⚠️ Medium: {risk_dist.get('medium', 0)} | "
          f"🔶 High: {risk_dist.get('high', 0)} | 🚨 Critical: {risk_dist.get('critical', 0)}")

    # High risk jobs
    high_risk = summary.get('high_risk_jobs', [])
    if high_risk:
        print(f"\n⚠️  HIGH RISK JOBS ({len(high_risk)}):")
        for job in high_risk[:10]:
            print(f"   - {job}")
        if len(high_risk) > 10:
            print(f"   ... and {len(high_risk) - 10} more")

    # Unknown/Unrecognized DataStage stages
    unknown_info = summary.get('unknown_stages', {})
    unknown_types = unknown_info.get('types', {})
    if unknown_types:
        print(f"\n❓ UNKNOWN DATASTAGE STAGES ({unknown_info.get('total_unique', 0)} types, {unknown_info.get('jobs_affected', 0)} jobs affected):")
        for stage_type, count in list(unknown_types.items())[:15]:
            print(f"   - {stage_type}: {count} occurrence(s)")
        if len(unknown_types) > 15:
            print(f"   ... and {len(unknown_types) - 15} more types")
        print("   💡 Consider adding these to GLUE_COMPLEXITY mapping for better predictions")

    # Top 10 jobs by priority
    ranked = results.get("ranked_jobs", [])
    if ranked:
        print(f"\n🎯 TOP 10 PRIORITY JOBS (easiest to migrate):")
        for job_name, score in ranked[:10]:
            pred = next((p for p in predictions if p.job_name == job_name), None)
            if pred:
                icons = {
                    MigrationCategory.AUTO: "🟢",
                    MigrationCategory.SEMI_AUTO: "🟡",
                    MigrationCategory.MANUAL: "🔴",
                }
                icon = icons.get(pred.category, "⚪")
                print(f"   {icon} {job_name} (score: {score:.3f}, {pred.estimated_hours:.1f}h)")

    # Commonality Analysis
    commonality = results.get("commonality")
    if commonality:
        print(f"\n📋 COMMONALITY ANALYSIS")
        print(f"   Total Jobs: {commonality.total_jobs}")
        print(f"   Unique Patterns: {commonality.unique_patterns}")

        # Exact duplicates
        dup_groups = commonality.exact_duplicate_groups
        if dup_groups:
            dup_job_count = sum(g.count for g in dup_groups)
            print(f"\n   🔁 Exact Duplicates: {dup_job_count} jobs in {len(dup_groups)} groups")
            for group in dup_groups[:5]:
                print(f"      - {group.count} jobs: {', '.join(group.job_names[:3])}{'...' if group.count > 3 else ''}")
            if len(dup_groups) > 5:
                print(f"      ... and {len(dup_groups) - 5} more groups")

        # Similar clusters
        clusters = commonality.similarity_clusters
        if clusters:
            cluster_job_count = sum(c.count for c in clusters)
            print(f"\n   🔗 Similar Jobs (>85%): {cluster_job_count} jobs in {len(clusters)} clusters")
            for cluster in clusters[:5]:
                print(f"      - {cluster.count} jobs ({cluster.pattern_signature})")
            if len(clusters) > 5:
                print(f"      ... and {len(clusters) - 5} more clusters")

        # Pattern families
        families = commonality.pattern_families
        if families:
            print(f"\n   📂 Pattern Families ({len(families)}):")
            for family in families[:8]:
                print(f"      - {family.pattern_name}: {family.count} jobs → {family.migration_template}")
            if len(families) > 8:
                print(f"      ... and {len(families) - 8} more patterns")

        # Effort reduction summary
        print(f"\n   💡 Effective Unique Jobs: {commonality.effective_unique_jobs} (vs {commonality.total_jobs} total)")
        print(f"   📉 Estimated Effort Reduction: {commonality.effort_reduction_percent:.1f}%")

    # Errors
    errors = results.get("errors", [])
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for err in errors[:5]:
            print(f"   - {err['file']}: {err['error']}")

    print("\n" + "=" * 60)


def export_csv(results: Dict[str, Any], output_path: str):
    """Export results to CSV."""
    predictions = results.get("predictions", [])
    commonality = results.get("commonality")

    # Build mappings for group identifiers
    job_to_duplicate_group: Dict[str, str] = {}
    job_to_similarity_cluster: Dict[str, str] = {}
    job_to_pattern_family: Dict[str, str] = {}

    if commonality:
        # Map jobs to duplicate groups (use group index as ID)
        for idx, group in enumerate(commonality.exact_duplicate_groups, 1):
            group_id = f"DUP_{idx}"
            for job_name in group.job_names:
                job_to_duplicate_group[job_name] = group_id

        # Map jobs to similarity clusters
        for cluster in commonality.similarity_clusters:
            cluster_id = f"SIM_{cluster.cluster_id}"
            for job_name in cluster.job_names:
                job_to_similarity_cluster[job_name] = cluster_id

        # Map jobs to pattern families
        for family in commonality.pattern_families:
            for job_name in family.job_names:
                job_to_pattern_family[job_name] = family.pattern_name

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'job_name',
            'category',
            'confidence',
            'success_probability',
            'estimated_hours',
            'risk_level',
            'duplicate_group',
            'similarity_cluster',
            'pattern_family',
            'risk_factors',
            'automation_blockers',
            'glue_features_needed',
            'recommendations',
            'unknown_stages'
        ])

        # Data
        for pred in predictions:
            unknown = getattr(pred, 'unknown_stages', []) or []
            writer.writerow([
                pred.job_name,
                pred.category.value,
                f"{pred.confidence:.2f}",
                f"{pred.success_probability:.2f}",
                f"{pred.estimated_hours:.1f}",
                pred.risk_level.value,
                job_to_duplicate_group.get(pred.job_name, ""),
                job_to_similarity_cluster.get(pred.job_name, ""),
                job_to_pattern_family.get(pred.job_name, ""),
                "; ".join(pred.risk_factors),
                "; ".join(pred.automation_blockers),
                "; ".join(pred.glue_features_needed),
                "; ".join(pred.recommendations),
                "; ".join(unknown),
            ])

    print(f"📄 CSV exported to: {output_path}")


def export_json(results: Dict[str, Any], output_path: str):
    """Export results to JSON."""
    # Convert commonality report to dict if present
    commonality = results.get("commonality")
    commonality_dict = commonality.to_dict() if commonality else {}

    # Convert predictions to dicts
    output = {
        "summary": results.get("summary", {}),
        "predictions": [p.to_dict() for p in results.get("predictions", [])],
        "ranked_jobs": results.get("ranked_jobs", []),
        "commonality": commonality_dict,
        "errors": results.get("errors", []),
        "analyzed_at": results.get("analyzed_at", ""),
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"📄 JSON exported to: {output_path}")


def _build_cluster_info(commonality) -> Optional[Dict[str, Dict]]:
    """
    Build cluster info dict from commonality report.

    Maps each job to its cluster identifiers for batch processing.
    """
    if not commonality:
        return None

    cluster_info: Dict[str, Dict] = {}

    # Map jobs to duplicate groups
    for idx, group in enumerate(commonality.exact_duplicate_groups, 1):
        group_id = f"DUP_{idx}"
        for job_name in group.job_names:
            if job_name not in cluster_info:
                cluster_info[job_name] = {}
            cluster_info[job_name]['duplicate_group'] = group_id

    # Map jobs to similarity clusters
    for cluster in commonality.similarity_clusters:
        cluster_id = f"SIM_{cluster.cluster_id}"
        for job_name in cluster.job_names:
            if job_name not in cluster_info:
                cluster_info[job_name] = {}
            cluster_info[job_name]['similarity_cluster'] = cluster_id

    # Map jobs to pattern families
    for family in commonality.pattern_families:
        for job_name in family.job_names:
            if job_name not in cluster_info:
                cluster_info[job_name] = {}
            cluster_info[job_name]['pattern_family'] = family.pattern_name

    return cluster_info if cluster_info else None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze DataStage jobs for AWS Glue migration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./data                    Analyze all DSX files in ./data
  %(prog)s ./data -o report.csv      Export to CSV
  %(prog)s ./data -f json -o out.json  Export to JSON
  %(prog)s ./data -v                 Verbose output
        """
    )

    parser.add_argument(
        "directory",
        help="Directory containing DSX files to analyze"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file path (CSV or JSON based on -f)"
    )

    parser.add_argument(
        "-f", "--format",
        choices=["console", "csv", "json"],
        default="console",
        help="Output format (default: console)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output for each job"
    )

    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug logging to diagnose parsing issues"
    )

    # Generation arguments
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate AWS Glue code after analysis"
    )

    parser.add_argument(
        "--generate-only",
        type=str,
        metavar="JOBS",
        help="Generate code for specific jobs (comma-separated or 'all')"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./generated",
        help="Output directory for generated code (default: ./generated)"
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM, use rule-based generation only"
    )

    parser.add_argument(
        "--llm-provider",
        choices=["anthropic", "azure", "azure_foundry", "aws", "gcp", "openrouter"],
        help="Override LLM provider from config"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate costs and preview batches without generating code"
    )

    parser.add_argument(
        "--budget",
        type=float,
        metavar="DOLLARS",
        help="Set budget limit for LLM costs (used with --dry-run for warnings)"
    )

    parser.add_argument(
        "--report",
        type=str,
        metavar="PATH",
        help="Generate HTML migration report at specified path"
    )

    args = parser.parse_args()

    # Run analysis
    print(f"\n🔍 DataStage to AWS Glue Migration Analyzer")
    print(f"   Analyzing: {args.directory}")
    print()

    analyzer = MigrationAnalyzer(verbose=args.verbose, debug=args.debug)

    try:
        results = analyzer.analyze_directory(args.directory)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    # Output results
    if args.format == "console" or not args.output:
        print_report(results)

    if args.output:
        if args.format == "csv" or args.output.endswith(".csv"):
            export_csv(results, args.output)
        elif args.format == "json" or args.output.endswith(".json"):
            export_json(results, args.output)
        else:
            export_csv(results, args.output)

    # Summary stats for exit
    summary = results.get("summary", {})
    if summary:
        auto_pct = summary.get("auto", {}).get("percentage", 0)
        print(f"\n✨ {auto_pct}% of jobs can be automatically migrated to AWS Glue")

    # HTML Report generation
    if args.report:
        try:
            from datastage_analysis.reporting import generate_html_report
            report_path = generate_html_report(results, output_path=args.report)
            print(f"\n📊 HTML report generated: {report_path}")
        except Exception as e:
            print(f"❌ Failed to generate HTML report: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()

    # Code generation (or dry-run)
    if args.generate or args.generate_only or args.dry_run:
        # Build cluster info from commonality report
        cluster_info = _build_cluster_info(results.get("commonality"))

        # Get structures from analysis results
        structures = results.get("structures", {})

        # Determine which jobs to generate
        predictions_to_use = results.get("predictions", [])
        jobs_filter = None
        if args.generate_only and args.generate_only.lower() != 'all':
            jobs_filter = [j.strip() for j in args.generate_only.split(',')]
            predictions_to_use = [p for p in predictions_to_use if p.job_name in jobs_filter]

        # Dry-run mode
        if args.dry_run:
            print("\n" + "=" * 60)
            print("🔍 DRY-RUN: COST ESTIMATION")
            print("=" * 60)

            try:
                from datastage_analysis.generation.dry_run import DryRunEstimator, format_dry_run_report

                provider = args.llm_provider or 'anthropic'
                estimator = DryRunEstimator(
                    provider=provider,
                    budget_limit=args.budget
                )

                dry_result = estimator.estimate(
                    predictions=predictions_to_use,
                    cluster_info=cluster_info,
                    use_batching=not args.no_llm
                )

                print(format_dry_run_report(dry_result, provider))

                # Export dry-run results if output specified
                if args.output and args.output.endswith('.json'):
                    import json
                    dry_output = {
                        'dry_run': True,
                        'estimate': dry_result.to_dict(),
                        'analysis_summary': results.get("summary", {}),
                    }
                    with open(args.output.replace('.json', '_dry_run.json'), 'w') as f:
                        json.dump(dry_output, f, indent=2)
                    print(f"\n📄 Dry-run results exported to: {args.output.replace('.json', '_dry_run.json')}")

            except Exception as e:
                print(f"❌ Dry-run failed: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()

            return  # Exit after dry-run

        print("\n" + "=" * 60)
        print("🔧 CODE GENERATION")
        print("=" * 60)

        try:
            from datastage_analysis.generation import MigrationGenerator

            if jobs_filter:
                print(f"Generating code for {len(jobs_filter)} specified jobs")

            # Initialize generator with batch processing enabled
            use_llm = not args.no_llm
            generator = MigrationGenerator(use_llm=use_llm, use_batch_processing=use_llm)

            # Override LLM provider if specified
            if args.llm_provider and use_llm:
                print(f"Using LLM provider: {args.llm_provider}")

            if cluster_info and use_llm:
                n_clustered = sum(1 for v in cluster_info.values()
                                 if v.get('similarity_cluster') or v.get('duplicate_group'))
                print(f"📦 Batch optimization: {n_clustered} jobs in clusters")

            # Run generation
            gen_results = generator.generate(
                predictions=results.get("predictions", []),
                structures=structures,
                jobs_filter=jobs_filter,
                output_dir=args.output_dir,
                cluster_info=cluster_info,
            )

            # Print generation summary
            gen_summary = gen_results.get_summary()
            print(f"\n📊 Generation Summary:")
            print(f"   Total: {gen_summary['total_jobs']}")
            print(f"   Success: {gen_summary['successful_jobs']} ({gen_summary['success_rate']}%)")
            print(f"   Failed: {gen_summary['failed_jobs']}")
            if gen_summary['total_llm_tokens'] > 0:
                print(f"   LLM Tokens Used: {gen_summary['total_llm_tokens']:,}")

            # Batch optimization stats
            batch_stats = gen_summary.get('batch_optimization', {})
            if batch_stats.get('batches_used', 0) > 0:
                print(f"\n   📦 Batch Optimization:")
                print(f"      Batches used: {batch_stats['batches_used']}")
                print(f"      Jobs from batches: {batch_stats['jobs_from_batches']}")
                print(f"      LLM calls saved: {batch_stats['llm_calls_saved']}")

            print(f"\n📁 Output directory: {args.output_dir}")

        except ImportError as e:
            print(f"❌ Generation module not available: {e}")
            print("   Install dependencies: uv pip install jinja2 anthropic")
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()

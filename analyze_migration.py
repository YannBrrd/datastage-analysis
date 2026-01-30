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
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datastage_analysis.parsers.dsx_parser import DSXParser
from datastage_analysis.analysis.pattern_analyzer import PatternAnalyzer
from datastage_analysis.prediction.migration_predictor import (
    MigrationPredictor,
    MigrationPrediction,
    MigrationCategory,
    MigrationRisk,
    MigrationPriorityRanker,
)


class MigrationAnalyzer:
    """Analyzes DataStage jobs for AWS Glue migration."""

    def __init__(self, verbose: bool = False):
        self.parser = DSXParser()
        self.pattern_analyzer = PatternAnalyzer()
        self.predictor = MigrationPredictor()
        self.verbose = verbose

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

        # Find all DSX files (both .dsx and .dsx.gz)
        dsx_files = list(dir_path.glob("**/*.dsx"))
        dsx_gz_files = list(dir_path.glob("**/*.dsx.gz"))
        all_files = dsx_files + dsx_gz_files

        if not all_files:
            print(f"⚠️  No DSX files found in {directory}")
            return {"error": "No DSX files found", "predictions": [], "summary": {}}

        print(f"📁 Found {len(dsx_files)} .dsx and {len(dsx_gz_files)} .dsx.gz files (total: {len(all_files)})")
        print("-" * 60)

        # Parse and analyze each file
        predictions: List[MigrationPrediction] = []
        structures: Dict[str, Dict] = {}
        errors: List[Dict] = []

        for i, dsx_file in enumerate(all_files, 1):
            if self.verbose:
                print(f"[{i}/{len(all_files)}] Analyzing: {dsx_file.name}")

            try:
                # Parse DSX file using the parser's internal method
                job = self.parser._parse_single_job(dsx_file)

                if job is None:
                    errors.append({
                        "file": str(dsx_file),
                        "error": "Failed to parse file"
                    })
                    continue

                job_name = job.name
                structure = job.structure
                structures[job_name] = structure

                # Handle multi-job DSX files
                jobs_in_file = structure.get('jobs', [])
                if jobs_in_file and len(jobs_in_file) > 1:
                    # Process each job separately
                    for sub_job in jobs_in_file:
                        sub_name = sub_job.get('name', job_name)
                        sub_structure = {
                            'name': sub_name,
                            'stages': sub_job.get('stages', []),
                            'links': sub_job.get('links', []),
                        }
                        self._analyze_single_job(sub_name, sub_structure, predictions, structures)
                else:
                    # Single job
                    self._analyze_single_job(job_name, structure, predictions, structures)

            except Exception as e:
                errors.append({
                    "file": str(dsx_file),
                    "error": str(e)
                })
                if self.verbose:
                    print(f"  ❌ Error: {e}")

        # Generate summary
        summary = self._generate_summary(predictions)

        # Rank jobs for migration priority
        ranker = MigrationPriorityRanker()
        ranked_jobs = ranker.rank_jobs(predictions)
        waves = ranker.create_migration_waves(ranked_jobs, wave_size=50)

        return {
            "predictions": predictions,
            "summary": summary,
            "ranked_jobs": ranked_jobs,
            "migration_waves": waves,
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

    # Migration waves
    waves = results.get("migration_waves", [])
    if waves:
        print(f"\n📦 MIGRATION WAVES ({len(waves)} waves):")
        for i, wave in enumerate(waves[:5], 1):
            print(f"   Wave {i}: {len(wave)} jobs")
        if len(waves) > 5:
            print(f"   ... and {len(waves) - 5} more waves")

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
            'risk_factors',
            'automation_blockers',
            'glue_features_needed',
            'recommendations'
        ])

        # Data
        for pred in predictions:
            writer.writerow([
                pred.job_name,
                pred.category.value,
                f"{pred.confidence:.2f}",
                f"{pred.success_probability:.2f}",
                f"{pred.estimated_hours:.1f}",
                pred.risk_level.value,
                "; ".join(pred.risk_factors),
                "; ".join(pred.automation_blockers),
                "; ".join(pred.glue_features_needed),
                "; ".join(pred.recommendations),
            ])

    print(f"📄 CSV exported to: {output_path}")


def export_json(results: Dict[str, Any], output_path: str):
    """Export results to JSON."""
    # Convert predictions to dicts
    output = {
        "summary": results.get("summary", {}),
        "predictions": [p.to_dict() for p in results.get("predictions", [])],
        "ranked_jobs": results.get("ranked_jobs", []),
        "migration_waves": results.get("migration_waves", []),
        "errors": results.get("errors", []),
        "analyzed_at": results.get("analyzed_at", ""),
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"📄 JSON exported to: {output_path}")


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

    args = parser.parse_args()

    # Run analysis
    print(f"\n🔍 DataStage to AWS Glue Migration Analyzer")
    print(f"   Analyzing: {args.directory}")
    print()

    analyzer = MigrationAnalyzer(verbose=args.verbose)

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


if __name__ == "__main__":
    main()

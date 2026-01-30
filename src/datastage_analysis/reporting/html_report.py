"""
HTML Report Generator

Generates comprehensive HTML migration reports with charts and detailed analysis.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path


def generate_html_report(
    analysis_results: Dict[str, Any],
    generation_results: Optional[Dict[str, Any]] = None,
    output_path: str = "migration_report.html"
) -> str:
    """
    Generate comprehensive HTML migration report.

    Args:
        analysis_results: Results from migration analysis
        generation_results: Optional results from code generation
        output_path: Path to write HTML file

    Returns:
        Path to generated HTML file
    """
    summary = analysis_results.get("summary", {})
    predictions = analysis_results.get("predictions", [])
    commonality = analysis_results.get("commonality")
    errors = analysis_results.get("errors", [])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataStage to AWS Glue Migration Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary: #232f3e;
            --secondary: #ff9900;
            --success: #1a8754;
            --warning: #ffc107;
            --danger: #dc3545;
            --light: #f8f9fa;
            --dark: #212529;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: var(--dark);
            background: var(--light);
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: var(--primary);
            color: white;
            padding: 30px 0;
            margin-bottom: 30px;
        }}

        header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}

        header .subtitle {{
            color: var(--secondary);
            font-size: 1.2rem;
        }}

        .card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 24px;
            margin-bottom: 24px;
        }}

        .card h2 {{
            color: var(--primary);
            border-bottom: 2px solid var(--secondary);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}

        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .metric {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .metric .value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--primary);
        }}

        .metric .label {{
            color: #666;
            font-size: 0.9rem;
            text-transform: uppercase;
        }}

        .metric.success .value {{ color: var(--success); }}
        .metric.warning .value {{ color: var(--warning); }}
        .metric.danger .value {{ color: var(--danger); }}

        .progress-bar {{
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            display: flex;
        }}

        .progress-bar .segment {{
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 0.85rem;
        }}

        .progress-bar .auto {{ background: var(--success); }}
        .progress-bar .semi {{ background: var(--warning); }}
        .progress-bar .manual {{ background: var(--danger); }}

        .chart-container {{
            position: relative;
            height: 300px;
            margin: 20px 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}

        th {{
            background: var(--primary);
            color: white;
        }}

        tr:hover {{
            background: #f5f5f5;
        }}

        .badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
        }}

        .badge-auto {{ background: #d4edda; color: #155724; }}
        .badge-semi {{ background: #fff3cd; color: #856404; }}
        .badge-manual {{ background: #f8d7da; color: #721c24; }}
        .badge-low {{ background: #d4edda; color: #155724; }}
        .badge-medium {{ background: #fff3cd; color: #856404; }}
        .badge-high {{ background: #f8d7da; color: #721c24; }}
        .badge-critical {{ background: #721c24; color: white; }}

        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }}

        @media (max-width: 768px) {{
            .two-col {{
                grid-template-columns: 1fr;
            }}
        }}

        .cluster-list {{
            max-height: 400px;
            overflow-y: auto;
        }}

        .cluster-item {{
            padding: 12px;
            border-left: 4px solid var(--secondary);
            background: #f8f9fa;
            margin-bottom: 8px;
            border-radius: 0 4px 4px 0;
        }}

        .cluster-item .name {{
            font-weight: bold;
            color: var(--primary);
        }}

        .cluster-item .count {{
            color: #666;
            font-size: 0.9rem;
        }}

        footer {{
            text-align: center;
            padding: 30px;
            color: #666;
            border-top: 1px solid #dee2e6;
            margin-top: 40px;
        }}

        .recommendation {{
            padding: 12px 16px;
            border-radius: 4px;
            margin: 8px 0;
        }}

        .recommendation.tip {{ background: #d1ecf1; border-left: 4px solid #0c5460; }}
        .recommendation.warning {{ background: #fff3cd; border-left: 4px solid #856404; }}
        .recommendation.success {{ background: #d4edda; border-left: 4px solid #155724; }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>DataStage to AWS Glue Migration Report</h1>
            <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </header>

    <div class="container">
        <!-- Executive Summary -->
        <div class="metrics">
            <div class="metric">
                <div class="value">{summary.get('total_jobs', 0)}</div>
                <div class="label">Total Jobs</div>
            </div>
            <div class="metric success">
                <div class="value">{summary.get('auto', {}).get('percentage', 0):.0f}%</div>
                <div class="label">Fully Automatable</div>
            </div>
            <div class="metric">
                <div class="value">{summary.get('total_estimated_hours', 0):.0f}h</div>
                <div class="label">Estimated Effort</div>
            </div>
            <div class="metric">
                <div class="value">{summary.get('avg_success_probability', 0):.0f}%</div>
                <div class="label">Success Probability</div>
            </div>
        </div>

        <!-- Migration Categories -->
        <div class="card">
            <h2>Migration Categories</h2>
            <div class="progress-bar">
                <div class="segment auto" style="width: {summary.get('auto', {}).get('percentage', 0)}%">
                    AUTO {summary.get('auto', {}).get('count', 0)}
                </div>
                <div class="segment semi" style="width: {summary.get('semi_auto', {}).get('percentage', 0)}%">
                    SEMI {summary.get('semi_auto', {}).get('count', 0)}
                </div>
                <div class="segment manual" style="width: {summary.get('manual', {}).get('percentage', 0)}%">
                    MANUAL {summary.get('manual', {}).get('count', 0)}
                </div>
            </div>
            <div class="two-col" style="margin-top: 20px;">
                <div class="chart-container">
                    <canvas id="categoryChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="riskChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Batch Optimization -->
        {_generate_commonality_section(commonality)}

        <!-- Risk Analysis -->
        <div class="card">
            <h2>Risk Analysis</h2>
            {_generate_risk_table(predictions)}
        </div>

        <!-- Job Details -->
        <div class="card">
            <h2>Job Details</h2>
            {_generate_jobs_table(predictions)}
        </div>

        <!-- Recommendations -->
        <div class="card">
            <h2>Recommendations</h2>
            {_generate_recommendations(summary, commonality, predictions)}
        </div>

        <!-- Errors -->
        {_generate_errors_section(errors)}
    </div>

    <footer>
        <p>Generated by DataStage Migration Analyzer</p>
        <p>AWS Glue Migration Toolkit</p>
    </footer>

    <script>
        // Category Pie Chart
        new Chart(document.getElementById('categoryChart'), {{
            type: 'doughnut',
            data: {{
                labels: ['AUTO', 'SEMI-AUTO', 'MANUAL'],
                datasets: [{{
                    data: [{summary.get('auto', {}).get('count', 0)},
                           {summary.get('semi_auto', {}).get('count', 0)},
                           {summary.get('manual', {}).get('count', 0)}],
                    backgroundColor: ['#1a8754', '#ffc107', '#dc3545']
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Migration Categories'
                    }}
                }}
            }}
        }});

        // Risk Bar Chart
        const riskDist = {json.dumps(summary.get('risk_distribution', {}))};
        new Chart(document.getElementById('riskChart'), {{
            type: 'bar',
            data: {{
                labels: ['Low', 'Medium', 'High', 'Critical'],
                datasets: [{{
                    label: 'Jobs',
                    data: [riskDist.low || 0, riskDist.medium || 0, riskDist.high || 0, riskDist.critical || 0],
                    backgroundColor: ['#1a8754', '#ffc107', '#fd7e14', '#dc3545']
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Risk Distribution'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    # Write to file
    output_path = Path(output_path)
    output_path.write_text(html)

    return str(output_path)


def _generate_commonality_section(commonality) -> str:
    """Generate HTML for commonality/batch optimization section."""
    if not commonality:
        return ""

    dup_groups = commonality.exact_duplicate_groups if hasattr(commonality, 'exact_duplicate_groups') else []
    sim_clusters = commonality.similarity_clusters if hasattr(commonality, 'similarity_clusters') else []
    families = commonality.pattern_families if hasattr(commonality, 'pattern_families') else []

    dup_count = sum(g.count for g in dup_groups) if dup_groups else 0
    sim_count = sum(c.count for c in sim_clusters) if sim_clusters else 0

    effort_reduction = commonality.effort_reduction_percent if hasattr(commonality, 'effort_reduction_percent') else 0

    clusters_html = ""
    if sim_clusters:
        for cluster in sim_clusters[:10]:
            clusters_html += f"""
            <div class="cluster-item">
                <span class="name">Cluster {cluster.cluster_id}</span>
                <span class="count"> - {cluster.count} jobs ({cluster.pattern_signature})</span>
            </div>"""

    return f"""
    <div class="card">
        <h2>Batch Optimization Potential</h2>
        <div class="metrics">
            <div class="metric success">
                <div class="value">{effort_reduction:.0f}%</div>
                <div class="label">Effort Reduction</div>
            </div>
            <div class="metric">
                <div class="value">{len(dup_groups)}</div>
                <div class="label">Duplicate Groups</div>
            </div>
            <div class="metric">
                <div class="value">{len(sim_clusters)}</div>
                <div class="label">Similar Clusters</div>
            </div>
            <div class="metric">
                <div class="value">{len(families)}</div>
                <div class="label">Pattern Families</div>
            </div>
        </div>
        <div class="recommendation tip">
            <strong>Batch Processing:</strong> {dup_count + sim_count} jobs can be processed in batches,
            reducing LLM calls by approximately {effort_reduction:.0f}%.
        </div>
        <h3 style="margin-top: 20px;">Similarity Clusters</h3>
        <div class="cluster-list">
            {clusters_html}
        </div>
    </div>"""


def _generate_risk_table(predictions: List) -> str:
    """Generate HTML table for high-risk jobs."""
    high_risk = [p for p in predictions
                 if hasattr(p, 'risk_level') and p.risk_level.value in ('HIGH', 'CRITICAL')]

    if not high_risk:
        return '<p class="recommendation success">No high-risk jobs identified.</p>'

    rows = ""
    for pred in high_risk[:20]:
        risk_class = 'badge-high' if pred.risk_level.value == 'HIGH' else 'badge-critical'
        blockers = ', '.join(pred.automation_blockers[:2]) if hasattr(pred, 'automation_blockers') else ''

        rows += f"""
        <tr>
            <td>{pred.job_name}</td>
            <td><span class="badge {risk_class}">{pred.risk_level.value}</span></td>
            <td>{pred.estimated_hours:.1f}h</td>
            <td>{blockers}</td>
        </tr>"""

    return f"""
    <table>
        <thead>
            <tr>
                <th>Job Name</th>
                <th>Risk Level</th>
                <th>Effort</th>
                <th>Blockers</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>"""


def _generate_jobs_table(predictions: List) -> str:
    """Generate HTML table for all jobs."""
    rows = ""
    for pred in predictions[:100]:  # Limit to first 100
        cat_class = f'badge-{pred.category.value.lower().replace("_", "")}'
        if cat_class == 'badge-semiauto':
            cat_class = 'badge-semi'

        rows += f"""
        <tr>
            <td>{pred.job_name}</td>
            <td><span class="badge {cat_class}">{pred.category.value}</span></td>
            <td>{pred.confidence:.0%}</td>
            <td>{pred.success_probability:.0%}</td>
            <td>{pred.estimated_hours:.1f}h</td>
        </tr>"""

    return f"""
    <div style="max-height: 500px; overflow-y: auto;">
        <table>
            <thead>
                <tr>
                    <th>Job Name</th>
                    <th>Category</th>
                    <th>Confidence</th>
                    <th>Success Prob.</th>
                    <th>Effort</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    {f'<p style="color: #666; margin-top: 10px;">Showing first 100 of {len(predictions)} jobs</p>' if len(predictions) > 100 else ''}"""


def _generate_recommendations(summary: Dict, commonality, predictions: List) -> str:
    """Generate recommendations section."""
    recs = []

    # Auto migration recommendation
    auto_pct = summary.get('auto', {}).get('percentage', 0)
    if auto_pct >= 70:
        recs.append(('success', f'{auto_pct:.0f}% of jobs can be fully automated. Start with AUTO jobs for quick wins.'))
    elif auto_pct >= 40:
        recs.append(('tip', f'{auto_pct:.0f}% automation rate. Consider addressing common blockers to increase automation.'))
    else:
        recs.append(('warning', f'Only {auto_pct:.0f}% automation. This migration will require significant manual effort.'))

    # Batch optimization recommendation
    if commonality and hasattr(commonality, 'effort_reduction_percent'):
        reduction = commonality.effort_reduction_percent
        if reduction > 20:
            recs.append(('success', f'Batch processing can reduce effort by {reduction:.0f}%. Use similarity clusters.'))

    # High risk jobs
    high_risk = summary.get('high_risk_jobs', [])
    if len(high_risk) > 5:
        recs.append(('warning', f'{len(high_risk)} high-risk jobs identified. Review these carefully before migration.'))

    # Unknown stages
    unknown = summary.get('unknown_stages', {})
    if unknown.get('total_unique', 0) > 0:
        recs.append(('tip', f"{unknown.get('total_unique', 0)} unknown DataStage stage types. Consider adding mappings for better analysis."))

    # Effort distribution
    total_hours = summary.get('total_estimated_hours', 0)
    if total_hours > 0:
        recs.append(('tip', f'Total estimated effort: {total_hours:.0f} hours. Plan sprints accordingly.'))

    html = ""
    for rec_type, text in recs:
        html += f'<div class="recommendation {rec_type}">{text}</div>'

    return html


def _generate_errors_section(errors: List) -> str:
    """Generate errors section if any."""
    if not errors:
        return ""

    rows = ""
    for err in errors[:20]:
        rows += f"""
        <tr>
            <td>{err.get('file', 'Unknown')}</td>
            <td>{err.get('error', 'Unknown error')}</td>
        </tr>"""

    return f"""
    <div class="card">
        <h2>Parsing Errors ({len(errors)})</h2>
        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th>Error</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>"""

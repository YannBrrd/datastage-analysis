# DataStage to AWS Glue Migration System

A high-performance, AI-powered system for analyzing 9000+ DataStage ETL jobs and automating their migration to **AWS Glue**. Uses a hybrid approach combining local analysis with targeted LLM validation to minimize costs while maximizing migration success.

## 🎯 Key Features

- **Predictive Migration**: Automatically classifies jobs as AUTO/SEMI-AUTO/MANUAL
- **Commonality Detection**: Identifies duplicate/similar jobs to reduce migration effort
- **Code Generation**: Generates AWS Glue Python scripts from DataStage patterns
- **Cost Optimization**: < $300 in LLM costs for 9000 jobs analysis
- **High Automation**: 65-75% of jobs can be migrated automatically
- **Infrastructure as Code**: Terraform templates for Glue resources
- **Multi-format Support**: Handles .dsx, .dsx.gz, .xml, .xml.gz files

## 🏗️ Architecture

The system implements a **7-phase pipeline**:

```
Phase 1: EXTRACTION        → Parse DSX files (0 tokens)
Phase 2: FINGERPRINTING    → Hash-based structural clustering (0 tokens)
Phase 3: SEMANTIC CLUSTER  → Sentence-transformers embeddings (0 tokens)
Phase 4: PATTERN ANALYSIS  → Complexity scoring for AWS Glue (0 tokens)
Phase 5: PREDICTION        → Classify AUTO/SEMI-AUTO/MANUAL (0 tokens)
Phase 6: CODE GENERATION   → Generate Glue scripts + Terraform (0 tokens)
Phase 7: LLM VALIDATION    → Claude AI for edge cases (budget-controlled)
```

## 📊 Migration Categories

| Category | Description | Automation Level |
|----------|-------------|------------------|
| **AUTO** | Simple patterns, full code generation | 100% automated |
| **SEMI-AUTO** | Template + manual adjustments | 60-80% automated |
| **MANUAL** | Complex CDC/SCD, custom code | Requires manual work |

## 🚀 Quick Start

### Installation

```bash
# Clone and setup
git clone <repository>
cd datastage-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Redis for caching
docker run -d -p 6379:6379 redis:alpine
```

### Configuration

```bash
# Set Claude API key (for LLM validation phase)
export ANTHROPIC_API_KEY=your_key_here

# Optional: AWS credentials for Glue deployment
export AWS_PROFILE=your_profile
```

#### config.yaml

Customize parser limits and prediction settings:

```yaml
# Parser settings
parser:
  max_file_size_mb: 510      # Max file size warning threshold
  max_lines: 0               # 0 = unlimited parsing
  max_workers: 4             # Parallel processing threads

# Prediction settings
prediction:
  success_baseline: 0.85     # Base success probability
  effort_factor: 1.0         # Effort multiplier

# Glue generation settings
glue:
  glue_version: "4.0"
  default_worker_type: "G.1X"
  default_num_workers: 2
```

### Usage

```bash
# Place DSX files in data/ directory (supports .dsx, .dsx.gz, .xml, .xml.gz)
cp /path/to/your/*.dsx data/
cp /path/to/your/*.dsx.gz data/
cp /path/to/your/*.xml.gz data/

# Quick migration analysis (no LLM, instant results)
python analyze_migration.py ./data

# Export analysis to CSV
python analyze_migration.py ./data -o migration_report.csv

# Export to JSON with verbose output
python analyze_migration.py ./data -f json -o report.json -v

# Debug mode (shows parsing details)
python analyze_migration.py ./data --debug

# Run full pipeline with LLM validation
python main.py

# Run without LLM (local analysis only)
python main.py --skip-genai

# View interactive report
streamlit run output/app.py
```

### Migration Analyzer CLI

The `analyze_migration.py` script provides instant classification of your DataStage jobs:

| Option | Description |
|--------|-------------|
| `-o, --output` | Output file path (CSV or JSON) |
| `-f, --format` | Output format: console, csv, json |
| `-v, --verbose` | Show detailed output for each job |
| `-d, --debug` | Enable debug logging for parsing issues |

```
$ python analyze_migration.py ./data

🔍 DataStage to AWS Glue Migration Analyzer
   Analyzing: ./data

📁 Found 0 .dsx, 47 .dsx.gz, 0 .xml, 5 .xml.gz (total: 52)
------------------------------------------------------------
⏳ [██████████████████████████████] 100% (52/52) last_file.dsx.gz
✅ Parsed 52 files, found 7049 jobs

============================================================
📊 MIGRATION ANALYSIS REPORT
============================================================

📈 SUMMARY
   Total Jobs Analyzed: 7049

   Migration Categories:
   🟢 AUTO      : 2115 ( 30.0%) |███████████████░░░░░░░░░░░░░░░░░|
   🟡 SEMI-AUTO : 3525 ( 50.0%) |█████████████████████████░░░░░░░|
   🔴 MANUAL    : 1409 ( 20.0%) |██████████░░░░░░░░░░░░░░░░░░░░░░|

   Average Success Probability: 82.5%
   Total Estimated Effort: 12,450 hours

📋 COMMONALITY ANALYSIS
   Total Jobs: 7049
   Unique Patterns: 892

   🔁 Exact Duplicates: 342 jobs in 45 groups
   🔗 Similar Jobs (>85%): 1205 jobs in 89 clusters

   📂 Pattern Families (12):
      - DB to File ETL: 523 jobs → jdbc_to_s3_etl
      - File Processing: 312 jobs → s3_to_s3_etl
      - Lookup Enrichment: 89 jobs → join_lookup_etl

   💡 Effective Unique Jobs: 892 (vs 7049 total)
   📉 Estimated Effort Reduction: 87.3%

✨ 30.0% of jobs can be automatically migrated to AWS Glue
```

## 📁 Project Structure

```
datastage-analysis/
├── main.py                           # Pipeline orchestrator
├── analyze_migration.py              # CLI migration analyzer
├── config.yaml                       # Configuration settings
├── src/datastage_analysis/
│   ├── config.py                    # Configuration loader
│   ├── parsers/
│   │   └── dsx_parser.py            # DSX/XML parsing (.dsx, .dsx.gz, .xml, .xml.gz)
│   ├── clustering/
│   │   ├── structural_clusterer.py  # Hash-based clustering
│   │   └── semantic_clusterer.py    # K-means semantic clustering
│   ├── embeddings/
│   │   └── semantic_embedder.py     # Sentence-transformers
│   ├── analysis/
│   │   ├── pattern_analyzer.py      # Glue complexity scoring
│   │   └── commonality_detector.py  # Duplicate/pattern detection
│   ├── prediction/
│   │   └── migration_predictor.py   # AUTO/SEMI/MANUAL classifier
│   ├── generators/
│   │   └── glue_generator.py        # AWS Glue script generator
│   ├── templates/
│   │   ├── patterns/                # Jinja2 templates for Glue jobs
│   │   └── infrastructure/          # Terraform templates
│   ├── api/
│   │   ├── claude_comparator.py     # LLM integration
│   │   └── job_summarizer.py        # Token optimization
│   └── report/
│       └── interactive_report.py    # Streamlit dashboard
├── data/                            # Input DSX files
├── output/                          # Generated reports and scripts
└── docs/
    └── ARCHITECTURE.md              # Detailed architecture docs
```

## 🔧 AWS Glue Mapping

| DataStage Stage | AWS Glue Equivalent | Complexity |
|-----------------|---------------------|------------|
| SequentialFile / CSeqFileStage | S3 DynamicFrame | 1/5 |
| OracleConnector / COracleOCIStage | Glue JDBC Connection | 2/5 |
| Transformer / CTransformerStage | ApplyMapping / Map | 2/5 |
| Join | Join.apply() | 2/5 |
| Lookup / CHashedFileStage | Broadcast join | 3/5 |
| Aggregator | groupBy().agg() | 2/5 |
| CCustomStage | Custom Python logic | 3/5 |
| ChangeCapture | Glue Bookmarks + Delta | 5/5 |
| CJS* (Job Sequencer) | Step Functions / Glue Workflows | N/A |

## 📈 Expected Results

For a typical 9000 job DataStage environment:

| Metric | Value |
|--------|-------|
| Analysis time | < 2 hours |
| LLM cost | ~$150-300 |
| AUTO migrations | 30-40% |
| SEMI-AUTO migrations | 40-50% |
| MANUAL migrations | 10-20% |
| Average success probability | > 85% |

## 🛠️ Generated Outputs

1. **Glue Scripts** (`output/glue_scripts/`)
   - Python ETL scripts ready for Glue
   - Includes all transformations and error handling

2. **Terraform Configs** (`output/terraform/`)
   - Glue job definitions
   - IAM roles and policies
   - Connections and triggers

3. **Migration Report** (`output/migration_report.json`)
   - Per-job predictions
   - Risk assessment
   - Effort estimates

4. **Interactive Dashboard** (`output/app.py`)
   - Streamlit visualization
   - Filter by category, complexity, risk
   - Export to CSV/Excel

## 📚 Documentation

- [Architecture Details](ARCHITECTURE.md) - Full technical documentation
- [Glue Templates](src/datastage_analysis/templates/) - Template reference

## 🔒 Requirements

- Python 3.10+
- Redis (optional, for caching)
- Claude API key (for LLM validation)
- AWS credentials (for Glue deployment)

## 📦 Dependencies

```
sentence-transformers  # Local semantic embeddings
anthropic              # Claude AI API
scikit-learn           # Clustering algorithms
jinja2                 # Template engine
pandas / numpy         # Data processing
streamlit              # Interactive reports
pyyaml                 # Configuration loading
redis                  # Caching (optional)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

# DataStage Migration System

A high-performance, AI-powered system for analyzing 9000+ DataStage ETL jobs and automating their migration to **AWS Glue** or **SQL databases (Teradata, PostgreSQL, etc.)**. Uses a hybrid approach combining local analysis with targeted LLM validation to minimize costs while maximizing migration success.

## 🎯 Key Features

- **Multi-Target Support**: Generate code for AWS Glue or SQL databases (Teradata, PostgreSQL)
- **Predictive Migration**: Automatically classifies jobs as AUTO/SEMI-AUTO/MANUAL
- **Commonality Detection**: Identifies duplicate/similar jobs to reduce migration effort
- **Batch Processing**: Groups similar jobs to minimize LLM calls (up to 90% cost reduction)
- **Multi-Provider LLM**: Support for Anthropic, Azure, AWS Bedrock, GCP Vertex, OpenRouter
- **Cost Optimization**: Dry-run mode for cost estimation before execution
- **Code Generation**: Generates Glue/SQL scripts, Terraform/DDL, unit tests, and docs
- **HTML Reports**: Interactive migration reports with charts and recommendations
- **High Automation**: 65-75% of jobs can be migrated automatically
- **Multi-format Support**: Handles .dsx, .dsx.gz, .xml, .xml.gz files

## 🏗️ Architecture

The system implements a **7-phase pipeline**:

```
Phase 1: EXTRACTION        → Parse DSX files (0 tokens)
Phase 2: FINGERPRINTING    → Hash-based structural clustering (0 tokens)
Phase 3: SEMANTIC CLUSTER  → Sentence-transformers embeddings (0 tokens)
Phase 4: PATTERN ANALYSIS  → Complexity scoring for AWS Glue (0 tokens)
Phase 5: PREDICTION        → Classify AUTO/SEMI-AUTO/MANUAL (0 tokens)
Phase 6: CODE GENERATION   → Generate Glue scripts + Terraform (LLM optimized)
Phase 7: LLM VALIDATION    → Claude AI for edge cases (budget-controlled)
```

### Batch Processing Architecture

```
Analysis Phase                         Generation Phase
┌─────────────────────┐               ┌─────────────────────────────┐
│  CommonalityDetector│       →       │    MigrationGenerator       │
│  ─────────────────── │               │    ───────────────────      │
│  • duplicate_group   │               │    cluster_info             │
│  • similarity_cluster│               │         ↓                   │
│  • pattern_family    │               │    BatchProcessor           │
└─────────────────────┘               └─────────────────────────────┘
                                                  ↓
                                      ┌─────────────────────────────┐
                                      │      LLMGenerator           │
                                      │  1. Template: full LLM call │
                                      │  2. Variations:             │
                                      │     - Simple: 0 LLM calls   │
                                      │     - Complex: light call   │
                                      └─────────────────────────────┘
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
```

### Configuration

```bash
# Set LLM API key (choose one provider)
export ANTHROPIC_API_KEY=your_key_here
# OR
export AZURE_OPENAI_API_KEY=your_key_here
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# OR
export OPENROUTER_API_KEY=your_key_here
```

#### config.yaml

```yaml
# Parser settings
parser:
  max_file_size_mb: 510
  max_lines: 0  # 0 = unlimited
  max_workers: 4  # Parallel processing

# Prediction settings
prediction:
  success_baseline: 0.85
  effort_factor: 1.0

# LLM Configuration
llm:
  enabled: true
  provider: "anthropic"  # anthropic, azure, azure_foundry, aws, gcp, openrouter

  models:
    anthropic: "claude-sonnet-4-20250514"
    azure: "gpt-4o"
    azure_foundry: "meta-llama-3.1-70b-instruct"
    aws: "anthropic.claude-3-sonnet-20240229-v1:0"
    gcp: "gemini-1.5-pro"
    openrouter: "anthropic/claude-sonnet-4"

  # Caching
  cache:
    enabled: true
    ttl_hours: 168  # 7 days
    db_path: ".cache/llm_cache.db"

  # Cost control
  budget:
    max_tokens_per_job: 8000
    max_cost_per_run: 50.0

# Glue generation settings
glue:
  glue_version: "4.0"
  default_worker_type: "G.1X"
  default_num_workers: 2

# Generation output
generation:
  output_dir: "./generated"
  generate_terraform: true
  generate_tests: true
  generate_docs: true
```

## 📋 CLI Reference

### Basic Analysis

```bash
# Analyze DataStage files
python analyze_migration.py ./data

# Export to CSV
python analyze_migration.py ./data -o report.csv

# Export to JSON
python analyze_migration.py ./data -f json -o report.json

# Verbose output
python analyze_migration.py ./data -v

# Debug mode
python analyze_migration.py ./data --debug
```

### Code Generation

```bash
# Analyze and generate AWS Glue code (default)
python analyze_migration.py ./data --generate

# Generate SQL/Teradata scripts instead of Glue
python analyze_migration.py ./data --generate --target sql --sql-dialect teradata

# Generate for specific jobs
python analyze_migration.py ./data --generate-only "JOB1,JOB2,JOB3"

# Generate without LLM (rule-based only)
python analyze_migration.py ./data --generate --no-llm

# Use specific LLM provider
python analyze_migration.py ./data --generate --llm-provider azure
```

### Multi-Target Generation

```bash
# AWS Glue (default)
python analyze_migration.py ./data --generate --target glue

# SQL/Teradata
python analyze_migration.py ./data --generate --target sql --sql-dialect teradata

# SQL/PostgreSQL
python analyze_migration.py ./data --generate --target sql --sql-dialect postgresql

# SQL/Oracle
python analyze_migration.py ./data --generate --target sql --sql-dialect oracle
```

### Dry-Run Mode (Cost Estimation)

```bash
# Estimate costs before generation
python analyze_migration.py ./data --dry-run

# With budget limit
python analyze_migration.py ./data --dry-run --budget 50.0

# Export dry-run results
python analyze_migration.py ./data --dry-run -o results.json
```

### HTML Reports

```bash
# Generate interactive HTML report
python analyze_migration.py ./data --report migration_report.html

# Combined: analysis + report + generation
python analyze_migration.py ./data --report report.html --generate
```

### Full CLI Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output file path (CSV or JSON) |
| `-f, --format` | Output format: console, csv, json |
| `-v, --verbose` | Show detailed output for each job |
| `-d, --debug` | Enable debug logging |
| `--generate` | Generate migration code after analysis |
| `--generate-only JOBS` | Generate for specific jobs (comma-separated) |
| `--output-dir DIR` | Output directory for generated code |
| `--target` | Target platform: glue, sql (default: glue) |
| `--sql-dialect` | SQL dialect: teradata, postgresql, oracle, sqlserver, generic |
| `--no-llm` | Disable LLM, use rule-based only |
| `--llm-provider` | LLM provider: anthropic, azure, azure_foundry, aws, gcp, openrouter |
| `--dry-run` | Estimate costs without generating |
| `--budget DOLLARS` | Set budget limit for warnings |
| `--report PATH` | Generate HTML report |

## 🔧 LLM Provider Configuration

### Anthropic (Claude)

```bash
export ANTHROPIC_API_KEY=sk-ant-xxx
```

```yaml
llm:
  provider: "anthropic"
  models:
    anthropic: "claude-sonnet-4-20250514"
```

### Azure OpenAI

```bash
export AZURE_OPENAI_API_KEY=xxx
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

```yaml
llm:
  provider: "azure"
  models:
    azure: "gpt-4o"
```

### Azure AI Foundry

```bash
export AZURE_AI_FOUNDRY_ENDPOINT=https://your-endpoint.inference.ai.azure.com
export AZURE_AI_FOUNDRY_API_KEY=xxx
```

```yaml
llm:
  provider: "azure_foundry"
  models:
    azure_foundry: "meta-llama-3.1-70b-instruct"
```

### AWS Bedrock

```bash
export AWS_REGION=us-east-1
# Uses default AWS credentials chain
```

```yaml
llm:
  provider: "aws"
  models:
    aws: "anthropic.claude-3-sonnet-20240229-v1:0"
```

### GCP Vertex AI

```bash
export GCP_PROJECT_ID=your-project
export GCP_LOCATION=us-central1
# Uses default Google credentials
```

```yaml
llm:
  provider: "gcp"
  models:
    gcp: "gemini-1.5-pro"
```

### OpenRouter

```bash
export OPENROUTER_API_KEY=sk-or-xxx
```

```yaml
llm:
  provider: "openrouter"
  models:
    openrouter: "anthropic/claude-sonnet-4"
```

## 📦 Batch Processing & Cost Optimization

### How Batch Processing Works

1. **Analysis Phase**: Jobs are grouped by similarity clusters
2. **Template Selection**: Best job selected as template per cluster
3. **Single LLM Call**: Full code generated for template only
4. **Variation Application**: Similar jobs get code via string replacement

### Cost Savings Example

| Cluster | Jobs | Without Batch | With Batch | Savings |
|---------|------|---------------|------------|---------|
| SIM_1 | 10 | 10 LLM calls | 1-2 calls | ~85% |
| SIM_2 | 5 | 5 LLM calls | 1 call | 80% |
| DUP_1 | 20 | 20 LLM calls | 1 call | 95% |

### Dry-Run Output

```
======================================================================
DRY-RUN: MIGRATION COST ESTIMATE
======================================================================

Provider: anthropic

JOB SUMMARY
----------------------------------------
Total Jobs:      150
  AUTO:          45 (rule-based, no LLM)
  SEMI-AUTO:     75 (hybrid, LLM enhanced)
  MANUAL:        30 (LLM skeleton generation)

BATCH OPTIMIZATION
----------------------------------------
Batches identified: 8

  SIM_1:
    Template: ETL_SALES_DAILY
    Jobs: 12
    Est. LLM calls: 2 (vs 12 without batching)
    Est. savings: $0.0450

COST ESTIMATE
----------------------------------------
Estimated LLM calls:    25
Estimated tokens:       87,500

Cost WITHOUT batching:  $0.3750
Cost WITH batching:     $0.0825
SAVINGS:                $0.2925 (78.0%)

======================================================================
To proceed with generation, remove --dry-run flag
======================================================================
```

## 📊 HTML Report Features

The HTML report includes:

- **Executive Summary**: Key metrics at a glance
- **Migration Categories**: Visual breakdown with charts
- **Batch Optimization**: Cluster analysis and savings potential
- **Risk Analysis**: High-risk jobs table
- **Job Details**: Searchable/sortable job listing
- **Recommendations**: Actionable insights
- **Error Summary**: Parsing failures if any

## 🛠️ Generated Outputs

### AWS Glue Target (--target glue)

```
generated/
├── glue_jobs/
│   ├── job_name.py           # AWS Glue PySpark script
│   └── ...
├── terraform/
│   ├── job_name.tf           # Glue job Terraform config
│   └── ...
├── tests/
│   ├── test_job_name.py      # pytest unit tests
│   └── ...
├── docs/
│   ├── job_name.md           # Per-job documentation
│   └── ...
└── generation_report.json    # Generation summary
```

### SQL Target (--target sql)

```
generated/
├── sql/
│   ├── job_name.sql          # SQL script (BTEQ for Teradata)
│   └── ...
├── ddl/
│   ├── job_name.sql          # DDL (CREATE TABLE, GRANT)
│   └── ...
├── tests/
│   ├── test_job_name.sql     # Validation queries
│   └── ...
├── docs/
│   ├── job_name.md           # Per-job documentation
│   └── ...
└── generation_report.json    # Generation summary
```

## 📁 Project Structure

```
datastage-analysis/
├── analyze_migration.py              # CLI entry point
├── config.yaml                       # Configuration
├── src/datastage_analysis/
│   ├── parsers/
│   │   └── dsx_parser.py             # DSX/XML parsing
│   ├── analysis/
│   │   ├── pattern_analyzer.py       # Pattern detection
│   │   └── commonality_detector.py   # Similarity clustering
│   ├── prediction/
│   │   └── migration_predictor.py    # Classification
│   ├── generation/
│   │   ├── generator.py              # Main generator
│   │   ├── dry_run.py                # Cost estimation
│   │   ├── rule_based/               # Template generation
│   │   ├── llm_based/                # LLM-assisted generation
│   │   └── targets/                  # Target-specific generators
│   │       ├── glue/                 # AWS Glue target
│   │       └── sql/                  # SQL/Teradata target
│   ├── llm/
│   │   ├── client.py                 # Base LLM interface
│   │   ├── factory.py                # Provider factory
│   │   ├── providers/                # All LLM providers
│   │   │   ├── anthropic.py
│   │   │   ├── azure_openai.py
│   │   │   ├── azure_foundry.py
│   │   │   ├── aws_bedrock.py
│   │   │   ├── gcp_vertex.py
│   │   │   └── openrouter.py
│   │   ├── cache/                    # Response caching
│   │   │   ├── response_cache.py     # Exact match
│   │   │   └── semantic_cache.py     # Similarity-based
│   │   └── optimization/             # Cost optimization
│   │       ├── token_counter.py
│   │       ├── prompt_optimizer.py
│   │       ├── batch_processor.py
│   │       └── cost_tracker.py
│   └── reporting/
│       └── html_report.py            # HTML report generation
├── data/                             # Input DSX files
└── generated/                        # Output directory
```

## 🔧 Target Mappings

### AWS Glue Mapping

| DataStage Stage | AWS Glue Equivalent | Complexity |
|-----------------|---------------------|------------|
| SequentialFile | S3 DynamicFrame | 1/5 |
| OracleConnector | Glue JDBC Connection | 2/5 |
| Transformer | ApplyMapping / Map | 2/5 |
| Join | Join.apply() | 2/5 |
| Lookup | Broadcast join | 3/5 |
| Aggregator | groupBy().agg() | 2/5 |
| CCustomStage | Custom Python logic | 3/5 |
| ChangeCapture | Glue Bookmarks + Delta | 5/5 |

### SQL/Teradata Mapping

| DataStage Stage | SQL Equivalent | Notes |
|-----------------|----------------|-------|
| SequentialFile | Staging table + BTEQ load | Use TPT for high-volume |
| OracleConnector | SELECT/INSERT | Direct table access |
| TeradataConnector | SELECT/INSERT | Native Teradata SQL |
| Transformer | SQL expressions | Column derivations |
| Join | JOIN clause | INNER/LEFT/RIGHT JOIN |
| Lookup | LEFT JOIN | Lookup as join operation |
| Aggregator | GROUP BY + aggregates | SUM, COUNT, AVG, etc. |
| Filter | WHERE clause | Filter conditions |
| Sort | ORDER BY | Sorting |
| RemoveDuplicates | DISTINCT / QUALIFY | ROW_NUMBER for dedup |

### Teradata-Specific Features

```sql
-- MULTISET TABLE (allows duplicates, better performance)
CREATE MULTISET TABLE my_table (
    id INTEGER NOT NULL,
    name VARCHAR(100)
)
PRIMARY INDEX (id);

-- MERGE for upsert patterns
MERGE INTO target_table AS tgt
USING source_query AS src
ON tgt.key = src.key
WHEN MATCHED THEN UPDATE SET ...
WHEN NOT MATCHED THEN INSERT ...;

-- BTEQ batch processing
.LOGON tdpid/user,password
DATABASE mydb;
-- SQL statements here
.LOGOFF
.QUIT
```

## 📈 Expected Results

For a typical 9000 job DataStage environment:

| Metric | Value |
|--------|-------|
| Analysis time | < 2 hours |
| LLM cost (with batching) | ~$50-150 |
| AUTO migrations | 30-40% |
| SEMI-AUTO migrations | 40-50% |
| MANUAL migrations | 10-20% |
| Batch optimization savings | 70-90% |

## 🔒 Requirements

- Python 3.10+
- LLM API key (any supported provider)
- AWS credentials (for Glue deployment)

## 📦 Dependencies

```
anthropic              # Claude API
openai                 # Azure OpenAI
boto3                  # AWS Bedrock
google-cloud-aiplatform  # GCP Vertex
sentence-transformers  # Local embeddings
scikit-learn           # Clustering
jinja2                 # Templates
pyyaml                 # Configuration
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

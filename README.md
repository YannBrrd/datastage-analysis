# DataStage to AWS Glue Migration System

A high-performance, AI-powered system for analyzing 9000+ DataStage ETL jobs and automating their migration to **AWS Glue**. Uses a hybrid approach combining local analysis with targeted LLM validation to minimize costs while maximizing migration success.

## 🎯 Key Features

- **Predictive Migration**: Automatically classifies jobs as AUTO/SEMI-AUTO/MANUAL
- **Code Generation**: Generates AWS Glue Python scripts from DataStage patterns
- **Cost Optimization**: < $300 in LLM costs for 9000 jobs analysis
- **High Automation**: 65-75% of jobs can be migrated automatically
- **Infrastructure as Code**: Terraform templates for Glue resources

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

### Usage

```bash
# Place DSX files in data/ directory
cp /path/to/your/*.dsx data/

# Run full analysis pipeline
python main.py

# Run without LLM (local analysis only)
python main.py --skip-genai

# Generate Glue scripts for analyzed jobs
python -m datastage_analysis.generators.glue_generator

# View interactive report
streamlit run output/app.py
```

## 📁 Project Structure

```
datastage-analysis/
├── main.py                           # Pipeline orchestrator
├── src/datastage_analysis/
│   ├── parsers/
│   │   └── dsx_parser.py            # DSX/XML parsing
│   ├── clustering/
│   │   ├── structural_clusterer.py  # Hash-based clustering
│   │   └── semantic_clusterer.py    # K-means semantic clustering
│   ├── embeddings/
│   │   └── semantic_embedder.py     # Sentence-transformers
│   ├── analysis/
│   │   └── pattern_analyzer.py      # Glue complexity scoring
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
| SequentialFile | S3 DynamicFrame | 1/5 |
| OracleConnector | Glue JDBC Connection | 2/5 |
| Transformer | ApplyMapping / Map | 2/5 |
| Join | Join.apply() | 2/5 |
| Lookup | Broadcast join | 3/5 |
| Aggregator | groupBy().agg() | 2/5 |
| ChangeCapture | Glue Bookmarks + Delta | 5/5 |

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
scikit-learn          # Clustering algorithms
jinja2                # Template engine
pandas / numpy        # Data processing
streamlit             # Interactive reports
redis                 # Caching (optional)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

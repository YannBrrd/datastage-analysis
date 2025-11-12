# DataStage Job Comparison System

A high-performance system for comparing 9000 DataStage ETL jobs using a 6-phase pipeline that minimizes Claude AI token consumption while maximizing analysis quality.

## Architecture

The system implements a 6-phase pipeline:

1. **Structural Fingerprinting**: Parse DSX/XML files to extract structural signatures (0 tokens)
2. **Structural Clustering**: Group jobs by identical hash fingerprints (0 tokens)
3. **Semantic Clustering**: Use sentence-transformers embeddings for semantic grouping (0 tokens)
4. **Representative Selection**: Choose ~1000 representative jobs from clusters
5. **Fine Comparison**: Batch comparison using Claude AI with prompt caching
6. **Interactive Report**: Generate Streamlit dashboard with analysis results

## Features

- **Asynchronous Processing**: asyncio for I/O operations
- **Multiprocessing**: Parallel XML parsing
- **Redis Caching**: Avoid redundant AI calls
- **Token Optimization**: < 100M tokens budget (~$300)
- **Performance**: Process 9000 jobs in < 2 hours
- **Interactive Reports**: Streamlit dashboard for exploration

## Installation

1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up Redis (optional, for caching):
   ```bash
   # Using Docker
   docker run -d -p 6379:6379 redis:alpine
   ```

## Usage

1. Place DSX files in the `data/` directory
2. Set Claude API key:
   ```bash
   export ANTHROPIC_API_KEY=your_key_here
   ```
3. Run the pipeline:
   ```bash
   python main.py
   ```
4. View results in `output/` directory
5. Launch interactive report:
   ```bash
   cd output
   streamlit run app.py
   ```

## Configuration

- Adjust cluster counts in `main.py`
- Modify embedding model in `semantic_embedder.py`
- Configure Claude model and prompts in `claude_comparator.py`
- Adjust Redis settings in `redis_cache.py`

## Requirements

- Python 3.10+
- Redis (optional)
- Claude API key

## Dependencies

- sentence-transformers: Local embeddings
- anthropic: Claude AI API
- redis: Caching
- scikit-learn: Clustering
- streamlit: Interactive reports
- pandas/numpy: Data processing
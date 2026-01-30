# Migration Code Generation - Implementation Plan

## Overview

This document outlines the architecture and implementation plan for automatic migration code generation from DataStage to AWS Glue, with optional LLM assistance.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Migration Generation Engine                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │   Rule-Based     │    │   LLM-Assisted   │    │     Hybrid       │      │
│  │   Generator      │    │   Generator      │    │   Generator      │      │
│  │                  │    │                  │    │                  │      │
│  │ • AUTO jobs      │    │ • Complex logic  │    │ • SEMI-AUTO jobs │      │
│  │ • Simple ETL     │    │ • Custom stages  │    │ • Rules + LLM    │      │
│  │ • Known patterns │    │ • MANUAL jobs    │    │   fallback       │      │
│  └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘      │
│           │                       │                       │                 │
│           └───────────────────────┼───────────────────────┘                 │
│                                   ▼                                         │
│                    ┌──────────────────────────────┐                         │
│                    │      Output Generator        │                         │
│                    │  • PySpark scripts           │                         │
│                    │  • Terraform/CloudFormation  │                         │
│                    │  • Unit tests                │                         │
│                    │  • Documentation             │                         │
│                    └──────────────────────────────┘                         │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                         LLM Abstraction Layer                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        LLMClient (Abstract)                          │   │
│  │  • complete(prompt, system, temperature, max_tokens) -> Response     │   │
│  │  • complete_with_cache(prompt, ...) -> Response                      │   │
│  │  • estimate_tokens(text) -> int                                      │   │
│  │  • get_model_info() -> ModelInfo                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │           │           │           │           │                 │
│           ▼           ▼           ▼           ▼           ▼                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────┐ │
│  │   Azure     │ │    AWS      │ │  Anthropic  │ │    GCP      │ │ Open  │ │
│  │   OpenAI    │ │   Bedrock   │ │    API      │ │   Vertex    │ │Router │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └───────┘ │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                         Caching & Optimization                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Prompt Cache   │  │  Response Cache │  │ Token Optimizer │             │
│  │  (semantic)     │  │  (exact match)  │  │ (compression)   │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## File Structure

```
src/datastage_analysis/
├── generation/                      # NEW: Code generation module
│   ├── __init__.py
│   ├── generator.py                 # Main MigrationGenerator orchestrator
│   ├── rule_based/
│   │   ├── __init__.py
│   │   ├── transformer.py           # Rule-based stage transformations
│   │   ├── stage_mappings.py        # DataStage → Glue stage mappings
│   │   └── templates/
│   │       ├── glue_job_base.py.j2
│   │       ├── glue_etl_simple.py.j2
│   │       ├── glue_etl_jdbc.py.j2
│   │       ├── glue_etl_s3.py.j2
│   │       └── glue_etl_transform.py.j2
│   ├── llm_based/
│   │   ├── __init__.py
│   │   ├── transformer.py           # LLM-assisted transformations
│   │   ├── prompts/
│   │   │   ├── system_prompt.txt
│   │   │   ├── transform_stage.txt
│   │   │   ├── custom_stage.txt
│   │   │   └── sql_conversion.txt
│   │   └── context_builder.py       # Build context for LLM
│   └── output/
│       ├── __init__.py
│       ├── glue_script.py           # Generate Glue Python scripts
│       ├── terraform.py             # Generate Terraform configs
│       ├── cloudformation.py        # Generate CFN templates
│       ├── tests.py                 # Generate unit tests
│       └── documentation.py         # Generate mapping docs
│
├── llm/                             # NEW: LLM abstraction layer
│   ├── __init__.py
│   ├── client.py                    # Abstract LLMClient base class
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── anthropic.py             # Anthropic Claude API
│   │   ├── azure_openai.py          # Azure OpenAI
│   │   ├── aws_bedrock.py           # AWS Bedrock
│   │   ├── gcp_vertex.py            # GCP Vertex AI
│   │   └── openrouter.py            # OpenRouter (multi-model)
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── response_cache.py        # Exact match response caching
│   │   ├── semantic_cache.py        # Semantic similarity caching
│   │   └── storage.py               # SQLite/Redis storage backend
│   └── optimization/
│       ├── __init__.py
│       ├── token_counter.py         # Token estimation
│       ├── prompt_optimizer.py      # Compress/optimize prompts
│       └── batch_processor.py       # Batch multiple requests
│
└── config.py                        # Updated with LLM settings
```

## Configuration Schema

```yaml
# config.yaml additions

# LLM Configuration
llm:
  # Enable/disable LLM features entirely
  enabled: true

  # Provider selection: anthropic, azure, aws, gcp, openrouter
  provider: "anthropic"

  # Model selection per provider
  models:
    anthropic: "claude-sonnet-4-20250514"
    azure: "gpt-4o"
    aws: "anthropic.claude-3-sonnet-20240229-v1:0"
    gcp: "gemini-1.5-pro"
    openrouter: "anthropic/claude-sonnet-4"

  # Provider-specific settings
  providers:
    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"  # From env var
      base_url: null  # Optional override

    azure:
      api_key: "${AZURE_OPENAI_API_KEY}"
      endpoint: "${AZURE_OPENAI_ENDPOINT}"
      api_version: "2024-02-15-preview"
      deployment_name: "gpt-4o"

    aws:
      region: "us-east-1"
      profile: null  # Use default credentials
      # Uses boto3 default credential chain

    gcp:
      project_id: "${GCP_PROJECT_ID}"
      location: "us-central1"
      # Uses Application Default Credentials

    openrouter:
      api_key: "${OPENROUTER_API_KEY}"
      base_url: "https://openrouter.ai/api/v1"

  # Request settings
  settings:
    temperature: 0.2          # Low for deterministic output
    max_tokens: 4096
    timeout_seconds: 120
    max_retries: 3
    retry_delay_seconds: 2

# Caching Configuration
cache:
  enabled: true

  # Cache backend: sqlite, redis, memory
  backend: "sqlite"

  # Cache location
  path: ".cache/llm_cache.db"

  # Redis settings (if backend: redis)
  redis:
    host: "localhost"
    port: 6379
    db: 0

  # Cache TTL (time-to-live)
  ttl_hours: 168  # 7 days

  # Semantic cache settings
  semantic:
    enabled: true
    similarity_threshold: 0.95  # 95% similarity = cache hit
    embedding_model: "text-embedding-3-small"

# Token Optimization
optimization:
  # Compress prompts to reduce token usage
  compress_prompts: true

  # Batch similar requests
  batch_similar_jobs: true
  batch_size: 5

  # Use smaller model for simple tasks
  tiered_models:
    enabled: true
    simple_tasks: "claude-haiku"    # For simple validations
    complex_tasks: "claude-sonnet"  # For code generation

# Generation Settings
generation:
  # Operating mode: batch, interactive
  mode: "batch"

  # Output directory
  output_dir: "./generated"

  # What to generate
  outputs:
    glue_scripts: true
    terraform: true
    cloudformation: false  # Choose one IaC
    unit_tests: true
    documentation: true

  # Generation strategy per category
  strategy:
    auto:
      use_llm: false
      generator: "rule_based"
    semi_auto:
      use_llm: true
      generator: "hybrid"
      llm_for: ["complex_transforms", "sql_conversion"]
    manual:
      use_llm: true
      generator: "llm_based"
      output_style: "skeleton_with_todos"

  # Validation
  validation:
    syntax_check: true        # Python syntax validation
    import_check: true        # Verify imports exist
    glue_api_check: true      # Verify Glue API usage

  # On generation failure
  fallback:
    on_llm_error: "template"  # template, skip, error
    on_validation_error: "mark_for_review"
```

## LLM Abstraction Layer

### Base Client Interface

```python
# src/datastage_analysis/llm/client.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class LLMResponse:
    """Response from LLM provider."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cached: bool = False
    latency_ms: float = 0.0

@dataclass
class ModelInfo:
    """Information about the model."""
    provider: str
    model_id: str
    max_tokens: int
    supports_system_prompt: bool
    cost_per_1k_input: float
    cost_per_1k_output: float

class LLMClient(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        stop_sequences: Optional[List[str]] = None,
    ) -> LLMResponse:
        """Generate completion from the model."""
        pass

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Get information about the current model."""
        pass

    def complete_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        **kwargs
    ) -> LLMResponse:
        """Complete with automatic retry on failure."""
        # Implementation with exponential backoff
        pass
```

### Provider Factory

```python
# src/datastage_analysis/llm/__init__.py

def get_llm_client(
    provider: str = None,
    model: str = None,
    **kwargs
) -> LLMClient:
    """
    Factory function to get LLM client.

    Usage:
        client = get_llm_client()  # Uses config defaults
        client = get_llm_client(provider="anthropic", model="claude-sonnet-4-20250514")
        client = get_llm_client(provider="azure")
    """
    config = get_config()

    provider = provider or config.get('llm', 'provider', default='anthropic')

    providers = {
        'anthropic': AnthropicClient,
        'azure': AzureOpenAIClient,
        'aws': BedrockClient,
        'gcp': VertexClient,
        'openrouter': OpenRouterClient,
    }

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}")

    return providers[provider](model=model, **kwargs)
```

## Caching Strategy

### Multi-Level Cache

```
Request Flow:

┌─────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────┐
│ Request │───▶│ Exact Match  │───▶│  Semantic    │───▶│   LLM   │
│         │    │    Cache     │    │    Cache     │    │   API   │
└─────────┘    └──────────────┘    └──────────────┘    └─────────┘
                     │                    │                  │
                     │ HIT               │ HIT              │
                     ▼                    ▼                  ▼
              ┌──────────────────────────────────────────────────┐
              │                    Response                       │
              └──────────────────────────────────────────────────┘
```

### Cache Key Generation

```python
def generate_cache_key(
    prompt: str,
    system: Optional[str],
    model: str,
    temperature: float
) -> str:
    """Generate deterministic cache key."""
    content = f"{model}:{temperature}:{system or ''}:{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()
```

### Semantic Cache

For near-duplicate jobs, use embedding similarity:

```python
class SemanticCache:
    """Cache based on semantic similarity."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.embeddings = {}  # prompt_hash -> embedding

    def find_similar(self, prompt: str) -> Optional[str]:
        """Find cached response for semantically similar prompt."""
        embedding = self._get_embedding(prompt)

        for cached_hash, cached_embedding in self.embeddings.items():
            similarity = cosine_similarity(embedding, cached_embedding)
            if similarity >= self.threshold:
                return self._get_response(cached_hash)

        return None
```

## Token Optimization

### 1. Prompt Compression

```python
class PromptOptimizer:
    """Optimize prompts to reduce token usage."""

    def compress(self, prompt: str, target_reduction: float = 0.3) -> str:
        """
        Compress prompt while preserving meaning.

        Techniques:
        - Remove redundant whitespace
        - Abbreviate common patterns
        - Use reference IDs for repeated structures
        """
        # Remove excessive whitespace
        prompt = re.sub(r'\n\s*\n', '\n\n', prompt)
        prompt = re.sub(r' +', ' ', prompt)

        # Deduplicate repeated stage definitions
        prompt = self._deduplicate_stages(prompt)

        return prompt
```

### 2. Batching Similar Jobs

```python
class BatchProcessor:
    """Batch similar jobs into single LLM request."""

    def batch_by_pattern(
        self,
        jobs: List[JobStructure],
        max_batch_size: int = 5
    ) -> List[List[JobStructure]]:
        """
        Group jobs by pattern for batch processing.

        Example:
        - 10 "DB to File ETL" jobs → 2 batches of 5
        - Each batch: generate template + parameterization
        """
        pattern_groups = defaultdict(list)

        for job in jobs:
            pattern = self._get_pattern_signature(job)
            pattern_groups[pattern].append(job)

        batches = []
        for pattern, group in pattern_groups.items():
            for i in range(0, len(group), max_batch_size):
                batches.append(group[i:i + max_batch_size])

        return batches
```

### 3. Tiered Model Selection

```python
def select_model_for_task(task_complexity: str) -> str:
    """Select appropriate model based on task complexity."""
    config = get_config()

    if not config.get('optimization', 'tiered_models', 'enabled'):
        return config.get('llm', 'model')

    tiers = {
        'simple': config.get('optimization', 'tiered_models', 'simple_tasks'),
        'complex': config.get('optimization', 'tiered_models', 'complex_tasks'),
    }

    return tiers.get(task_complexity, tiers['complex'])
```

## Generation Flow

### Main Orchestrator

```python
# src/datastage_analysis/generation/generator.py

class MigrationGenerator:
    """Orchestrates migration code generation."""

    def __init__(self, config: Config):
        self.config = config
        self.rule_based = RuleBasedGenerator()
        self.llm_client = get_llm_client() if config.get('llm', 'enabled') else None
        self.llm_generator = LLMGenerator(self.llm_client) if self.llm_client else None

    def generate(
        self,
        predictions: List[MigrationPrediction],
        structures: Dict[str, Dict],
        mode: str = "batch"
    ) -> GenerationResult:
        """
        Generate migration code for analyzed jobs.

        Args:
            predictions: Analysis results from MigrationPredictor
            structures: Job structures from parser
            mode: "batch" or "interactive"
        """
        results = GenerationResult()

        # Group by category
        auto_jobs = [p for p in predictions if p.category == MigrationCategory.AUTO]
        semi_jobs = [p for p in predictions if p.category == MigrationCategory.SEMI_AUTO]
        manual_jobs = [p for p in predictions if p.category == MigrationCategory.MANUAL]

        # Phase 1: Rule-based generation for AUTO jobs
        for pred in auto_jobs:
            structure = structures.get(pred.job_name)
            result = self.rule_based.generate(pred, structure)
            results.add(pred.job_name, result)

        # Phase 2: Hybrid generation for SEMI-AUTO jobs
        if self.llm_generator and semi_jobs:
            # Batch similar jobs
            batches = self._batch_by_pattern(semi_jobs, structures)
            for batch in batches:
                batch_results = self._generate_batch_hybrid(batch, structures)
                results.merge(batch_results)

        # Phase 3: LLM generation for MANUAL jobs
        if self.llm_generator and manual_jobs:
            for pred in manual_jobs:
                structure = structures.get(pred.job_name)
                result = self.llm_generator.generate_skeleton(pred, structure)
                results.add(pred.job_name, result)

        # Phase 4: Generate outputs
        self._write_outputs(results)

        return results
```

## CLI Integration

```python
# analyze_migration.py additions

def main():
    parser = argparse.ArgumentParser(...)

    # Existing arguments...

    # New generation arguments
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate migration code after analysis"
    )

    parser.add_argument(
        "--generate-only",
        type=str,
        metavar="JOBS",
        help="Generate code for specific jobs (comma-separated or 'all')"
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM, use rule-based generation only"
    )

    parser.add_argument(
        "--llm-provider",
        choices=["anthropic", "azure", "aws", "gcp", "openrouter"],
        help="Override LLM provider from config"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: review each generated file"
    )

    # Usage examples:
    # python analyze_migration.py ./data --generate
    # python analyze_migration.py ./data --generate --no-llm
    # python analyze_migration.py ./data --generate --llm-provider azure
    # python analyze_migration.py ./data --generate-only "job1,job2,job3"
    # python analyze_migration.py ./data --generate --interactive
```

## Output Artifacts

### 1. Glue Script

```python
# generated/glue_jobs/job_name.py

"""
AWS Glue Job: job_name
Generated from DataStage job: JOB_NAME
Generation: AUTO (rule-based)
Date: 2024-01-15

Source stages: OracleConnector, Transformer, SequentialFile
Target: S3
"""

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

# ... generated code ...
```

### 2. Terraform

```hcl
# generated/terraform/job_name.tf

resource "aws_glue_job" "job_name" {
  name     = "job_name"
  role_arn = var.glue_role_arn

  command {
    name            = "glueetl"
    script_location = "s3://${var.scripts_bucket}/glue_jobs/job_name.py"
    python_version  = "3"
  }

  default_arguments = {
    "--job-language"        = "python"
    "--enable-metrics"      = "true"
    "--enable-job-insights" = "true"
  }

  worker_type       = "G.1X"
  number_of_workers = 2
  glue_version      = "4.0"
}
```

### 3. Unit Tests

```python
# generated/tests/test_job_name.py

"""Unit tests for job_name Glue job."""

import pytest
from pyspark.sql import SparkSession
from unittest.mock import Mock, patch

class TestJobName:

    @pytest.fixture
    def spark(self):
        return SparkSession.builder.master("local[1]").getOrCreate()

    def test_transform_logic(self, spark):
        # Test transformation logic
        input_df = spark.createDataFrame([...])
        # ... test assertions ...

    def test_null_handling(self, spark):
        # Test NULL value handling
        pass
```

### 4. Documentation

```markdown
# generated/docs/job_name.md

# Job: job_name

## Overview
- **DataStage Job**: JOB_NAME
- **Migration Category**: SEMI-AUTO
- **Confidence**: 85%

## Stage Mapping

| DataStage Stage | Type | Glue Equivalent |
|----------------|------|-----------------|
| SRC_ORACLE | OracleConnector | JDBC Connection |
| TRX_MAIN | Transformer | PySpark transforms |
| TGT_FILE | SequentialFile | S3 write |

## Transformation Details

### TRX_MAIN (Transformer)
Original DataStage derivations:
```
OUT.FULL_NAME = IN.FIRST_NAME || ' ' || IN.LAST_NAME
OUT.STATUS = If IN.ACTIVE = 1 Then 'ACTIVE' Else 'INACTIVE'
```

Converted to PySpark:
```python
df = df.withColumn("FULL_NAME", concat(col("FIRST_NAME"), lit(" "), col("LAST_NAME")))
df = df.withColumn("STATUS", when(col("ACTIVE") == 1, "ACTIVE").otherwise("INACTIVE"))
```

## Manual Review Required
- [ ] Verify JDBC connection string
- [ ] Confirm S3 bucket permissions
- [ ] Test with sample data
```

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] LLM abstraction layer with Anthropic provider
- [ ] Basic caching (exact match)
- [ ] Rule-based generator for AUTO jobs
- [ ] Glue script output

### Phase 2: Multi-Provider (Week 3)
- [ ] Azure OpenAI provider
- [ ] AWS Bedrock provider
- [ ] GCP Vertex provider
- [ ] OpenRouter provider
- [ ] Provider factory and config

### Phase 3: LLM Generation (Week 4-5)
- [ ] LLM-based generator for MANUAL jobs
- [ ] Hybrid generator for SEMI-AUTO jobs
- [ ] Prompt templates and optimization
- [ ] Semantic caching

### Phase 4: Outputs (Week 6)
- [ ] Terraform generator
- [ ] CloudFormation generator
- [ ] Unit test generator
- [ ] Documentation generator

### Phase 5: Polish (Week 7-8)
- [ ] Interactive mode
- [ ] Batch optimization
- [ ] Token usage reporting
- [ ] End-to-end testing

## Cost Estimation

Based on 12,000 jobs:
- AUTO (70%): 8,400 jobs → $0 (rule-based)
- SEMI-AUTO (20%): 2,400 jobs → ~$20-50 (partial LLM)
- MANUAL (10%): 1,200 jobs → ~$30-80 (full LLM)

**Estimated total**: $50-130 with caching enabled

With 95% cache hit rate on duplicates:
- Effective LLM calls: ~500-800
- **Optimized cost**: $15-40

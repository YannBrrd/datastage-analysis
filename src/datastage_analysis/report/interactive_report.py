"""
Interactive Report Module

Generates interactive reports using Streamlit.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class InteractiveReport:
    """Generates interactive reports for the analysis results."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    async def generate_report(self, jobs: List[Any], structural_clusters: Dict[str, List[int]],
                            semantic_clusters: Dict[int, List[int]], comparisons: List[Any]) -> None:
        """
        Generate the complete interactive report.

        Args:
            jobs: List of parsed jobs
            structural_clusters: Structural clustering results
            semantic_clusters: Semantic clustering results
            comparisons: Comparison results from Claude
        """
        logger.info("Generating interactive report...")

        # Save data for Streamlit app
        self._save_report_data(jobs, structural_clusters, semantic_clusters, comparisons)

        # Create Streamlit app
        self._create_streamlit_app()

        logger.info(f"Report generated in {self.output_dir}")

    def _save_report_data(self, jobs: List[Any], structural_clusters: Dict[str, List[int]],
                         semantic_clusters: Dict[int, List[int]], comparisons: List[Any]) -> None:
        """Save analysis data for the report."""
        # Convert jobs to DataFrame
        job_data = []
        for i, job in enumerate(jobs):
            job_data.append({
                'index': i,
                'name': job.name,
                'path': str(job.path),
                'fingerprint': job.fingerprint,
                'content_hash': job.content_hash
            })
        jobs_df = pd.DataFrame(job_data)
        jobs_df.to_csv(self.output_dir / 'jobs.csv', index=False)

        # Structural clusters
        structural_data = []
        for fingerprint, indices in structural_clusters.items():
            for idx in indices:
                structural_data.append({
                    'job_index': idx,
                    'cluster_fingerprint': fingerprint,
                    'cluster_size': len(indices)
                })
        structural_df = pd.DataFrame(structural_data)
        structural_df.to_csv(self.output_dir / 'structural_clusters.csv', index=False)

        # Semantic clusters
        semantic_data = []
        for cluster_id, indices in semantic_clusters.items():
            for idx in indices:
                semantic_data.append({
                    'job_index': idx,
                    'semantic_cluster': cluster_id,
                    'cluster_size': len(indices)
                })
        semantic_df = pd.DataFrame(semantic_data)
        semantic_df.to_csv(self.output_dir / 'semantic_clusters.csv', index=False)

        # Comparisons
        comparison_data = []
        for comp in comparisons:
            comparison_data.append({
                'job1_index': comp.job1_idx,
                'job2_index': comp.job2_idx,
                'similarity_score': comp.similarity_score,
                'differences': '; '.join(comp.differences),
                'reasoning': comp.reasoning
            })
        comparisons_df = pd.DataFrame(comparison_data)
        comparisons_df.to_csv(self.output_dir / 'comparisons.csv', index=False)

    def _create_streamlit_app(self) -> None:
        """Create the Streamlit application file."""
        app_code = '''
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="DataStage Job Analysis", layout="wide")

st.title("DataStage Job Comparison Analysis")

# Load data
@st.cache_data
def load_data():
    data_dir = Path(".")
    jobs_df = pd.read_csv(data_dir / "jobs.csv")
    structural_df = pd.read_csv(data_dir / "structural_clusters.csv")
    semantic_df = pd.read_csv(data_dir / "semantic_clusters.csv")
    comparisons_df = pd.read_csv(data_dir / "comparisons.csv")
    return jobs_df, structural_df, semantic_df, comparisons_df

jobs_df, structural_df, semantic_df, comparisons_df = load_data()

# Overview
st.header("Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Jobs", len(jobs_df))
with col2:
    st.metric("Structural Clusters", structural_df['cluster_fingerprint'].nunique())
with col3:
    st.metric("Semantic Clusters", semantic_df['semantic_cluster'].nunique())

# Structural Clustering
st.header("Structural Clustering")
fig1 = px.histogram(structural_df, x='cluster_size', title="Structural Cluster Size Distribution")
st.plotly_chart(fig1)

# Semantic Clustering
st.header("Semantic Clustering")
fig2 = px.histogram(semantic_df, x='cluster_size', title="Semantic Cluster Size Distribution")
st.plotly_chart(fig2)

# Comparisons
st.header("Job Comparisons")
if not comparisons_df.empty:
    fig3 = px.histogram(comparisons_df, x='similarity_score', title="Similarity Score Distribution")
    st.plotly_chart(fig3)

    # Show top similar pairs
    st.subheader("Most Similar Job Pairs")
    top_similar = comparisons_df.nlargest(10, 'similarity_score')
    st.dataframe(top_similar)

# Job Details
st.header("Job Details")
selected_job = st.selectbox("Select a job to view details:", jobs_df['name'].tolist())
job_info = jobs_df[jobs_df['name'] == selected_job].iloc[0]

st.write(f"**Path:** {job_info['path']}")
st.write(f"**Fingerprint:** {job_info['fingerprint']}")
st.write(f"**Content Hash:** {job_info['content_hash']}")

# Show cluster memberships
job_idx = job_info['index']
structural_cluster = structural_df[structural_df['job_index'] == job_idx]
if not structural_cluster.empty:
    st.write(f"**Structural Cluster:** {structural_cluster.iloc[0]['cluster_fingerprint']} "
             f"(size: {structural_cluster.iloc[0]['cluster_size']})")

semantic_cluster = semantic_df[semantic_df['job_index'] == job_idx]
if not semantic_cluster.empty:
    st.write(f"**Semantic Cluster:** {semantic_cluster.iloc[0]['semantic_cluster']} "
             f"(size: {semantic_cluster.iloc[0]['cluster_size']})")
'''

        app_file = self.output_dir / 'app.py'
        app_file.write_text(app_code)

        # Create requirements for the app
        requirements = '''
streamlit
pandas
plotly
'''
        (self.output_dir / 'requirements.txt').write_text(requirements)
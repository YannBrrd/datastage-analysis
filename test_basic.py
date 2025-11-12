"""
Basic tests for the DataStage analysis system.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datastage_analysis.parsers.dsx_parser import DSXParser, DataStageJob


def test_parser():
    """Test the DSX parser with a mock job."""
    parser = DSXParser()

    # Create a mock job
    mock_structure = {
        'name': 'TestJob',
        'stages': [
            {'name': 'Source', 'type': 'SequentialFile', 'properties': {'FileName': 'input.txt'}},
            {'name': 'Target', 'type': 'SequentialFile', 'properties': {'FileName': 'output.txt'}}
        ],
        'links': [{'from': 'Source', 'to': 'Target', 'name': 'link1'}]
    }

    mock_job = DataStageJob(
        name='TestJob',
        path=Path('test.dsx'),
        structure=mock_structure,
        fingerprint='test_fingerprint',
        content_hash='test_hash'
    )

    fingerprint = parser.extract_fingerprint(mock_job)
    assert fingerprint == 'test_fingerprint'
    print("✓ Parser test passed")


async def test_embedder():
    """Test the semantic embedder."""
    from datastage_analysis.embeddings.semantic_embedder import SemanticEmbedder

    embedder = SemanticEmbedder()

    # Mock jobs
    jobs = [DataStageJob(
        name=f'Job{i}',
        path=Path(f'job{i}.dsx'),
        structure={'stages': [{'name': f'Stage{i}', 'type': 'SequentialFile'}]},
        fingerprint=f'fp{i}',
        content_hash=f'hash{i}'
    ) for i in range(3)]

    embeddings = await embedder.generate_embeddings(jobs)
    assert embeddings.shape[0] == 3
    print("✓ Embedder test passed")


def test_structural_clusterer():
    """Test structural clustering."""
    from datastage_analysis.clustering.structural_clusterer import StructuralClusterer

    clusterer = StructuralClusterer()
    fingerprints = ['fp1', 'fp1', 'fp2', 'fp3', 'fp3', 'fp3']

    clusters = clusterer.cluster_by_hash(fingerprints)
    assert len(clusters) == 3  # 3 unique fingerprints
    assert len(clusters['fp1']) == 2
    assert len(clusters['fp3']) == 3
    print("✓ Structural clusterer test passed")


async def main():
    """Run all tests."""
    print("Running basic tests...")

    test_parser()
    await test_embedder()
    test_structural_clusterer()

    print("All tests passed! ✓")


if __name__ == "__main__":
    asyncio.run(main())
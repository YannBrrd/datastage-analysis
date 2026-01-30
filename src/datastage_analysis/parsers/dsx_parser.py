"""
DSX Parser Module

Parses DataStage DSX/XML files to extract job structures and fingerprints.
"""

import asyncio
import hashlib
import logging
import gzip
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor
import xml.etree.ElementTree as ET
from dataclasses import dataclass

try:
    from ..config import get_config
    _config = get_config()
except ImportError:
    _config = None

logger = logging.getLogger(__name__)


@dataclass
class DataStageJob:
    """Represents a parsed DataStage job."""
    name: str
    path: Path
    structure: Dict[str, Any]
    fingerprint: str
    content_hash: str


class DSXParser:
    """Parser for DataStage DSX files."""

    def __init__(self, max_workers: int = None, max_file_size_mb: int = None):
        # Load from config if not provided
        if _config:
            parser_config = _config.parser
            self.max_workers = max_workers or parser_config.get('max_workers', 4)
            self.max_file_size_mb = max_file_size_mb or parser_config.get('max_file_size_mb', 510)
            self.max_lines = parser_config.get('max_lines', 500000)
        else:
            self.max_workers = max_workers or 4
            self.max_file_size_mb = max_file_size_mb or 510
            self.max_lines = 500000

    async def parse_all_jobs(self, data_dir: Path) -> List[DataStageJob]:
        """Parse all DataStage files in the data directory asynchronously."""
        # Find .dsx, .dsx.gz, .xml, and .xml.gz files recursively
        dsx_files = list(data_dir.glob("**/*.dsx"))
        dsx_gz_files = list(data_dir.glob("**/*.dsx.gz"))
        xml_files = list(data_dir.glob("**/*.xml"))
        xml_gz_files = list(data_dir.glob("**/*.xml.gz"))
        all_files = dsx_files + dsx_gz_files + xml_files + xml_gz_files
        logger.info(f"Found {len(dsx_files)} .dsx, {len(dsx_gz_files)} .dsx.gz, {len(xml_files)} .xml, {len(xml_gz_files)} .xml.gz (total: {len(all_files)})")

        # Use multiprocessing for CPU-intensive XML parsing
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, self._parse_single_job, file_path)
                for file_path in all_files
            ]
            jobs = await asyncio.gather(*tasks)

        return [job for job in jobs if job is not None]

    def _parse_single_job(self, file_path: Path) -> Optional[DataStageJob]:
        """Parse a single DataStage file (supports .dsx, .dsx.gz, .xml, .xml.gz)."""
        try:
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                logger.warning(f"Large file detected: {file_path.name} ({file_size_mb:.1f} MB)")

            # Determine if gzip compressed
            is_gzip = file_path.suffix == '.gz' or str(file_path).endswith('.gz')

            # Determine if it's explicitly XML format
            is_xml_file = '.xml' in file_path.suffixes or file_path.name.endswith('.xml.gz')

            # Try to determine if it's native DSX format or XML
            if is_gzip:
                with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    first_line = f.readline().strip()
                    f.seek(0)

                    if first_line == "BEGIN HEADER" and not is_xml_file:
                        # Native DSX format
                        return self._parse_native_dsx(f, file_path)
                    else:
                        # XML parsing (either .xml.gz or XML-format .dsx.gz)
                        return self._parse_xml_dsx(f, file_path, is_gzip=True)
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    first_line = f.readline().strip()
                    f.seek(0)

                    if first_line == "BEGIN HEADER" and not is_xml_file:
                        # Native DSX format
                        return self._parse_native_dsx(f, file_path)
                    else:
                        # XML parsing
                        return self._parse_xml_dsx(f, file_path, is_gzip=False)

            # Extract job structure (simplified)
            structure = self._extract_job_structure(root)

            # Generate fingerprint
            fingerprint = self._generate_fingerprint(structure)

            job_name = structure.get('name', file_path.stem)

            return DataStageJob(
                name=job_name,
                path=file_path,
                structure=structure,
                fingerprint=fingerprint,
                content_hash=content_hash
            )

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return None

    def _parse_xml_dsx(self, file_obj, file_path: Path, is_gzip: bool) -> Optional[DataStageJob]:
        """Parse XML format DSX."""
        try:
            root = ET.parse(file_obj).getroot()
            structure = self._extract_job_structure(root)
            content_hash = self._calculate_hash_incremental(file_path, is_gzip=is_gzip)
            fingerprint = self._generate_fingerprint(structure)
            job_name = structure.get('name', file_path.stem)
            
            return DataStageJob(
                name=job_name,
                path=file_path,
                structure=structure,
                fingerprint=fingerprint,
                content_hash=content_hash
            )
        except Exception as e:
            logger.error(f"XML parsing failed for {file_path}: {e}")
            return None

    def _parse_native_dsx(self, file_obj, file_path: Path) -> Optional[DataStageJob]:
        """Parse native IBM DataStage DSX format."""
        try:
            structure = {'name': file_path.stem, 'stages': [], 'links': [], 'jobs': []}
            current_section = None
            current_job = None
            current_record = None
            job_count = 0
            stage_count = 0
            line_count = 0
            max_lines = self.max_lines  # Configurable via config.yaml
            in_stage_record = False
            
            for line in file_obj:
                line = line.strip()
                line_count += 1
                
                # Limit parsing for huge files
                if line_count > max_lines:
                    logger.warning(f"Reached max lines ({max_lines}) for {file_path.name}, stopping parse")
                    break
                
                # Track job sections
                if line.startswith("BEGIN DSJOB"):
                    job_count += 1
                    current_job = {'name': '', 'stages': [], 'links': [], 'stage_types': []}
                    current_section = 'JOB'
                    
                elif line.startswith("END DSJOB"):
                    if current_job and current_job['name']:
                        structure['jobs'].append(current_job)
                        logger.debug(f"Found job: {current_job['name']} with {len(current_job['stage_types'])} stage types")
                    current_job = None
                    current_section = None
                    
                # Track DSRECORD sections inside jobs
                elif line.startswith("BEGIN DSRECORD") and current_job:
                    current_record = {'name': '', 'type': '', 'properties': {}}
                    in_stage_record = False
                    
                elif line.startswith("END DSRECORD") and current_job:
                    # Check if this record represents a stage
                    if current_record and current_record.get('type'):
                        # This is likely a stage definition
                        current_job['stages'].append({
                            'name': current_record.get('name', ''),
                            'type': current_record.get('type', ''),
                            'properties': current_record.get('properties', {})
                        })
                        stage_count += 1
                    current_record = None
                    in_stage_record = False
                    
                # Extract identifiers
                elif line.startswith("Identifier") and '"' in line and current_job:
                    name = line.split('"')[1] if len(line.split('"')) > 1 else ''
                    if not current_job['name'] and not current_record:
                        # This is the job name
                        current_job['name'] = name
                    elif current_record:
                        # This is a record/stage name
                        current_record['name'] = name
                        
                # Extract stage types (stored as OLEType or similar)
                elif current_record and line.startswith("OLEType") and '"' in line:
                    stage_type = line.split('"')[1] if len(line.split('"')) > 1 else ''
                    current_record['type'] = stage_type
                    current_job['stage_types'].append(stage_type)
                    in_stage_record = True
                    
                # Extract properties
                elif current_record and in_stage_record and '=' in line and '"' in line:
                    parts = line.split('"')
                    if len(parts) >= 3:
                        prop_name = parts[0].strip()
                        prop_value = parts[1]
                        current_record['properties'][prop_name] = prop_value
            
            # Use first job if multiple found, or aggregate all jobs
            if structure['jobs']:
                if len(structure['jobs']) == 1:
                    # Single job - use it
                    first_job = structure['jobs'][0]
                    structure['name'] = first_job['name']
                    structure['stages'] = first_job['stages']
                    structure['links'] = first_job['links']
                else:
                    # Multiple jobs - aggregate stages from all
                    structure['name'] = f"{file_path.stem}_MultiJob"
                    for job in structure['jobs']:
                        structure['stages'].extend(job['stages'])
                        structure['links'].extend(job['links'])
                    logger.info(f"Aggregated {len(structure['jobs'])} jobs from {file_path.name}")
            
            total_stages = len(structure['stages'])
            total_links = len(structure['links'])
            
            # Calculate hash and fingerprint
            content_hash = self._calculate_hash_incremental(file_path, is_gzip=file_path.suffix == '.gz')
            fingerprint = self._generate_fingerprint(structure)
            
            logger.info(f"Parsed native DSX: {structure['name']} with {total_stages} stages, {total_links} links, {len(structure['jobs'])} jobs")
            
            return DataStageJob(
                name=structure['name'] or file_path.stem,
                path=file_path,
                structure=structure,
                fingerprint=fingerprint,
                content_hash=content_hash
            )
        except Exception as e:
            logger.error(f"Native DSX parsing failed for {file_path}: {e}")
            return None

    def _calculate_hash_incremental(self, file_path: Path, is_gzip: bool, chunk_size: int = 8192) -> str:
        """Calculate file hash incrementally without loading full content in memory."""
        hash_obj = hashlib.sha256()
        
        try:
            if is_gzip:
                with gzip.open(file_path, 'rb') as f:
                    while chunk := f.read(chunk_size):
                        hash_obj.update(chunk)
            else:
                with open(file_path, 'rb') as f:
                    while chunk := f.read(chunk_size):
                        hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate hash for {file_path}: {e}")
            # Return a placeholder hash based on file size and name
            return hashlib.sha256(f"{file_path.name}:{file_path.stat().st_size}".encode()).hexdigest()

    def _extract_job_structure(self, root: ET.Element) -> Dict[str, Any]:
        """Extract job structure from XML root."""
        # Simplified structure extraction
        # In real DataStage, this would parse stages, links, parameters, etc.
        structure = {}

        # Find job elements
        for job in root.findall(".//JOB"):
            structure['name'] = job.get('NAME', '')
            structure['stages'] = []

            # Extract stages
            for stage in job.findall(".//STAGE"):
                stage_info = {
                    'name': stage.get('NAME', ''),
                    'type': stage.get('STGTYPE', ''),
                    'properties': {}
                }

                # Extract properties
                for prop in stage.findall(".//PROPERTY"):
                    name = prop.get('NAME', '')
                    value = prop.text or ''
                    stage_info['properties'][name] = value

                structure['stages'].append(stage_info)

            # Extract links
            structure['links'] = []
            for link in job.findall(".//LINK"):
                link_info = {
                    'from': link.get('FROMSTAGE', ''),
                    'to': link.get('TOSTAGE', ''),
                    'name': link.get('NAME', '')
                }
                structure['links'].append(link_info)

        return structure

    def _generate_fingerprint(self, structure: Dict[str, Any]) -> str:
        """Generate a structural fingerprint for the job."""
        # Create a normalized string representation for hashing
        fingerprint_parts = []

        # Sort stages by name for consistency
        stages = sorted(structure.get('stages', []), key=lambda x: x['name'])
        for stage in stages:
            fingerprint_parts.append(f"{stage['type']}:{stage['name']}")
            # Include key properties
            props = stage.get('properties', {})
            for key in sorted(props.keys()):
                if key in ['SQL', 'FileName', 'TableName']:  # Key identifying properties
                    fingerprint_parts.append(f"{key}:{props[key]}")

        # Sort links
        links = sorted(structure.get('links', []), key=lambda x: (x['from'], x['to']))
        for link in links:
            fingerprint_parts.append(f"link:{link['from']}->{link['to']}")

        fingerprint_str = '|'.join(fingerprint_parts)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()

    def extract_fingerprint(self, job: DataStageJob) -> str:
        """Extract fingerprint from a job object."""
        return job.fingerprint
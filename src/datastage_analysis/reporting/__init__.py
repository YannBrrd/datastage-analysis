"""
Reporting Module

Generates various report formats for migration analysis.
"""

from .html_report import generate_html_report

__all__ = [
    'generate_html_report',
]

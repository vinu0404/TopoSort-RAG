"""
Chunking utilities for regex-based section detection.

Provides pattern-based section parsing for PDF/DOCX documents
and graceful handling of unstructured content (CSV, plain text).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


class SectionParser:
    """Regex-based section detection for structured documents."""
    
    PATTERNS = {
        # Markdown-style headers: # Header, ## Subheader, ### Sub-subheader
        'markdown': re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE),
        
        # Numbered sections: "1. Introduction", "1.1 Background", "Section 2.3: Methods"
        'numbered': re.compile(
            r'^(?:(?:Section|Chapter|Part|Article)\s+)?'
            r'(\d+(?:\.\d+)*)\s*[:\-\.]?\s*(.+)$',
            re.MULTILINE | re.IGNORECASE
        ),
        
        # ALL CAPS headers (minimum 3 chars to avoid false positives)
        'caps': re.compile(r'^([A-Z][A-Z\s]{2,}[A-Z])$', re.MULTILINE),
        
        # Underlined headers (text followed by === or ---)
        'underlined': re.compile(r'^(.+)\n([=\-]{3,})$', re.MULTILINE),
        
        # Roman numerals: "I. Introduction", "II. Methods"
        'roman': re.compile(
            r'^([IVXLCDM]+)\.\s+(.+)$',
            re.MULTILINE
        ),
    }
    
    # Table detection patterns
    TABLE_PATTERNS = {
        # Pipe-delimited tables: | col1 | col2 |
        'pipe': re.compile(r'(?:^\s*\|.+\|\s*$\n){3,}', re.MULTILINE),
        
        # Tab-delimited tables (at least 3 rows with tabs)
        'tab': re.compile(r'(?:^.+\t.+$\n){3,}', re.MULTILINE),
        
        # CSV-like content (comma-separated, at least 3 rows)
        'csv_like': re.compile(r'(?:^[^,]+,[^,]+.+$\n){3,}', re.MULTILINE),
    }

    @classmethod
    def parse_sections(
        cls,
        content: str,
        doc_type: str = "pdf"
    ) -> List[Dict[str, Any]]:
        """
        Parse document content into sections using regex patterns.
        
        Args:
            content: Document text content
            doc_type: Document type (pdf, docx, csv, txt, etc.)
            
        Returns:
            List of section dicts with format:
            [
                {"type": "section", "title": "...", "content": "...", "level": 1},
                {"type": "table", "content": "...", "description": "..."},
                ...
            ]
        """
        if doc_type in ('csv', 'xls', 'xlsx'):
            return cls._parse_tabular_content(content)
        
        sections = cls._detect_sections(content)
        
        if not sections:
            return [{"type": "section", "title": "", "content": content, "level": 0}]
        
        return sections

    @classmethod
    def _detect_sections(cls, content: str) -> List[Dict[str, Any]]:
        """
        Detect sections using multiple regex patterns.
        Returns sections with their positions in the document.
        """
        # First, detect and extract tables
        tables = cls._extract_tables(content)
        
        # Find all potential section headers with their positions
        headers = cls._find_all_headers(content)
        
        if not headers:
            # No headers found, return content as single section (minus tables)
            clean_content = cls._remove_tables(content, tables)
            if clean_content.strip():
                sections = [{"type": "section", "title": "", "content": clean_content, "level": 0}]
            else:
                sections = []
        else:
            # Build sections from headers
            sections = cls._build_sections_from_headers(content, headers, tables)
        
        # Add tables as separate sections
        for table_info in tables:
            sections.append({
                "type": "table",
                "content": table_info['content'],
                "description": cls._generate_table_description(table_info['content']),
            })
        
        return sections

    @classmethod
    def _find_all_headers(cls, content: str) -> List[Tuple[int, str, int]]:
        """
        Find all headers in the content.
        
        Returns:
            List of (position, title, level) tuples, sorted by position
        """
        headers = []
        
        # Try markdown headers
        for match in cls.PATTERNS['markdown'].finditer(content):
            level = len(match.group(1))  # Number of # symbols
            title = match.group(2).strip()
            headers.append((match.start(), title, level))
        
        # Try numbered sections
        for match in cls.PATTERNS['numbered'].finditer(content):
            number = match.group(1)
            title = match.group(2).strip()
            # Level based on number of dots (1 = level 1, 1.1 = level 2, etc.)
            level = number.count('.') + 1
            headers.append((match.start(), title, level))
        
        # Try ALL CAPS headers (only if not too many to avoid false positives)
        caps_matches = list(cls.PATTERNS['caps'].finditer(content))
        if len(caps_matches) < 50:  # Avoid treating too many lines as headers
            for match in caps_matches:
                title = match.group(1).strip()
                # Skip very short caps (likely acronyms)
                if len(title) >= 8:
                    headers.append((match.start(), title, 1))
        
        # Try underlined headers
        for match in cls.PATTERNS['underlined'].finditer(content):
            title = match.group(1).strip()
            underline_char = match.group(2)[0]
            # = is typically level 1, - is level 2
            level = 1 if underline_char == '=' else 2
            headers.append((match.start(), title, level))
        
        # Try Roman numerals
        for match in cls.PATTERNS['roman'].finditer(content):
            title = match.group(2).strip()
            headers.append((match.start(), title, 1))
        
        # Sort by position and remove duplicates (same position)
        headers = sorted(set(headers), key=lambda x: x[0])
        
        return headers

    @classmethod
    def _build_sections_from_headers(
        cls,
        content: str,
        headers: List[Tuple[int, str, int]],
        tables: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build section objects from detected headers.
        """
        sections = []
        
        for i, (pos, title, level) in enumerate(headers):
            # Determine section content (from this header to next header or end)
            start_pos = pos
            if i < len(headers) - 1:
                end_pos = headers[i + 1][0]
            else:
                end_pos = len(content)
            
            section_content = content[start_pos:end_pos]
            
            # Remove the header line itself from content
            lines = section_content.split('\n')
            if lines:
                section_content = '\n'.join(lines[1:])
            
            # Remove any tables from this section (they'll be added separately)
            section_content = cls._remove_tables(section_content, tables)
            
            if section_content.strip():
                sections.append({
                    "type": "section",
                    "title": title,
                    "content": section_content.strip(),
                    "level": level,
                })
        
        return sections

    @classmethod
    def _extract_tables(cls, content: str) -> List[Dict[str, Any]]:
        """
        Find and extract tables from content.
        
        Returns:
            List of dicts with 'content' and 'start'/'end' positions
        """
        tables = []
        
        # Check each table pattern
        for pattern_name, pattern in cls.TABLE_PATTERNS.items():
            for match in pattern.finditer(content):
                tables.append({
                    'content': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                    'pattern': pattern_name,
                })
        
        # Remove overlapping tables (keep the longest match)
        tables = cls._remove_overlapping_tables(tables)
        
        return tables

    @staticmethod
    def _remove_overlapping_tables(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping table matches, keeping the longest."""
        if not tables:
            return []
        
        # Sort by start position
        tables = sorted(tables, key=lambda t: t['start'])
        
        non_overlapping = [tables[0]]
        
        for table in tables[1:]:
            last = non_overlapping[-1]
            
            # Check if overlaps with last added table
            if table['start'] < last['end']:
                # Keep the longer table
                if (table['end'] - table['start']) > (last['end'] - last['start']):
                    non_overlapping[-1] = table
            else:
                non_overlapping.append(table)
        
        return non_overlapping

    @staticmethod
    def _remove_tables(content: str, tables: List[Dict[str, Any]]) -> str:
        """Remove table content from text."""
        if not tables:
            return content
        
        # Sort tables by position (reverse order to maintain positions)
        tables_sorted = sorted(tables, key=lambda t: t['start'], reverse=True)
        
        for table in tables_sorted:
            content = content[:table['start']] + content[table['end']:]
        
        return content

    @staticmethod
    def _generate_table_description(table_content: str) -> str:
        """Generate a simple description for a table."""
        lines = table_content.strip().split('\n')
        num_rows = len(lines)
        
        # Try to count columns (use first line)
        if lines:
            first_line = lines[0]
            if '|' in first_line:
                num_cols = first_line.count('|') - 1
            elif '\t' in first_line:
                num_cols = first_line.count('\t') + 1
            elif ',' in first_line:
                num_cols = first_line.count(',') + 1
            else:
                num_cols = 1
        else:
            num_cols = 1
        
        return f"Table with approximately {num_rows} rows and {num_cols} columns"

    @classmethod
    def _parse_tabular_content(cls, content: str) -> List[Dict[str, Any]]:
        """
        Parse CSV/Excel content - treat the entire content as a table.
        """
        # Check if content looks like a table
        lines = content.strip().split('\n')
        
        if len(lines) < 2:
            # Too short to be a meaningful table, treat as regular content
            return [{"type": "section", "title": "", "content": content, "level": 0}]
        
        # Check for common delimiters
        first_line = lines[0]
        has_delimiters = ',' in first_line or '\t' in first_line or '|' in first_line
        
        if has_delimiters:
            return [{
                "type": "table",
                "content": content,
                "description": cls._generate_table_description(content),
            }]
        else:
            # Not clearly tabular, treat as regular content
            return [{"type": "section", "title": "", "content": content, "level": 0}]


def detect_document_structure(content: str, doc_type: str = "pdf") -> Dict[str, Any]:
    """
    Analyze document structure and return metadata.
    
    Args:
        content: Document text
        doc_type: Document type
        
    Returns:
        Dict with structure info: {
            "has_sections": bool,
            "num_sections": int,
            "num_tables": int,
            "header_style": str,  # "markdown", "numbered", "caps", etc.
        }
    """
    parser = SectionParser()
    sections = parser.parse_sections(content, doc_type)
    
    section_count = sum(1 for s in sections if s.get("type") == "section")
    table_count = sum(1 for s in sections if s.get("type") == "table")
    header_style = "none"
    if section_count > 1:
        headers = parser._find_all_headers(content)
        if headers:
            if content[headers[0][0]:headers[0][0]+10].startswith('#'):
                header_style = "markdown"
            elif re.match(r'\d+\.', content[headers[0][0]:headers[0][0]+20]):
                header_style = "numbered"
            elif content[headers[0][0]:headers[0][0]+20].isupper():
                header_style = "caps"
            else:
                header_style = "mixed"
    
    return {
        "has_sections": section_count > 1,
        "num_sections": section_count,
        "num_tables": table_count,
        "header_style": header_style,
    }
"""
Local File Reader Tool for CrewAI - Retrieve template files and other local resources
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Type, Union, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class LocalFileInput(BaseModel):
    """Input model for local file reading"""
    file_path: str = Field(..., description="Path to the local file to read")
    encoding: str = Field(default="utf-8", description="File encoding (default: utf-8)")
    return_format: str = Field(default="content", description="Return format: 'content', 'json', or 'info'")


class LocalFileReaderTool(BaseTool):
    """
    Tool to read local files, particularly useful for loading template files,
    configuration files, or any local text resources.
    """
    
    name: str = "Local File Reader"
    description: str = """
        Reads local files and returns their content. Particularly useful for:
        - Loading test templates (e.g., pytest templates)
        - Reading configuration files
        - Accessing code templates
        - Loading any text-based local resources
        
        Supports multiple return formats:
        - 'content': Returns raw file content
        - 'json': Parses and returns JSON content
        - 'info': Returns file metadata along with content
    """
    args_schema: Type[BaseModel] = LocalFileInput
    base_path: Path = Field(default_factory=Path.cwd, description="Base directory for resolving relative paths")

    def __init__(self, base_path: Optional[str] = None, **kwargs):
        """
        Initialize the tool with an optional base path for relative file access.
        
        Args:
            base_path: Base directory for resolving relative paths (default: current directory)
        """
        if base_path:
            kwargs['base_path'] = Path(base_path)
        super().__init__(**kwargs)

    def _run(self, file_path: str, encoding: str = "utf-8", return_format: str = "content") -> str:
        """
        Read a local file and return its content or metadata.
        
        Args:
            file_path: Path to the file (can be absolute or relative to base_path)
            encoding: File encoding to use
            return_format: Format for return value ('content', 'json', or 'info')
        
        Returns:
            File content or information based on return_format
        """
        try:
            # Resolve the file path
            resolved_path = self._resolve_path(file_path)
            
            # Validate the file exists and is readable
            self._validate_file(resolved_path)
            
            # Read the file based on return format
            if return_format == "json":
                return self._read_json_file(resolved_path)
            elif return_format == "info":
                return self._read_file_with_info(resolved_path, encoding)
            else:  # default to 'content'
                return self._read_file_content(resolved_path, encoding)
                
        except Exception as e:
            return json.dumps({
                "error": f"Error reading file: {str(e)}",
                "file_path": file_path,
                "resolved_path": str(resolved_path) if 'resolved_path' in locals() else None
            }, indent=2)

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve file path relative to base_path if needed"""
        path = Path(file_path)
        
        # If absolute path, use as-is
        if path.is_absolute():
            return path
        
        # Otherwise, resolve relative to base_path
        return self.base_path / path

    def _validate_file(self, file_path: Path):
        """Validate that the file exists and is readable"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"File is not readable: {file_path}")

    def _read_file_content(self, file_path: Path, encoding: str) -> str:
        """Read and return raw file content"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # Try alternative encodings
            for alt_encoding in ['latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=alt_encoding) as f:
                        content = f.read()
                        print(f"Successfully read file using {alt_encoding} encoding")
                        return content
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Unable to decode file with any supported encoding")

    def _read_json_file(self, file_path: Path) -> str:
        """Read and parse JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return json.dumps(data, indent=2)
        except json.JSONDecodeError as e:
            return json.dumps({
                "error": f"Invalid JSON: {str(e)}",
                "file_path": str(file_path)
            }, indent=2)

    def _read_file_with_info(self, file_path: Path, encoding: str) -> str:
        """Read file and return content with metadata"""
        content = self._read_file_content(file_path, encoding)
        
        # Get file stats
        stats = file_path.stat()
        
        # Determine file type
        file_extension = file_path.suffix.lower()
        file_type = self._determine_file_type(file_extension)
        
        # Count lines and characters
        lines = content.split('\n')
        
        info = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_type": file_type,
            "extension": file_extension,
            "size_bytes": stats.st_size,
            "size_readable": self._format_size(stats.st_size),
            "modified_timestamp": stats.st_mtime,
            "lines": len(lines),
            "characters": len(content),
            "encoding_used": encoding,
            "content": content
        }
        
        return json.dumps(info, indent=2)

    def _determine_file_type(self, extension: str) -> str:
        """Determine file type based on extension"""
        type_mapping = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.txt': 'text',
            '.md': 'markdown',
            '.rst': 'restructuredtext',
            '.html': 'html',
            '.xml': 'xml',
            '.csv': 'csv',
            '.toml': 'toml',
            '.ini': 'ini',
            '.cfg': 'config',
            '.conf': 'config',
            '.sh': 'shell',
            '.bash': 'bash',
            '.sql': 'sql',
            '.template': 'template'
        }
        return type_mapping.get(extension, 'unknown')

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"


class TemplateFileReaderTool(LocalFileReaderTool):
    """
    Specialized version of LocalFileReaderTool for reading template files.
    Automatically looks in common template directories.
    """
    
    name: str = "Template File Reader"
    description: str = """
        Specialized tool for reading template files. Automatically searches in:
        - templates/
        - test_templates/
        - .templates/
        - tests/templates/
        
        Perfect for loading pytest templates, code generation templates, etc.
    """
    
    template_dirs: List[str] = Field(
        default_factory=lambda: [
            "templates",
            #"test_templates", 
            #".templates",
            #"tests/templates",
            #"src/templates"
        ],
        description="List of directories to search for templates"
    )
    
    def __init__(self, template_dirs: Optional[List[str]] = None, **kwargs):
        """
        Initialize with template directories to search.
        
        Args:
            template_dirs: List of directories to search for templates
        """
        if template_dirs:
            kwargs['template_dirs'] = template_dirs
        super().__init__(**kwargs)

    def _resolve_path(self, file_path: str) -> Path:
        """Override to search in template directories"""
        path = Path(file_path)
        
        # If absolute path, use as-is
        if path.is_absolute() and path.exists():
            return path
        
        # Search in template directories
        for template_dir in self.template_dirs:
            base_dir = self.base_path / template_dir
            if base_dir.exists():
                potential_path = base_dir / file_path
                if potential_path.exists():
                    return potential_path
        
        # Fall back to base implementation
        return super()._resolve_path(file_path)

    def list_templates(self, pattern: str = "*") -> str:
        """
        List available templates matching the pattern.
        
        Args:
            pattern: Glob pattern to match template files (default: "*")
        
        Returns:
            JSON string with list of available templates
        """
        templates = []
        
        for template_dir in self.template_dirs:
            base_dir = self.base_path / template_dir
            if base_dir.exists():
                for template_file in base_dir.glob(f"**/{pattern}"):
                    if template_file.is_file():
                        relative_path = template_file.relative_to(base_dir)
                        templates.append({
                            "name": template_file.stem,
                            "file": template_file.name,
                            "path": str(relative_path),
                            "full_path": str(template_file),
                            "type": self._determine_file_type(template_file.suffix)
                        })
        
        return json.dumps({
            "templates": templates,
            "count": len(templates),
            "search_pattern": pattern,
            "template_directories": self.template_dirs
        }, indent=2)


# Create tool instances for easy import
local_file_reader = LocalFileReaderTool
template_file_reader = TemplateFileReaderTool


# Example usage in CrewAI:
"""
# In your crew.py or main.py:
from test_generation_with_claude.tools.local_file_reader import TemplateFileReaderTool

# Create instance when needed
template_reader = TemplateFileReaderTool()

# Or with custom base path
template_reader = TemplateFileReaderTool(base_path="/path/to/project")
"""
from mcp.server.fastmcp import FastMCP
import hashlib
from pathlib import Path

mcp = FastMCP("FileSystemRAG")

@mcp.tool()
async def monitor_directory(path: str, file_types: list[str]) -> dict:
    """Monitor directory for new documents and return them for ingestion"""
    files = []
    for ext in file_types:
        files.extend(Path(path).glob(f"**/*.{ext}"))
    
    return {
        "files": [str(f) for f in files],
        "count": len(files),
        "hash": hashlib.md5(str(files).encode()).hexdigest()
    }

@mcp.tool()
async def get_file_metadata(filepath: str) -> dict:
    """Extract metadata from files for enhanced indexing"""
    path = Path(filepath)
    return {
        "size": path.stat().st_size,
        "modified": path.stat().st_mtime,
        "type": path.suffix,
        "name": path.name
    }

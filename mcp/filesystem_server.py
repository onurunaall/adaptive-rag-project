from mcp.server.fastmcp import FastMCP
import hashlib
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Any
import mimetypes
import os

mcp = FastMCP("FileSystemRAG")

# Cache for file tracking
file_cache = {}

@mcp.tool()
async def monitor_directory(path: str, file_types: list[str] = None, recursive: bool = True) -> dict:
    """
    Monitor directory for documents and track changes.
    
    Args:
        path: Directory path to monitor
        file_types: List of file extensions to track (e.g., ['pdf', 'txt', 'md'])
        recursive: Whether to search subdirectories
    
    Returns:
        Dictionary with file information and change status
    """
    if file_types is None:
        file_types = ['pdf', 'txt', 'md', 'docx', 'html']
    
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            return {"error": f"Path does not exist: {path}"}
        
        files = []
        pattern = "**/*" if recursive else "*"
        
        for ext in file_types:
            for file_path in path_obj.glob(f"{pattern}.{ext}"):
                if file_path.is_file():
                    stat = file_path.stat()
                    file_hash = hashlib.md5(
                        f"{file_path}_{stat.st_mtime}_{stat.st_size}".encode()
                    ).hexdigest()
                    
                    files.append({
                        "path": str(file_path),
                        "name": file_path.name,
                        "extension": file_path.suffix,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "hash": file_hash
                    })
        
        # Detect changes
        directory_hash = hashlib.md5(
            json.dumps(sorted([f["hash"] for f in files])).encode()
        ).hexdigest()
        
        previous_hash = file_cache.get(path, {}).get("hash")
        has_changes = previous_hash != directory_hash if previous_hash else True
        
        # Find new and modified files
        new_files = []
        modified_files = []
        
        if path in file_cache:
            previous_files = {f["path"]: f["hash"] for f in file_cache[path].get("files", [])}
            
            for file in files:
                if file["path"] not in previous_files:
                    new_files.append(file["path"])
                elif previous_files[file["path"]] != file["hash"]:
                    modified_files.append(file["path"])
        
        # Update cache
        file_cache[path] = {
            "hash": directory_hash,
            "files": files,
            "last_check": datetime.now().isoformat()
        }
        
        return {
            "path": path,
            "total_files": len(files),
            "files": files,
            "has_changes": has_changes,
            "new_files": new_files,
            "modified_files": modified_files,
            "directory_hash": directory_hash
        }
        
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_file_metadata(filepath: str) -> dict:
    """
    Extract detailed metadata from a file for enhanced indexing.
    
    Args:
        filepath: Path to the file
    
    Returns:
        Dictionary with file metadata
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return {"error": f"File does not exist: {filepath}"}
        
        stat = path.stat()
        mime_type, _ = mimetypes.guess_type(str(path))
        
        metadata = {
            "path": str(path),
            "name": path.name,
            "stem": path.stem,
            "extension": path.suffix,
            "size_bytes": stat.st_size,
            "size_human": _format_bytes(stat.st_size),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
            "mime_type": mime_type,
            "is_hidden": path.name.startswith('.'),
            "parent_directory": str(path.parent),
            "absolute_path": str(path.absolute())
        }
        
        # Add content preview for text files
        if mime_type and mime_type.startswith('text'):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read(500)
                    metadata["preview"] = content[:500] + ("..." if len(content) > 500 else "")
                    metadata["line_count"] = content.count('\n') + 1
            except:
                pass
        
        return metadata
        
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def read_file_content(filepath: str, encoding: str = "utf-8") -> dict:
    """
    Read the content of a file for ingestion.
    
    Args:
        filepath: Path to the file
        encoding: File encoding (default: utf-8)
    
    Returns:
        Dictionary with file content and metadata
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return {"error": f"File does not exist: {filepath}"}
        
        with open(path, 'r', encoding=encoding) as f:
            content = f.read()
        
        return {
            "path": str(path),
            "content": content,
            "length": len(content),
            "encoding": encoding
        }
        
    except UnicodeDecodeError:
        return {"error": f"Could not decode file with {encoding} encoding"}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def batch_prepare_documents(directory: str, file_types: list[str] = None) -> list:
    """
    Prepare multiple documents for batch ingestion.
    
    Args:
        directory: Directory containing documents
        file_types: List of file extensions to process
    
    Returns:
        List of prepared documents ready for RAG ingestion
    """
    if file_types is None:
        file_types = ['txt', 'md', 'pdf']
    
    monitor_result = await monitor_directory(directory, file_types, recursive=True)
    
    if "error" in monitor_result:
        return [{"error": monitor_result["error"]}]
    
    documents = []
    for file_info in monitor_result["files"]:
        if file_info["extension"][1:] in file_types:  # Remove dot from extension
            doc = {
                "path": file_info["path"],
                "metadata": {
                    "source": file_info["path"],
                    "filename": file_info["name"],
                    "modified": file_info["modified"],
                    "size": file_info["size"],
                    "type": file_info["extension"]
                }
            }
            
            # For text files, include content
            if file_info["extension"] in ['.txt', '.md']:
                content_result = await read_file_content(file_info["path"])
                if "content" in content_result:
                    doc["content"] = content_result["content"]
            
            documents.append(doc)
    
    return documents

def _format_bytes(bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

if __name__ == "__main__":
    mcp.run(transport="stdio")

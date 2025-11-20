try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    # Create a dummy FastMCP for when mcp is not available
    class FastMCP:
        def __init__(self, name):
            self.name = name
            print(f"Warning: MCP package not available. {name} server will not function.")
        
        def tool(self):
            def decorator(func):
                return func
            return decorator
        
        def run(self, transport="stdio"):
            print(f"Warning: Cannot run {self.name} - MCP package not installed")
            pass
    
    MCP_AVAILABLE = False

import hashlib
from pathlib import Path
import json
from datetime import datetime
from typing import Set
import mimetypes
import os
import asyncio
from threading import Thread
import queue

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("Warning: watchdog not installed. Using polling fallback.")

mcp = FastMCP("FileSystemRAG")

# Security configuration - allowed base paths for file operations
# Prevents path traversal attacks by restricting access to specific directories
ALLOWED_BASE_PATHS = [
    Path.cwd(),  # Current working directory
    Path.home() / "documents",  # User documents
    Path("/tmp"),  # Temporary files
]

# You can also configure this via environment variable
_custom_paths = os.getenv("MCP_ALLOWED_PATHS")
if _custom_paths:
    for custom_path in _custom_paths.split(":"):
        try:
            ALLOWED_BASE_PATHS.append(Path(custom_path).resolve())
        except Exception:
            pass


def _validate_path_security(filepath: str) -> tuple[bool, str, Path]:
    """
    Validate that a file path is within allowed directories to prevent path traversal attacks.

    Args:
        filepath: Path to validate

    Returns:
        Tuple of (is_valid, error_message, resolved_path)
    """
    try:
        path = Path(filepath).resolve()

        # Check if path is within any allowed base path
        for allowed_base in ALLOWED_BASE_PATHS:
            try:
                allowed_base_resolved = allowed_base.resolve()
                # Use is_relative_to() to check if path is within allowed directory
                # This prevents ../../../etc/passwd type attacks
                if path.is_relative_to(allowed_base_resolved):
                    return True, "", path
            except (ValueError, RuntimeError):
                continue

        allowed_paths_str = ", ".join(str(p) for p in ALLOWED_BASE_PATHS)
        return False, f"Access denied: Path '{filepath}' is not within allowed directories: {allowed_paths_str}", path

    except Exception as e:
        return False, f"Invalid path: {e}", Path()


# Global state for file tracking
file_cache = {}
deleted_files_queue = queue.Queue()
modified_files_queue = queue.Queue()
new_files_queue = queue.Queue()

class MCPFileHandler(FileSystemEventHandler):
    """Watchdog event handler for file system changes"""
    
    def __init__(self, file_types: Set[str]):
        self.file_types = file_types
        
    def _is_relevant_file(self, path: str) -> bool:
        """Check if file type is relevant"""
        return any(path.endswith(f'.{ext}') for ext in self.file_types)
    
    def on_created(self, event: FileSystemEvent):
        if not event.is_directory and self._is_relevant_file(event.src_path):
            new_files_queue.put(event.src_path)
    
    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory and self._is_relevant_file(event.src_path):
            modified_files_queue.put(event.src_path)
    
    def on_deleted(self, event: FileSystemEvent):
        if not event.is_directory and self._is_relevant_file(event.src_path):
            deleted_files_queue.put(event.src_path)
    
    def on_moved(self, event: FileSystemEvent):
        if not event.is_directory:
            if self._is_relevant_file(event.src_path):
                deleted_files_queue.put(event.src_path)
            if self._is_relevant_file(event.dest_path):
                new_files_queue.put(event.dest_path)

# Active observers for different directories
active_observers = {}

@mcp.tool()
async def start_monitoring(path: str, file_types: list[str] = None) -> dict:
    """
    Start real-time monitoring of a directory using watchdog.

    Security: Path is validated to prevent monitoring of unauthorized directories.

    Args:
        path: Directory path to monitor
        file_types: List of file extensions to track

    Returns:
        Status of monitoring initialization
    """
    if not WATCHDOG_AVAILABLE:
        return {"error": "watchdog library not installed. Install with: pip install watchdog"}

    file_types = file_types or ['pdf', 'txt', 'md', 'docx', 'html']

    try:
        # Validate path security first
        is_valid, error_msg, path_obj = _validate_path_security(path)
        if not is_valid:
            return {"error": error_msg}

        if not path_obj.exists():
            return {"error": f"Path does not exist: {path}"}
        
        # Stop existing observer for this path if any
        if path in active_observers:
            active_observers[path].stop()
            active_observers[path].join()
        
        # Create and start new observer
        event_handler = MCPFileHandler(set(file_types))
        observer = Observer()
        observer.schedule(event_handler, str(path_obj), recursive=True)
        observer.start()
        
        active_observers[path] = observer
        
        return {
            "success": True,
            "message": f"Started monitoring {path}",
            "file_types": file_types
        }
        
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def stop_monitoring(path: str) -> dict:
    """Stop monitoring a directory."""
    if path in active_observers:
        active_observers[path].stop()
        active_observers[path].join()
        del active_observers[path]
        return {"success": True, "message": f"Stopped monitoring {path}"}
    return {"error": f"No active monitoring for {path}"}

@mcp.tool()
async def get_file_changes() -> dict:
    """
    Get accumulated file changes since last check.
    
    Returns:
        Dictionary with new, modified, and deleted files
    """
    new_files = []
    modified_files = []
    deleted_files = []
    
    # Collect all changes from queues
    while not new_files_queue.empty():
        try:
            new_files.append(new_files_queue.get_nowait())
        except queue.Empty:
            break
    
    while not modified_files_queue.empty():
        try:
            modified_files.append(modified_files_queue.get_nowait())
        except queue.Empty:
            break
    
    while not deleted_files_queue.empty():
        try:
            deleted_files.append(deleted_files_queue.get_nowait())
        except queue.Empty:
            break
    
    # Remove duplicates
    new_files = list(set(new_files))
    modified_files = list(set(modified_files))
    deleted_files = list(set(deleted_files))
    
    return {
        "new_files": new_files,
        "modified_files": modified_files,
        "deleted_files": deleted_files,
        "has_changes": bool(new_files or modified_files or deleted_files)
    }

@mcp.tool()
async def monitor_directory(path: str, file_types: list[str] = None, recursive: bool = True) -> dict:
    """
    Fallback polling-based monitoring for compatibility.
    Also provides initial scan of directory.
    
    Args:
        path: Directory path to monitor
        file_types: List of file extensions to track
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
        
        current_files = {}
        pattern = "**/*" if recursive else "*"
        
        for ext in file_types:
            for file_path in path_obj.glob(f"{pattern}.{ext}"):
                if file_path.is_file():
                    stat = file_path.stat()
                    file_hash = hashlib.md5(
                        f"{file_path}_{stat.st_mtime}_{stat.st_size}".encode()
                    ).hexdigest()
                    
                    current_files[str(file_path)] = {
                        "path": str(file_path),
                        "name": file_path.name,
                        "extension": file_path.suffix,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "hash": file_hash
                    }
        
        # Compare with cache to detect changes
        new_files = []
        modified_files = []
        deleted_files = []
        
        if path in file_cache:
            previous_files = file_cache[path]
            
            # Find new and modified files
            for file_path, file_info in current_files.items():
                if file_path not in previous_files:
                    new_files.append(file_path)
                elif previous_files[file_path]["hash"] != file_info["hash"]:
                    modified_files.append(file_path)
            
            # Find deleted files
            for file_path in previous_files:
                if file_path not in current_files:
                    deleted_files.append(file_path)
        else:
            # First scan - all files are new
            new_files = list(current_files.keys())
        
        # Update cache
        file_cache[path] = current_files
        
        return {
            "path": path,
            "total_files": len(current_files),
            "files": list(current_files.values()),
            "has_changes": bool(new_files or modified_files or deleted_files),
            "new_files": new_files,
            "modified_files": modified_files,
            "deleted_files": deleted_files
        }
        
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_file_metadata(filepath: str) -> dict:
    """
    Extract detailed metadata from a file for enhanced indexing.

    Security: Path is validated to prevent path traversal attacks.

    Args:
        filepath: Path to the file

    Returns:
        Dictionary with file metadata
    """
    try:
        # Validate path security first
        is_valid, error_msg, path = _validate_path_security(filepath)
        if not is_valid:
            return {"error": error_msg}

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

    Security: Path is validated to prevent path traversal attacks.

    Args:
        filepath: Path to the file
        encoding: File encoding (default: utf-8)

    Returns:
        Dictionary with file content and metadata
    """
    try:
        # Validate path security first
        is_valid, error_msg, path = _validate_path_security(filepath)
        if not is_valid:
            return {"error": error_msg}

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

# Cleanup function
def cleanup():
    """Stop all active observers on shutdown."""
    for path, observer in active_observers.items():
        observer.stop()
        observer.join()

if __name__ == "__main__":
    if MCP_AVAILABLE:
        try:
            mcp.run(transport="stdio")
        finally:
            cleanup()
    else:
        print("MCP package not available. Server cannot run.")
        print("Install with: pip install mcp>=1.6.0")

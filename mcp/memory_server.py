from fastmcp import FastMCP
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib
import numpy as np
from pathlib import Path

mcp = FastMCP("MemoryRAG")

# In-memory cache for fast access
memory_store = {}
embedding_cache = {}

# Persistent storage path
STORAGE_PATH = Path("mcp_data/memory_storage")
STORAGE_PATH.mkdir(parents=True, exist_ok=True)
DB_PATH = STORAGE_PATH / "memory.db"

# Initialize database
def _init_db():
    """Initialize SQLite database with proper schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create conversations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            context_docs TEXT,
            metadata TEXT,
            keywords TEXT,
            embedding TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    ''')
    
    # Create sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            user_id TEXT,
            metadata TEXT
        )
    ''')
    
    # Create embeddings table for semantic search
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            embedding_vector TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
        )
    ''')
    
    # Create indices for better performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON conversations(session_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_keywords ON conversations(keywords)')
    
    conn.commit()
    conn.close()

_init_db()

@mcp.tool()
async def store_conversation_context(
    session_id: str, 
    question: str, 
    answer: str, 
    context_docs: list = None,
    metadata: dict = None,
    embedding_vector: list = None
) -> dict:
    """
    Store conversation context with enhanced metadata for later retrieval.
    
    Args:
        session_id: Unique session identifier
        question: User's question
        answer: System's answer
        context_docs: Documents used to generate answer
        metadata: Additional metadata
        embedding_vector: Optional embedding for semantic search
    
    Returns:
        Success status and storage ID
    """
    try:
        timestamp = datetime.now()
        conversation_id = hashlib.md5(
            f"{session_id}_{timestamp.isoformat()}".encode()
        ).hexdigest()[:12]
        
        keywords = _extract_keywords(question + " " + answer)
        
        # Store in SQLite
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Ensure session exists
        cursor.execute('''
            INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?)
        ''', (session_id, timestamp.isoformat(), timestamp.isoformat(), json.dumps({})))
        
        # Insert conversation
        cursor.execute('''
            INSERT INTO conversations 
            (conversation_id, session_id, timestamp, question, answer, 
             context_docs, metadata, keywords, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            conversation_id,
            session_id,
            timestamp.isoformat(),
            question,
            answer,
            json.dumps(context_docs or []),
            json.dumps(metadata or {}),
            json.dumps(keywords),
            json.dumps(embedding_vector) if embedding_vector else None
        ))
        
        # Store embedding if provided
        if embedding_vector:
            cursor.execute('''
                INSERT INTO embeddings (conversation_id, embedding_vector)
                VALUES (?, ?)
            ''', (conversation_id, json.dumps(embedding_vector)))
        
        conn.commit()
        conn.close()
        
        # Update in-memory cache
        if session_id not in memory_store:
            memory_store[session_id] = []
        
        memory_entry = {
            "conversation_id": conversation_id,
            "session_id": session_id,
            "timestamp": timestamp.isoformat(),
            "question": question,
            "answer": answer,
            "context_docs": context_docs or [],
            "metadata": metadata or {},
            "keywords": keywords,
            "embedding": embedding_vector
        }
        memory_store[session_id].append(memory_entry)
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "session_id": session_id,
            "message": "Conversation context stored successfully"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def retrieve_relevant_memories(
    query: str, 
    session_id: Optional[str] = None, 
    top_k: int = 5,
    similarity_threshold: float = 0.5,
    query_embedding: list = None
) -> list:
    """
    Retrieve relevant past conversations using semantic search when embeddings are available.
    
    Args:
        query: Search query
        session_id: Optional session filter
        top_k: Number of results to return
        similarity_threshold: Minimum similarity score
        query_embedding: Optional embedding vector for semantic search
    
    Returns:
        List of relevant conversation memories
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        if query_embedding:
            # Semantic search using embeddings
            query_embedding_np = np.array(query_embedding)
            
            # Get all conversations with embeddings
            if session_id:
                cursor.execute('''
                    SELECT c.*, e.embedding_vector 
                    FROM conversations c
                    JOIN embeddings e ON c.conversation_id = e.conversation_id
                    WHERE c.session_id = ?
                ''', (session_id,))
            else:
                cursor.execute('''
                    SELECT c.*, e.embedding_vector 
                    FROM conversations c
                    JOIN embeddings e ON c.conversation_id = e.conversation_id
                ''')
            
            results = []
            for row in cursor.fetchall():
                stored_embedding = json.loads(row[-1])
                stored_embedding_np = np.array(stored_embedding)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding_np, stored_embedding_np) / (
                    np.linalg.norm(query_embedding_np) * np.linalg.norm(stored_embedding_np)
                )
                
                if similarity >= similarity_threshold:
                    results.append({
                        "conversation_id": row[0],
                        "session_id": row[1],
                        "timestamp": row[2],
                        "question": row[3],
                        "answer": row[4],
                        "score": float(similarity)
                    })
            
            # Sort by similarity score
            results.sort(key=lambda x: x["score"], reverse=True)
            
        else:
            # Fallback to keyword search
            query_keywords = set(_extract_keywords(query.lower()))
            
            # Get conversations from database
            if session_id:
                cursor.execute('''
                    SELECT conversation_id, session_id, timestamp, question, answer, keywords
                    FROM conversations
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                ''', (session_id,))
            else:
                cursor.execute('''
                    SELECT conversation_id, session_id, timestamp, question, answer, keywords
                    FROM conversations
                    ORDER BY timestamp DESC
                ''')
            
            results = []
            for row in cursor.fetchall():
                stored_keywords = set(json.loads(row[5]) if row[5] else [])
                
                # Calculate keyword overlap score
                if query_keywords:
                    overlap = len(query_keywords.intersection(stored_keywords))
                    score = overlap / len(query_keywords) if query_keywords else 0
                else:
                    score = 0
                
                # Check for direct text matches
                text_match = any(
                    term in row[3].lower() or term in row[4].lower()
                    for term in query.lower().split()
                )
                if text_match:
                    score += 0.5
                
                if score >= similarity_threshold:
                    results.append({
                        "conversation_id": row[0],
                        "session_id": row[1],
                        "timestamp": row[2],
                        "question": row[3],
                        "answer": row[4],
                        "score": score
                    })
            
            results.sort(key=lambda x: x["score"], reverse=True)
        
        conn.close()
        return results[:top_k]
        
    except Exception as e:
        return [{"error": str(e)}]

@mcp.tool()
async def get_session_history(
    session_id: str, 
    limit: int = 10,
    include_context: bool = False
) -> list:
    """
    Get conversation history for a specific session.
    
    Args:
        session_id: Session identifier
        limit: Maximum number of conversations to return
        include_context: Whether to include document context
    
    Returns:
        List of conversations in chronological order
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT conversation_id, session_id, timestamp, question, answer, 
                   context_docs, metadata, keywords
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (session_id, limit))
        
        conversations = []
        for row in cursor.fetchall():
            conv = {
                "conversation_id": row[0],
                "session_id": row[1],
                "timestamp": row[2],
                "question": row[3],
                "answer": row[4],
                "metadata": json.loads(row[6]) if row[6] else {},
                "keywords": json.loads(row[7]) if row[7] else []
            }
            
            if include_context:
                conv["context_docs"] = json.loads(row[5]) if row[5] else []
            
            conversations.append(conv)
        
        conn.close()
        return conversations
        
    except Exception as e:
        return [{"error": str(e)}]

@mcp.tool()
async def summarize_session(session_id: str) -> dict:
    """
    Generate a summary of a conversation session.
    
    Args:
        session_id: Session identifier
    
    Returns:
        Summary statistics and key topics
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get all conversations for session
        cursor.execute('''
            SELECT question, answer, keywords, timestamp
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp
        ''', (session_id,))
        
        rows = cursor.fetchall()
        
        if not rows:
            conn.close()
            return {"error": f"Session {session_id} not found"}
        
        # Collect all keywords
        all_keywords = []
        total_answer_length = 0
        
        for row in rows:
            keywords = json.loads(row[2]) if row[2] else []
            all_keywords.extend(keywords)
            total_answer_length += len(row[1])
        
        # Count keyword frequency
        keyword_freq = {}
        for kw in all_keywords:
            keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
        
        # Get top topics
        top_topics = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        conn.close()
        
        return {
            "session_id": session_id,
            "total_conversations": len(rows),
            "first_interaction": rows[0][3] if rows else None,
            "last_interaction": rows[-1][3] if rows else None,
            "top_topics": [{"topic": t[0], "frequency": t[1]} for t in top_topics],
            "average_answer_length": total_answer_length / len(rows) if rows else 0
        }
        
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def clear_session_memory(session_id: str) -> dict:
    """
    Clear memory for a specific session.
    
    Args:
        session_id: Session identifier
    
    Returns:
        Success status
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Delete from database
        cursor.execute('DELETE FROM conversations WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
        
        conn.commit()
        conn.close()
        
        # Clear from memory cache
        if session_id in memory_store:
            del memory_store[session_id]
        
        return {"success": True, "message": f"Session {session_id} cleared"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def export_memories(session_id: Optional[str] = None, format: str = "json") -> dict:
    """
    Export memories in various formats.
    
    Args:
        session_id: Optional session filter
        format: Export format (json, markdown)
    
    Returns:
        Exported data or file path
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute('''
                SELECT * FROM conversations 
                WHERE session_id = ?
                ORDER BY timestamp
            ''', (session_id,))
        else:
            cursor.execute('SELECT * FROM conversations ORDER BY session_id, timestamp')
        
        rows = cursor.fetchall()
        conn.close()
        
        if format == "json":
            export_path = STORAGE_PATH / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            data = []
            for row in rows:
                data.append({
                    "conversation_id": row[0],
                    "session_id": row[1],
                    "timestamp": row[2],
                    "question": row[3],
                    "answer": row[4],
                    "context_docs": json.loads(row[5]) if row[5] else [],
                    "metadata": json.loads(row[6]) if row[6] else {},
                    "keywords": json.loads(row[7]) if row[7] else []
                })
            
            with open(export_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return {"success": True, "path": str(export_path), "format": "json"}
            
        elif format == "markdown":
            export_path = STORAGE_PATH / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
            with open(export_path, 'w') as f:
                f.write("# Conversation Memory Export\n\n")
                
                current_session = None
                for row in rows:
                    if row[1] != current_session:
                        current_session = row[1]
                        f.write(f"## Session: {current_session}\n\n")
                    
                    f.write(f"### {row[2]}\n")
                    f.write(f"**Question:** {row[3]}\n\n")
                    f.write(f"**Answer:** {row[4]}\n\n")
                    f.write("---\n\n")
            
            return {"success": True, "path": str(export_path), "format": "markdown"}
            
        else:
            return {"success": False, "error": f"Unsupported format: {format}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

# Helper functions
def _extract_keywords(text: str, max_keywords: int = 10) -> list:
    """Extract simple keywords from text."""
    import re
    
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Common stop words to ignore
    stop_words = {
        'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are', 
        'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
        'this', 'that', 'these', 'those', 'i', 'you', 'we', 'they', 'it'
    }
    
    # Extract words
    words = text.split()
    keywords = [w for w in words if len(w) > 3 and w not in stop_words]
    
    # Count frequency
    word_freq = {}
    for word in keywords:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Return top keywords
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [kw[0] for kw in sorted_keywords[:max_keywords]]

if __name__ == "__main__":
    mcp.run(transport="stdio")

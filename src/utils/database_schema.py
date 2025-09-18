"""
Database schema and initialization for vessel detection system.
Provides complete database setup and migration capabilities.
"""

import sqlite3
import logging
import os
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database initialization, schema, and migrations."""
    
    def __init__(self, db_path: str = "vessel_detection.db"):
        self.db_path = db_path
        self.conn = None
        
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection, creating database if it doesn't exist."""
        if self.conn is None:
            # Create database directory if it doesn't exist
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
                
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Enable dict-like access
            
            # Initialize schema if database is empty
            if self._is_database_empty():
                self._initialize_schema()
                logger.info(f"Initialized new database: {self.db_path}")
            else:
                logger.info(f"Connected to existing database: {self.db_path}")
                
        return self.conn
    
    def _is_database_empty(self) -> bool:
        """Check if database has any tables."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        return len(tables) == 0
    
    def _initialize_schema(self):
        """Initialize database schema with all required tables."""
        cursor = self.conn.cursor()
        
        # Create datasets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                task TEXT NOT NULL,
                categories TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create images table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                format TEXT NOT NULL,
                channels TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                preprocessed BOOLEAN DEFAULT FALSE,
                hidden BOOLEAN DEFAULT FALSE,
                bounds TEXT,
                time TIMESTAMP,
                projection TEXT,
                column_offset INTEGER DEFAULT 0,
                row_offset INTEGER DEFAULT 0,
                zoom INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create windows table for spatial data organization
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS windows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                image_id INTEGER NOT NULL,
                row INTEGER NOT NULL,
                column INTEGER NOT NULL,
                height INTEGER NOT NULL,
                width INTEGER NOT NULL,
                hidden BOOLEAN DEFAULT FALSE,
                split TEXT DEFAULT 'train',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id),
                FOREIGN KEY (image_id) REFERENCES images (id)
            )
        """)
        
        # Create labels table for training annotations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                window_id INTEGER NOT NULL,
                row INTEGER NOT NULL,
                column INTEGER NOT NULL,
                height INTEGER NOT NULL,
                width INTEGER NOT NULL,
                extent TEXT,
                value TEXT,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (window_id) REFERENCES windows (id)
            )
        """)
        
        # Create detections table for storing inference results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_id TEXT NOT NULL,
                detect_id TEXT UNIQUE NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                score REAL NOT NULL,
                vessel_length_m REAL,
                vessel_width_m REAL,
                vessel_speed_k REAL,
                is_fishing_vessel REAL,
                heading_bucket_0 REAL,
                heading_bucket_1 REAL,
                heading_bucket_2 REAL,
                heading_bucket_3 REAL,
                heading_bucket_4 REAL,
                heading_bucket_5 REAL,
                heading_bucket_6 REAL,
                heading_bucket_7 REAL,
                heading_bucket_8 REAL,
                heading_bucket_9 REAL,
                heading_bucket_10 REAL,
                heading_bucket_11 REAL,
                heading_bucket_12 REAL,
                heading_bucket_13 REAL,
                heading_bucket_14 REAL,
                heading_bucket_15 REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_uuid ON images (uuid)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_windows_dataset ON windows (dataset_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_windows_image ON windows (image_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_labels_window ON labels (window_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_scene ON detections (scene_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_coords ON detections (lat, lon)")
        
        # Insert default dataset if none exists
        cursor.execute("SELECT COUNT(*) FROM datasets")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO datasets (collection_id, name, task, categories)
                VALUES (1, 'sentinel1_vessel_detection', 'object_detection', '["vessel"]')
            """)
        
        self.conn.commit()
        logger.info("Database schema initialized successfully")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def initialize_database(db_path: str = "vessel_detection.db") -> DatabaseManager:
    """Initialize database with proper schema.
    
    Args:
        db_path: Path to database file
        
    Returns:
        DatabaseManager instance
    """
    db_manager = DatabaseManager(db_path)
    db_manager.get_connection()  # This will initialize schema if needed
    return db_manager


def get_database_connection(db_path: str = "vessel_detection.db") -> sqlite3.Connection:
    """Get database connection with proper initialization.
    
    Args:
        db_path: Path to database file
        
    Returns:
        SQLite connection with schema initialized
    """
    db_manager = DatabaseManager(db_path)
    return db_manager.get_connection()


# Backward compatibility functions
def dict_factory(cursor, row):
    """Converts rows returned from sqlite queries to dicts."""
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def get_dataset(conn, id):
    """Get dataset by ID."""
    query = "SELECT d.id, d.collection_id, d.name, d.task, d.categories FROM datasets AS d WHERE d.id = ?"
    cur = conn.cursor()
    cur.execute(query, (id,))
    row = cur.fetchone()
    return row


def get_windows(conn, dataset_id, split=None):
    """Get windows for a dataset."""
    query = "SELECT w.id, w.dataset_id, w.image_id, w.row, w.column, w.height, w.width, w.hidden, w.split FROM windows AS w WHERE w.dataset_id = ?"
    params = [dataset_id]
    
    if split:
        query += " AND w.split = ?"
        params.append(split)
        
    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    return rows


def get_image(conn, image_id):
    """Get image by ID."""
    query = """
        SELECT ims.id, ims.uuid, ims.name, ims.format, ims.channels, ims.width, ims.height, 
               ims.preprocessed, ims.hidden, ims.bounds, ims.time, ims.projection, 
               ims.column_offset, ims.row_offset, ims.zoom 
        FROM images AS ims WHERE ims.id = ?
    """
    cur = conn.cursor()
    cur.execute(query, (image_id,))
    row = cur.fetchone()
    return row


def get_labels(conn, window_id):
    """Get labels for a window."""
    query = "SELECT l.id, l.window_id, l.row, l.column, l.height, l.width, l.extent, l.value, l.properties FROM labels AS l WHERE l.window_id = ?"
    cur = conn.cursor()
    cur.execute(query, (window_id,))
    rows = cur.fetchall()
    return rows


def get_dataset_labels(conn, dataset_id, splits=[]):
    """Get all labels for a dataset."""
    query = """
        SELECT l.id, l.window_id, l.row, l.column, l.height, l.width, l.extent, l.value, l.properties 
        FROM labels AS l 
        WHERE l.window_id IN (
            SELECT id FROM windows WHERE dataset_id = ?
        )
    """
    params = [dataset_id]
    
    if splits:
        placeholders = ','.join(['?' for _ in splits])
        query += f" AND split IN ({placeholders})"
        params.extend(splits)
        
    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    return rows

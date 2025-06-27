import sqlite3
import json
from typing import Dict, Any, List

DB_PATH = "blockchain.db"


def init_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        (
            "CREATE TABLE IF NOT EXISTS blocks ("
            "hash TEXT PRIMARY KEY,"
            "data TEXT"
            ")"
        )
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS utxos (
            key TEXT PRIMARY KEY,
            data TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_block(block_hash: str, block_data: Dict[str, Any], db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        "REPLACE INTO blocks (hash, data) VALUES (?, ?)",
        (block_hash, json.dumps(block_data)),
    )
    conn.commit()
    conn.close()


def load_blocks(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT data FROM blocks ORDER BY rowid ASC")
    rows = c.fetchall()
    conn.close()
    return [json.loads(row[0]) for row in rows]


def save_utxo(key: str, utxo_data: Dict[str, Any], db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        "REPLACE INTO utxos (key, data) VALUES (?, ?)",
        (key, json.dumps(utxo_data)),
    )
    conn.commit()
    conn.close()


def load_utxos(db_path: str = DB_PATH) -> Dict[str, Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT key, data FROM utxos")
    rows = c.fetchall()
    conn.close()
    return {row[0]: json.loads(row[1]) for row in rows}


def delete_block(block_hash: str, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("DELETE FROM blocks WHERE hash = ?", (block_hash,))
    conn.commit()
    conn.close()


def delete_utxo(key: str, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("DELETE FROM utxos WHERE key = ?", (key,))
    conn.commit()
    conn.close() 
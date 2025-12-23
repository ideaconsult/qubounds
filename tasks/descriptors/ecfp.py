# ecfp_cache.py
from rdkit import Chem
from rdkit.Chem import AllChem, inchi
import sqlite3
import numpy as np
import threading
import logging 


_conn_lock = threading.Lock()
_cached_conn = None

logger = logging.getLogger(__name__)

def init_cache(db_path="ecfp_cache.db"):
    global _cached_conn
    with _conn_lock:
        if _cached_conn is None:
            _cached_conn = sqlite3.connect(db_path)
            c = _cached_conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS ecfp_cache (
                    inchikey TEXT PRIMARY KEY,
                    ecfp BLOB
                )
            ''')
            c.execute('SELECT COUNT(*) FROM ecfp_cache')
            count = c.fetchone()[0]
            print(f"[ecfp_cache] Number of cached entries: {count}")
            _cached_conn.commit()
    return _cached_conn


def get_conn():
    if _cached_conn is None:
        raise RuntimeError("ECFP cache not initialized. Call init_cache() first.")
    return _cached_conn


def smiles_to_ecfp_cached(smiles: str, radius: int = 4, n_bits: int = 2048, conn = None) -> np.ndarray:
    if conn is None:
        conn = get_conn()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=int)

    # Get InChIKey
    try:
        inchikey = inchi.MolToInchiKey(mol)
    except Exception:
        return np.zeros(n_bits, dtype=int)

    # Check cache
    if conn:
        c = conn.cursor()
        c.execute("SELECT ecfp FROM ecfp_cache WHERE inchikey=?", (inchikey,))
        row = c.fetchone()
        if row:
            return np.frombuffer(row[0], dtype=np.int8)

    # Compute ECFP
    bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(bit_vect, arr)

    # Cache result
    if conn:
        c.execute("INSERT OR REPLACE INTO ecfp_cache (inchikey, ecfp) VALUES (?, ?)", 
                  (inchikey, arr.astype(np.int8).tobytes()))
        conn.commit()

    return arr

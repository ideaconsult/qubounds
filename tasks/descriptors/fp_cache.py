import sqlite3
import pickle
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import inchi


"""
Usage with pandas DataFrames
============================

This cache is designed to work naturally with tabular compound data where
SMILES and optional compound identifiers are stored in a pandas DataFrame.

The only requirement is:
    * SMILES must be valid RDKit SMILES strings
    * compound_id may be missing (None / NaN)

--------------------------------------------------------------------------

Example DataFrame
-----------------

>>> import pandas as pd
>>> df = pd.DataFrame({
...     "compound_id": ["CMPD_001", "CMPD_002", None, "CMPD_004"],
...     "smiles": ["CCO", "c1ccccc1", "CCO", "CCN"]
... })

--------------------------------------------------------------------------

Preferred usage (fast and explicit)
-----------------------------------

Use zip() to pass SMILES and optional IDs together. Missing IDs should be
represented as None (NaN must be normalized explicitly).

>>> import pandas as pd
>>> fps = [
...     fp_cache.get_fingerprint(smi, compound_id=cid)
...     for cid, smi in zip(df["compound_id"], df["smiles"])
... ]

--------------------------------------------------------------------------

Alternative: itertuples() (often fastest)
-----------------------------------------

>>> fps = [
...     fp_cache.get_fingerprint(row.smiles, compound_id=row.compound_id)
...     for row in df.itertuples(index=False)
... ]

--------------------------------------------------------------------------

Handling NaN compound IDs
-------------------------

pandas uses NaN for missing values, which must be converted to None so that
SQLite stores NULL correctly.

>>> def normalize_id(x):
...     return None if pd.isna(x) else str(x)

>>> fps = [
...     fp_cache.get_fingerprint(smi, compound_id=normalize_id(cid))
...     for cid, smi in zip(df["compound_id"], df["smiles"])
... ]

--------------------------------------------------------------------------

Storing fingerprints back into the DataFrame
--------------------------------------------

>>> df["fingerprint"] = [
...     fp_cache.get_fingerprint(smi, compound_id=cid)
...     for cid, smi in zip(df["compound_id"], df["smiles"])
... ]

--------------------------------------------------------------------------

Notes
-----

* When compound_id is None, lookup and deduplication are performed by InChI.
* When compound_id is provided, ID-based lookup is preferred.
* Same structure with different IDs results in distinct cache entries.
* Always close the cache when finished:

>>> fp_cache.close()
"""


class FingerprintCache:
    def __init__(self, db_path="fp_id_cache.db", radius=2, nbits=2048):
        self.db_path = Path(db_path)
        self.radius = radius
        self.nbits = nbits
        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()
        self._init_db()

    def _init_db(self):
        """Create cache table if not exists."""
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS fp_id_cache (
            compound_id TEXT,
            inchi TEXT,
            fingerprint BLOB NOT NULL,
            PRIMARY KEY (compound_id, inchi)
        )
        """)
        self.cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_fp_cache_compound_id ON fp_cache(compound_id)"
        )
        self.cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_fp_cache_inchi ON fp_cache(inchi)"
        )
        self.conn.commit()

    @staticmethod
    def _mol_to_inchi(mol):
        """Convert RDKit Mol to canonical InChI string."""
        return inchi.MolToInchi(mol)

    def _get_fp_from_cache(self, compound_id=None, inchi_str=None):
        if compound_id is not None:
            self.cur.execute(
                "SELECT fingerprint FROM fp_id_cache WHERE compound_id = ?",
                (compound_id,)
            )
        else:
            self.cur.execute(
                "SELECT fingerprint FROM fp_id_cache WHERE inchi = ?",
                (inchi_str,)
            )

        row = self.cur.fetchone()
        if row:
            return pickle.loads(row[0])
        return None

    def _store_fp_in_cache(self, fp, compound_id=None, inchi_str=None):
        self.cur.execute(
            """
            INSERT OR REPLACE INTO fp_id_cache (compound_id, inchi, fingerprint)
            VALUES (?, ?, ?)
            """,
            (compound_id, inchi_str, pickle.dumps(fp))
        )
        self.conn.commit()

    def get_fingerprint(self, smiles, compound_id=None):
        """
        Retrieve fingerprint using compound_id if available,
        otherwise fall back to InChI-based lookup.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        inchi_str = self._mol_to_inchi(mol)

        # Prefer compound_id lookup when provided
        fp = self._get_fp_from_cache(
            compound_id=compound_id,
            inchi_str=inchi_str
        )

        if fp is None:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=self.radius,
                nBits=self.nbits
            )
            self._store_fp_in_cache(
                fp,
                compound_id=compound_id,
                inchi_str=inchi_str
            )

        return fp

    def close(self):
        """Close database connection."""
        self.conn.close()

import sqlite3
import pickle
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import inchi


class FingerprintCache:
    def __init__(self, db_path="fingerprint_cache.db", radius=2, nbits=2048):
        self.db_path = Path(db_path)
        self.radius = radius
        self.nbits = nbits
        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()
        self._init_db()

    def _init_db(self):
        """Create cache table if not exists."""
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS fp_cache (
            inchi TEXT PRIMARY KEY,
            fingerprint BLOB
        )
        """)
        self.conn.commit()

    def _mol_to_inchi(self, mol):
        """Convert RDKit Mol to canonical InChI string."""
        return inchi.MolToInchi(mol)

    def _get_fp_from_cache(self, inchi_str):
        self.cur.execute("SELECT fingerprint FROM fp_cache WHERE inchi = ?", (inchi_str,))
        row = self.cur.fetchone()
        if row:
            return pickle.loads(row[0])
        return None

    def _store_fp_in_cache(self, inchi_str, fp):
        self.cur.execute(
            "INSERT OR REPLACE INTO fp_cache (inchi, fingerprint) VALUES (?, ?)",
            (inchi_str, pickle.dumps(fp))
        )
        self.conn.commit()

    def get_fingerprint(self, smiles):
        """Retrieve fingerprint (by InChI), computing & caching if needed."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        inchi_str = self._mol_to_inchi(mol)
        fp = self._get_fp_from_cache(inchi_str)
        if fp is None:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=self.radius, nBits=self.nbits
            )
            self._store_fp_in_cache(inchi_str, fp)
        return fp

    def close(self):
        """Close database connection."""
        self.conn.close()

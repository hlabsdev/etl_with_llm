# =========================================================
# loaders/parallel_excel_loader.py
# =========================================================

import pandas as pd
import numpy as np
import os
from multiprocessing import Pool, cpu_count
from typing import Dict


def _load_single_sheet(args):
    file_path, sheet_name = args
    try:
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=None,
            dtype=str,
            na_filter=False
        )

        df = df.replace([np.nan, None], "")
        df = df.astype(str)

        return sheet_name, df

    except Exception as e:
        print(f"⚠️ Erreur feuille '{sheet_name}' : {e}")
        return sheet_name, pd.DataFrame()


def load_excel_with_sheets_parallel(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load Excel in parallel (1 process per sheet)
    """
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names

    print(f"📊 Fichier chargé : {os.path.basename(file_path)}")
    print(f"📑 Feuilles trouvées : {sheet_names}")

    # → Parallel load
    args = [(file_path, sheet) for sheet in sheet_names]
    workers = min(len(sheet_names), cpu_count())

    with Pool(processes=workers) as pool:
        results = pool.map(_load_single_sheet, args)

    sheets_data = {name: df for name, df in results}

    for name, df in sheets_data.items():
        print(f"   → {name}: {df.shape[0]} lignes × {df.shape[1]} colonnes")

    return sheets_data

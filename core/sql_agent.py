"""
Natural-language SQL agent using DuckDB + Ollama (mistral).
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Any

import pandas as pd

logger = logging.getLogger("apex_ds.sql_agent")


def _clean_sql(raw: str) -> str:
    """Strip markdown fences, backticks, and SQL comments from LLM output."""
    # Remove code fences
    raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("`", "")
    # Remove -- comments
    raw = re.sub(r"--[^\n]*", "", raw)
    # Remove /* */ comments
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
    return raw.strip()


def _first_token(sql: str) -> str:
    tokens = sql.split()
    return tokens[0].upper() if tokens else ""


class SQLAgent:
    def __init__(self) -> None:
        import duckdb  # type: ignore[import]
        self._con = duckdb.connect(database=":memory:")
        self._tables: Dict[str, pd.DataFrame] = {}

    def load_dataframe(self, name: str, df: pd.DataFrame) -> None:
        self._tables[name] = df
        # Register with DuckDB
        self._con.register(name, df)

    def ask(self, question: str) -> Dict[str, Any]:
        """Convert NL question to SQL, execute, return results."""
        table_names = list(self._tables.keys())
        schema_parts = []
        for tname, tdf in self._tables.items():
            cols = ", ".join(f"{c} ({tdf[c].dtype})" for c in tdf.columns)
            schema_parts.append(f"Table '{tname}': {cols}")
        schema_str = "\n".join(schema_parts)

        system_prompt = (
            "You are a DuckDB SQL expert. Given the schema and question, output ONLY a single valid SELECT SQL statement. "
            "No explanations, no markdown, no comments, no semicolons at the end. Only SELECT statements are allowed.\n\n"
            f"Schema:\n{schema_str}\n\nQuestion: {question}\n\nSQL:"
        )

        raw_sql = ""
        try:
            import ollama  # type: ignore[import]
            resp = ollama.chat(model="mistral", messages=[{"role": "user", "content": system_prompt}])
            raw_sql = resp.get("message", {}).get("content", "") or ""
        except Exception:
            # Fallback: simple heuristic query
            if table_names:
                raw_sql = f"SELECT * FROM {table_names[0]} LIMIT 10"
            else:
                return {"success": False, "sql": "", "error": "No tables loaded and Ollama unavailable."}

        cleaned = _clean_sql(raw_sql)

        # Safety check: must start with SELECT
        if _first_token(cleaned) != "SELECT":
            return {
                "success": False,
                "sql": cleaned,
                "error": f"Generated SQL does not start with SELECT: '{cleaned[:120]}'",
            }

        try:
            result_df = self._con.execute(cleaned).df()
            return {"success": True, "sql": cleaned, "result": result_df, "error": ""}
        except Exception as exc:
            return {"success": False, "sql": cleaned, "error": str(exc)}
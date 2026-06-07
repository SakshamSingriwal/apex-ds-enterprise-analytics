import re
try:
    import ollama
    _OLLAMA_AVAILABLE = True
except Exception:
    ollama = None
    _OLLAMA_AVAILABLE = False
try:
    import duckdb
    _DUCKDB_AVAILABLE = True
except Exception:
    duckdb = None
    _DUCKDB_AVAILABLE = False
class SQLAgent:
    def __init__(self):
        self.table_name = 'data'
        if _DUCKDB_AVAILABLE:
            try:
                self.conn = duckdb.connect(':memory:')
            except Exception:
                self.conn = None
        else:
            self.conn = None
    
    def load_dataframe(self, name, df):
        self.table_name = name
        if self.conn is None:
            raise RuntimeError('duckdb is not available in this environment')
        self.conn.register(name, df)
    
    def generate_sql(self, question):
        if not _OLLAMA_AVAILABLE:
            return None, 'ollama package not installed or unavailable'
        # Get column info
        cursor = self.conn.execute(f"SELECT * FROM {self.table_name} LIMIT 0")
        columns = [desc[0] for desc in cursor.description]
        prompt = f"""Table name: {self.table_name}
Columns: {', '.join(columns)}
User question: {question}
Generate ONLY a SELECT SQL query (no DDL, no DML). Output only the SQL statement."""
        try:
            resp = ollama.chat(model='mistral', messages=[{"role": "user", "content": prompt}])
            if isinstance(resp, dict) and 'message' in resp:
                sql = resp['message'].get('content', '')
            else:
                sql = str(resp)
            # Clean SQL
            sql = re.sub(r'```sql\n?', '', sql)
            sql = re.sub(r'```', '', sql)
            sql = sql.strip()
            # Security: only allow SELECT
            if not sql.strip().upper().startswith('SELECT'):
                return None, "Only SELECT queries are allowed"
            return sql, None
        except Exception as e:
            return None, str(e)
    
    def execute_sql(self, sql):
        try:
            if self.conn is None:
                return None, 'duckdb not available'
            result = self.conn.execute(sql).df()
            return result, None
        except Exception as e:
            return None, str(e)
    
    def ask(self, question):
        if not _OLLAMA_AVAILABLE:
            return {'success': False, 'error': 'ollama package not installed or unavailable'}
        sql, error = self.generate_sql(question)
        if error:
            return {'success': False, 'error': error}
        result, exec_error = self.execute_sql(sql)
        if exec_error:
            return {'success': False, 'error': exec_error}
        return {'success': True, 'sql': sql, 'result': result}

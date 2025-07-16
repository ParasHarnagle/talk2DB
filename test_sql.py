import json
import time
import pyodbc


MSSQL_CONFIG = {
    "driver_path": "/usr/lib/x86_64-linux-gnu/odbc/libtdsodbc.so",
    "server": "172.31.110.82",
    #"server": "pandoarch-db.deepfoods.net",
    "port": 1433,
    "uid": "chatbot",
    "pwd": "chat@#$BOT639",
    "database": "APTtower_1",
    "tds_version": "7.3",
    "charset": "UTF-8"
}

GLOBAL_MSSQL_CONNECTION = None

def connect_to_mssql():
    """Establish and return a fresh connection to MSSQL database using pyodbc + FreeTDS."""
    global GLOBAL_MSSQL_CONNECTION

    def _create_new_connection():
        conn_str = (
            f"DRIVER={MSSQL_CONFIG['driver_path']};"
            f"SERVER={MSSQL_CONFIG['server']};"
            f"PORT={MSSQL_CONFIG['port']};"
            f"UID={MSSQL_CONFIG['uid']};"
            f"PWD={MSSQL_CONFIG['pwd']};"
            f"DATABASE={MSSQL_CONFIG['database']};"
            f"TDS_Version={MSSQL_CONFIG['tds_version']};"
            f"Charset={MSSQL_CONFIG['charset']};"
        )
        return pyodbc.connect(conn_str, autocommit=True)

    # Check if existing connection is alive
    if GLOBAL_MSSQL_CONNECTION:
        try:
            cursor = GLOBAL_MSSQL_CONNECTION.cursor()
            cursor.execute("SELECT 1")
            return GLOBAL_MSSQL_CONNECTION
        except Exception as e:
            print(f"⚠️ Old connection invalid: {e}")
            try:
                GLOBAL_MSSQL_CONNECTION.close()
            except:
                pass
            GLOBAL_MSSQL_CONNECTION = None

    # Create new connection
    try:
        GLOBAL_MSSQL_CONNECTION = _create_new_connection()
        print("✅ Reconnected to MSSQL")
        return GLOBAL_MSSQL_CONNECTION
    except Exception as e:
        print(f"[❌ MSSQL connection failed]: {e}")
        raise


def get_full_schema():
    try:
        print("📘 Fetching full schema...")
        conn = connect_to_mssql()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE 
                FROM INFORMATION_SCHEMA.COLUMNS
                ORDER BY TABLE_NAME, ORDINAL_POSITION;
            """)
            rows = cursor.fetchall()
            schema = {}
            for table, column, dtype in rows:
                schema.setdefault(table, []).append({"column": column, "type": dtype})
            return schema
    except Exception as e:
        return {"error": str(e)}


def run_query_tool_with_fallback(**kwargs):
    """SQL executor for MSSQL Server"""
    sql = kwargs.get("sql_query")
    fetch_schema_only = kwargs.get("fetch_schema_only", False)
    
    start_time = time.time()
    
    if fetch_schema_only:
        schema = get_full_schema()
        return json.dumps({
            "schema": schema,
            "execution_time_ms": int((time.time() - start_time) * 1000)
        }, indent=2)
    
    try:
        with connect_to_mssql() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description]
                result = [serialize_row(row, col_names) for row in rows]
                
                return json.dumps({
                    "success": True,
                    "rows": result,
                    "row_count": len(result),
                    "execution_time_ms": int((time.time() - start_time) * 1000)
                }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "schema": get_full_schema(),
            "execution_time_ms": int((time.time() - start_time) * 1000),
            "suggestion": "SQL execution failed. Schema returned to help regenerate query."
        }, indent=2)

from datetime import date, datetime

def serialize_row(row, columns):
    def safe(val):
        if isinstance(val, (date, datetime)):
            return val.isoformat()
        return val
    return {col: safe(val) for col, val in zip(columns, row)}

def test_connection():
    try:
        print("Connecting....")
        conn = connect_to_mssql()
        print("✅ Connected to MSSQL successfully!")
        conn.close()
    except Exception as e:
        print("❌ Connection failed:", str(e))


if __name__ == "__main__":
    test_connection()
    #schema_result = run_query_tool_with_fallback(fetch_schema_only=True)
    #print("\n📘 Full Schema Info:\n", schema_result)
    print("\n🧪 Running test query: SELECT TOP 5 * FROM CUSTOMER")
    result = run_query_tool_with_fallback(
        sql_query="SELECT TOP 5 * FROM CUSTOMER"
    )
    print("\n📦 Query Result:\n", result)
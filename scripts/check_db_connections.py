"""
Check active database connections on Supabase.
"""

from nba_ou.postgre_db.config.db_config import connect_nba_db


def check_active_connections():
    """Query to see how many connections are currently active."""
    conn = None
    try:
        conn = connect_nba_db()
        with conn.cursor() as cur:
            # Check active connections for your database
            cur.execute("""
                SELECT 
                    datname,
                    usename,
                    application_name,
                    client_addr,
                    state,
                    COUNT(*) as connection_count
                FROM pg_stat_activity
                WHERE datname = current_database()
                GROUP BY datname, usename, application_name, client_addr, state
                ORDER BY connection_count DESC;
            """)

            results = cur.fetchall()
            print("\n=== Active Database Connections ===")
            for row in results:
                print(
                    f"DB: {row[0]}, User: {row[1]}, App: {row[2]}, "
                    f"Client: {row[3]}, State: {row[4]}, Count: {row[5]}"
                )

            # Total count
            cur.execute("""
                SELECT COUNT(*) as total
                FROM pg_stat_activity
                WHERE datname = current_database();
            """)
            total = cur.fetchone()[0]
            print(f"\nTotal connections to your database: {total}")

            cur.execute("""
                SELECT
                    pid,
                    usename,
                    application_name,
                    client_addr,
                    state,
                    now() - backend_start AS backend_age,
                    now() - xact_start AS transaction_age,
                    wait_event_type,
                    wait_event,
                    left(regexp_replace(query, '\\s+', ' ', 'g'), 220) AS query
                FROM pg_stat_activity
                WHERE datname = current_database()
                    AND pid <> pg_backend_pid()
                    AND (
                        state = 'idle in transaction'
                        OR application_name = 'Supavisor'
                    )
                ORDER BY
                    state = 'idle in transaction' DESC,
                    xact_start NULLS LAST,
                    backend_start;
            """)

            details = cur.fetchall()
            print("\n=== Supavisor / Idle Transaction Details ===")
            if not details:
                print("No Supavisor or idle-in-transaction sessions found.")
            for row in details:
                (
                    pid,
                    user,
                    app_name,
                    client_addr,
                    state,
                    backend_age,
                    transaction_age,
                    wait_event_type,
                    wait_event,
                    query,
                ) = row
                print(
                    f"PID: {pid}, User: {user}, App: {app_name}, "
                    f"Client: {client_addr}, State: {state}, "
                    f"Backend age: {backend_age}, Transaction age: {transaction_age}, "
                    f"Wait: {wait_event_type}/{wait_event}"
                )
                print(f"  Last query: {query}")

    except Exception as e:
        print(f"Error checking connections: {e}")
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    check_active_connections()

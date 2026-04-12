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

    except Exception as e:
        print(f"Error checking connections: {e}")
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    check_active_connections()

"""
Terminate idle database connections on the configured database.
"""

from nba_ou.postgre_db.config.db_config import connect_nba_db


def close_idle_connections() -> None:
    """Terminate idle and idle-in-transaction sessions for the current database."""
    conn = None
    try:
        conn = connect_nba_db()
        conn.autocommit = True

        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    pid,
                    datname,
                    usename,
                    application_name,
                    client_addr,
                    state
                FROM pg_stat_activity
                WHERE datname = current_database()
                    AND pid <> pg_backend_pid()
                    AND state LIKE 'idle%'
                ORDER BY pid;
            """)

            idle_connections = cur.fetchall()

            results = []
            for row in idle_connections:
                pid, datname, usename, app_name, client_addr, state = row
                try:
                    cur.execute(
                        "SELECT pg_terminate_backend(%s) AS terminated;",
                        (pid,),
                    )
                    terminated = cur.fetchone()[0]
                    error = None
                except Exception as exc:
                    terminated = False
                    error = str(exc)

                results.append(
                    (
                        pid,
                        datname,
                        usename,
                        app_name,
                        client_addr,
                        state,
                        terminated,
                        error,
                    )
                )

        print("\n=== Closing Idle Database Connections ===")
        if not results:
            print("No idle connections found.")
            return

        terminated_count = 0
        for row in results:
            pid, datname, usename, app_name, client_addr, state, terminated, error = row
            if terminated:
                terminated_count += 1

            print(
                f"PID: {pid}, DB: {datname}, User: {usename}, App: {app_name}, "
                f"Client: {client_addr}, State: {state}, Terminated: {terminated}"
            )
            if error:
                print(f"  Error: {error}")

        print(f"\nTerminated idle connections: {terminated_count} of {len(results)}")

    except Exception as e:
        print(f"Error closing idle connections: {e}")
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    close_idle_connections()


from config import PIPELINE_TABLE as table

def fetch_next_job(conn, stage, status='pending'):
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT * FROM {table}
            WHERE stage = %s AND status = %s
            ORDER BY priority DESC, updated_at DESC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        """, (stage, status))
        job = cur.fetchone()
        if job:
            cur.execute(f"""
                UPDATE {table}
                SET status = 'in_progress', updated_at = NOW()
                WHERE id = %s
            """, (job['id'],))
            conn.commit()
        return job

def update_job_stage(conn, job_id, new_stage, new_status='pending', addons=[]):
    addon_conditions = ', '.join(addons)
    with conn.cursor() as cur:
        cur.execute(f"""
            UPDATE {table}
            SET stage = %s, status = %s, updated_at = NOW() {', ' + addon_conditions if addons else ''}
            WHERE id = %s
        """, (new_stage, new_status, job_id))
        conn.commit()

def mark_job_done(conn, job_id, addons=[]):
    addon_conditions = ', '.join(addons)
    with conn.cursor() as cur:
        cur.execute(f"""
            UPDATE {table}
            SET status = 'done', updated_at = NOW() {', ' + addon_conditions if addons else ''}
            WHERE id = %s
        """, (job_id,))
        conn.commit()

def mark_job_failed(conn, job_id, addons=[]):
    addon_conditions = ', '.join(addons)
    with conn.cursor() as cur:
        cur.execute(f"""
            UPDATE {table}
            SET status = 'failed', updated_at = NOW() {', ' + addon_conditions if addons else ''}
            WHERE id = %s
        """, (job_id,))
        conn.commit()

def update_job_priority(conn, job_id, new_priority):
    with conn.cursor() as cur:
        cur.execute(f"""
            UPDATE {table}
            SET priority = %s, updated_at = NOW()
            WHERE id = %s
        """, (new_priority, job_id))
        conn.commit()

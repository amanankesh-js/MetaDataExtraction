import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# LOAD ENV VARIABLES
# ──────────────────────────────────────────────
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")

AUDIO_TABLE = "pg_audio_table"
AUDIO_COLLECTION = "milvus_audio_table"

VIDEO_TABLE = "pg_frame_table"
VIDEO_COLLECTION = "milvus_frame_table"
# ──────────────────────────────────────────────
# INIT CONNECTIONS
# ──────────────────────────────────────────────
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
model = SentenceTransformer("all-mpnet-base-v2")


# ──────────────────────────────────────────────
# PostgreSQL Connection
# ──────────────────────────────────────────────
def get_pg_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )


def search_milvus(collection_name, embedding, top_k=5, threshold=None):
    collection = Collection(collection_name)
    collection.load()

    results = collection.search(
        data=[embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["id"]
    )
    if threshold is not None:
        ids = []
        for hits in results:
            for hit in hits:
                distance = hit.distance
                if distance <= threshold:         # ✔ keep only similar ones
                    ids.append(hit.id)
    else:
        ids = [hit.id for hit in results[0]]
    return ids



# ──────────────────────────────────────────────
# PostgreSQL Metadata Fetch
# ──────────────────────────────────────────────
def fetch_metadata(ids, table):
    if not ids:
        return pd.DataFrame()

    conn = get_pg_conn()
    placeholders = ",".join(["%s"] * len(ids))
    query = f"SELECT * FROM {table} WHERE id IN ({placeholders});"
    df = pd.read_sql(query, conn, params=ids)
    conn.close()
    return df


# ──────────────────────────────────────────────
# FULL MULTIMODAL SEARCH
# ──────────────────────────────────────────────
def multimodal_search(query_text, top_k=5):
    # 1️⃣ Embed query using SentenceTransformer
    query_embedding = model.encode(query_text).tolist()

    # 2️⃣ Milvus search
    audio_ids = search_milvus(AUDIO_COLLECTION, query_embedding, top_k)
    video_ids = search_milvus(VIDEO_COLLECTION, query_embedding, top_k)
    print("video ids: ", video_ids)
    # ids = [*audio_ids, *video_ids]
    audio_df = fetch_metadata(audio_ids, AUDIO_TABLE)
    video_df = fetch_metadata(video_ids, VIDEO_TABLE)

    # 4️⃣ Return in final JSON-style format
    return {
        "query_embedding_dim": len(query_embedding),
        "audio_results": audio_df.to_dict(orient="records"),
        "video_results": video_df.to_dict(orient="records")
    }


# ──────────────────────────────────────────────
# USAGE
# ──────────────────────────────────────────────
if __name__ == "__main__":
    query = "a screen"
    while True:
        query = input("Enter search query: ")
        if not query.strip():
            break
        results = multimodal_search(query, top_k=2)

        print("\n🎤 AUDIO RESULTS")
        for r in results["audio_results"]:
            print("content_summary: ", r["content_summary"])
            print("transcript: ", r["transcript_full_text"], "...\n")

        print("\n🎬 VIDEO RESULTS")
        for r in results["video_results"]:
            print(r)

        print("\nEmbedding dimension:", results["query_embedding_dim"])

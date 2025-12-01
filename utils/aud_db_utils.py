from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusClient

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.extras import execute_values

from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
import ast
import json
from sentence_transformers import SentenceTransformer   


embedding_model = SentenceTransformer("all-mpnet-base-v2")

load_dotenv(find_dotenv())

feature_columns = [
    'overall_sentiment', 
    'overall_audio_emotion', 
    'background_instruments',
    'background_emotion',
    'brand_utterances'
]
def get_pg_conn():
    return psycopg2.connect(
        dbname=os.environ['DB_NAME'],
        user=os.environ['DB_USER'],
        password=os.environ['PASSWORD'],
        host=os.environ['HOST'],
        port=os.environ['PORT'],
        cursor_factory=RealDictCursor
    )

def get_milvus_conn():
    MILVUS_HOST = os.environ['MILVUS_HOST']
    MILVUS_PORT = os.environ['MILVUS_PORT']
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    return True

def get_audio_collection(COLLECTION_NAME, feature_embed=False, drop=False,):
    if drop and utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"Dropped existing collection: {COLLECTION_NAME}")

    if not feature_embed:

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=500, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="movie", dtype=DataType.VARCHAR, max_length=500, enable_analyzer=True, enable_match=True)
        ]
        schema = CollectionSchema(fields, description="Text embeddings collection")
    else:
        feature_fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=500, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
        ]
        schema = CollectionSchema(feature_fields, description="Feature embeddings collection")

    # Create collection
    collection = Collection(name=COLLECTION_NAME, schema=schema)  # Get or create
    total_entities = collection.num_entities
    if total_entities == 0:
        collection.create_index(field_name="embedding", index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 64, "efConstruction": 1000}})

    collection.load()
    print(COLLECTION_NAME, ":", total_entities)
    return collection



def safe_eval(x):
    """Safely evaluate stringified lists/dicts, else return as-is."""
    try:
        if isinstance(x, str):
            x = x.strip()
            if x.startswith('[') or x.startswith('{'):
                return ast.literal_eval(x)
        return x
    except Exception:
        return x


def to_pg_array(value):
    """Convert Python list or stringified list to PostgreSQL array literal."""
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, str):
            value = value.strip()
            if value.startswith('[') and value.endswith(']'):
                value = ast.literal_eval(value)
            elif value.startswith('{') and value.endswith('}'):
                # JSON dict, not a list — don't convert to array
                return None
        if isinstance(value, list):
            return '{' + ','.join(f'"{str(v)}"' for v in value) + '}'
        return None
    except Exception:
        return None


def create_audio_table(conn, table, drop=False):
    """
    Create a PostgreSQL table for structured audio and transcript feature outputs.
    """
    cursor = conn.cursor()
    if drop:
        cursor.execute(f"DROP TABLE IF EXISTS {table};")

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table} (
        id TEXT PRIMARY KEY,
        movie TEXT,
        chunk_id TEXT,
        model TEXT,
        temperature DECIMAL(4, 2),

        -- Transcript-level fields
        content_summary TEXT,
        speakers JSONB,
        transcript_full_text TEXT,
        translation_approximate TEXT,

        -- Audio-level fields
        audio_events JSONB,
        overall_tone TEXT,
        overall_sentiment TEXT[],
        overall_audio_emotion TEXT[],
        tone_timestamp_start TEXT,
        tone_timestamp_end TEXT,

        -- Background features
        background_type TEXT,
        background_description TEXT,
        background_instruments TEXT[],
        background_emotion TEXT[],

        -- Song / music features
        song_transcript TEXT,
        song_timestamp_start TEXT,
        song_timestamp_end TEXT,
        song_event TEXT,
        song_placement_flag TEXT,

        -- Brand utterances
        brand_utterances TEXT[],

        created_at TIMESTAMP DEFAULT NOW()
    );
    """

    cursor.execute(create_table_query)
    conn.commit()
    cursor.close()
    print(f"✅ Successfully created audio features table: {table}")
    return True


def insert_to_audio_db(conn, table, df):
    """
    Insert structured audio + transcript feature data into PostgreSQL table.
    """
    cursor = conn.cursor()
    df['movie'] = df['movie'].str.upper()
    if 'id' not in df.columns:
        df['id'] = df.apply(lambda x: f"{x['movie']}_{x['chunk_id']}", axis=1)


    # Safe eval & NaN cleanup
    df = df.map(lambda x: safe_eval(x))
    df = df.where(pd.notna(df), None)

    # Columns aligned with DB schema
    selected_columns = [
        "id",
        "movie",
        "chunk_id",
        # "model",
        # "temperature",
        "content_summary",
        "speakers",
        "transcript_full_text",
        "translation_approximate",
        "audio_events",
        "overall_tone",
        "overall_sentiment",
        "overall_audio_emotion",
        "tone_timestamp_start",
        "tone_timestamp_end",
        "background_type",
        "background_description",
        "background_instruments",
        "background_emotion",
        "song_transcript",
        "song_timestamp_start",
        "song_timestamp_end",
        "song_event",
        "song_placement_flag",
        "brand_utterances"
    ]

    # Check for missing columns
    missing_cols = [c for c in selected_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in DataFrame: {missing_cols}")

    # Handle JSONB and array conversions
    jsonb_cols = ["speakers", "audio_events"]
    array_cols = [
        "overall_sentiment",
        "overall_audio_emotion",
        "background_instruments",
        "background_emotion",
        "brand_utterances"
    ]

    # for col in jsonb_cols:
    #     if col in df.columns:
    #         df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

    for col in jsonb_cols:
        def to_json(v):
            if v is None:
                return None
            if isinstance(v, (dict, list)):
                return json.dumps(v)
            if isinstance(v, str):
                v = v.strip()
                # If already JSON-like
                if (v.startswith('{') and v.endswith('}')) or (v.startswith('[') and v.endswith(']')):
                    return v
                # convert string to JSON string
                return json.dumps(v)
            # Fallback for unexpected formats
            return json.dumps(v)

    df[col] = df[col].apply(to_json)


    df[col] = df[col].apply(to_json)

    for col in array_cols:
        if col in df.columns:
            df[col] = df[col].apply(to_pg_array)

    # Convert to list of tuples
    data = list(df[selected_columns].itertuples(index=False, name=None))
    #model, temperature,
    insert_query = f"""
        INSERT INTO {table} (
            id, movie, chunk_id, 
            content_summary, speakers, transcript_full_text, translation_approximate,
            audio_events, overall_tone, overall_sentiment, overall_audio_emotion,
            tone_timestamp_start, tone_timestamp_end,
            background_type, background_description, background_instruments, background_emotion,
            song_transcript, song_timestamp_start, song_timestamp_end, song_event, song_placement_flag,
            brand_utterances
        ) VALUES %s
        ON CONFLICT (id) DO NOTHING;
    """
    print(df[["movie", "overall_sentiment", "overall_audio_emotion"]].head(10))
    execute_values(cursor, insert_query, data)
    conn.commit()
    cursor.close()

    print(f"✅ Successfully inserted {len(data)} audio feature rows into {table}")
    return True


def insert_to_audio_collection(collection, df, feature='content_summary'):
    df['movie'] = df['movie'].str.upper()
    df['ids'] = df[['movie', 'chunk_id']].apply(lambda x: f"{x['movie']}_{x['chunk_id']}", axis=1)
    id_list = df["ids"].tolist()

    existing = collection.query(f"id in {id_list}", output_fields=["id"])
    if len(id_list) == len(existing):
        print("Skipping: ", df['movie'].unique())
        return

    embeddings = embedding_model.encode(df[feature].tolist(), show_progress_bar=True).tolist()
    
    # Insert data
    if feature == 'content_summary':
        data = [id_list, embeddings, df['movie'].tolist()]
    else:
        data = [id_list, embeddings]
   
    print("Inserting to collection:", collection.name, data[0][:5], data[1][1][:5], data[2][:5] if len(data) >2 else "")
    collection.insert(data)
    collection.flush()
    return True


def insert_features_to_audio_collection(conn, table, feature_columns, drop=False, drop_index=False):
    cursor = conn.cursor()
    for feature in feature_columns:
        collection = get_audio_collection(feature, drop=drop, feature_embed=True)

        selection_query = f"""
        SELECT DISTINCT unnest({feature}) AS {feature}
        FROM {table};
        """

        cursor.execute(selection_query)
        columns = [desc[0] for desc in cursor.description]
        metadata_results = cursor.fetchall()
        meta_df = pd.DataFrame(metadata_results, columns=columns)
        id_list = meta_df[feature].tolist()
        
        existing = collection.query(f"id in {id_list}", output_fields=["id"])
        existing_ids = [e['id'] for e in existing]
        complement_ids = [idx for idx in id_list if idx not in existing_ids]
        embeddings = embedding_model.encode(complement_ids, show_progress_bar=True).tolist()
        data = [complement_ids, embeddings]

        if data[0]:
            collection.insert(data)
            collection.flush()

        total_entities = collection.num_entities
        if drop_index:
            collection.release()
            collection.drop_index()
            collection.create_index(field_name="embedding", index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 64, "efConstruction": 1000}})
        print(f"{feature} collection:", total_entities)

    cursor.close()
    return True

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    import pandas as pd

    # selected_columns = [
    #     "audio_events",
    #     "overall_tone",
    #     "overall_sentiment",
    #     "overall_audio_emotion",
    #     "tone_timestamp_start",
    #     "tone_timestamp_end",
    #     "background_type",
    #     "background_description",
    #     "background_instruments",
    #     "background_emotion",
    #     "song_transcript",
    #     "song_timestamp_start",
    #     "song_timestamp_end",
    #     "song_event",
    #     "song_placement_flag",
    #     "brand_utterances"
    #     "content_summary",
    #     "speakers",
    #     "transcript_full_text",
    #     "translation_approximate"
    # ]

    feature_columns = [
        'overall_sentiment', 
        'overall_audio_emotion', 
        'background_instruments',
        'background_emotion',
        'brand_utterances'
    ]


    conn = get_pg_conn()
    get_milvus_conn()
    
    pg_table_name = "pg_table"
    collection_name = 'audio_db'
    processed_output = '/Users/amana1/working_dir/zMetaDataExtraction/output/TX_MASTER_FC_Anupamaa_SH4164_S1_E1599_DYN1492441_v2_763507606_900790931_877219001/2025-11-20_19-42-56/video_analysis/prompt1_prompt2_merged.xlsx'

    create_audio_table(conn, pg_table_name)
    collection = get_audio_collection(collection_name, drop=False)
    
    df = pd.read_excel(processed_output)
    insert_to_audio_db(conn, pg_table_name, df)
    print("Inserted to audio db table:", pg_table_name)
    insert_to_audio_collection(collection, df, embedding_model, feature='content_summary')
    print("Inserted to audio milvus collection:", collection_name)
    insert_features_to_audio_collection(conn, pg_table_name, embedding_model, feature_columns)
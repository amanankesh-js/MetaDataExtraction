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
    'person_emotion',
    'scene_emotion',
    'brand_based_on_logos'
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
    # Milvus connection details
    MILVUS_HOST = os.environ['MILVUS_HOST'] # "localhost"
    MILVUS_PORT = os.environ['MILVUS_PORT']
    # Connect to Milvus
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    return True

def get_collection(COLLECTION_NAME, drop=False, feature_embed=False):
    if drop and utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"Dropped existing collection: {COLLECTION_NAME}")

    # exit()

    if not feature_embed:
        # Define schema
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

def create_frame_table(conn, table, drop=False):
    # noticeable_objects_bottom TEXT[],
    # unnoticeable_objects_top TEXT[],
    # model TEXT,
    # temperature DECIMAL(3, 2),
    cursor = conn.cursor()
    if drop:
        cursor.execute(f"DROP TABLE IF EXISTS {table};")

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS  {table} (
        id TEXT PRIMARY KEY,
        movie TEXT,
        chunk_id TEXT,
        objects TEXT[],
        object_count JSONB,
        gender TEXT[],
        ocr_text TEXT[],
        noticeable JSONB,
        unnoticeable JSONB,
        scene_emotion TEXT[],
        age_group TEXT[],
        scene_tags TEXT[],
        scene_label TEXT,
        weather TEXT,
        day_night INT,
        person_emotion TEXT[],
        clarity_of_image TEXT,
        actions TEXT[],
        celebrity TEXT[],
        timestamp TEXT,
        brand_based_on_logos TEXT[],
        location TEXT[],
        setting TEXT[],
        description TEXT,
        sentiment TEXT
    );
    """

    # Execute the query
    cursor.execute(create_table_query)

    # Commit changes and close connection
    conn.commit()
    cursor.close()
    print("Successfully created DB")
    return True

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

def insert_to_db(conn, table, df):
    cursor = conn.cursor()
    df['movie'] = df['movie'].str.upper()
    df['id'] = df.apply(lambda x: f"{x['movie']}_{x['chunk_id']}", axis=1)
    df = df.map(safe_eval)
    df = df.where(pd.notna(df), None)

    df.loc[df['day_night'].isna(), 'day_night'] = -100
    df['day_night'] = df['day_night'].apply(lambda x: sum(x)/len(x) if isinstance(x, list) else x)
    df['ocr_text'] = df['ocr_text'].apply(lambda x: x if isinstance(x, list) else [str(x)])
    
    selected_columns=[
        "id",
        "movie",
        "chunk_id",
        "objects",
        "object_count",
        "gender",
        "ocr_text",
        "noticeable",
        "unnoticeable",
        "scene_emotion",
        "age_group",
        "scene_tags",
        "scene_label",
        "weather",
        "day_night",
        "person_emotion",
        "clarity_of_image",
        "actions",
        "celebrity",
        "timestamp",
        "brand_based_on_logos",
        "location",
        "setting",
        "description",
        "sentiment"
    ]

    jsonb_cols = [
        "noticeable", "unnoticeable", "object_count"
    ]

    array_cols = [
        "objects", "gender", "ocr_text", "scene_emotion", "age_group",
        "scene_tags", "person_emotion", "actions", "celebrity",
        "brand_based_on_logos", "location", "setting"
    ]
    for col in jsonb_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
    
    for col in array_cols:
        if col in df.columns:
            df[col] = df[col].apply(to_pg_array)
    
    data = list(df[selected_columns].itertuples(index=False, name=None))
    #model, temperature,  noticeable_objects_top, noticeable_objects_bottom, #unnoticeable_objects_top, unnoticeable_objects_bottom, 
                         
    execute_values(cursor, f"INSERT INTO {table} \
                         (id, movie, chunk_id,  \
                         objects, object_count, gender, ocr_text, \
                         noticeable, unnoticeable, \
                         scene_emotion, age_group, scene_tags, scene_label, \
                         weather, day_night, person_emotion, clarity_of_image, \
                         actions, celebrity, timestamp, brand_based_on_logos, \
                         location, setting, description, sentiment) VALUES %s \
             ON CONFLICT (id) DO NOTHING;", data)

    print("*"*10, "Commiting!", "*"*10)
    conn.commit()
    cursor.close()
    return True

def insert_to_collection(collection, df, feature='description'):
    df['movie'] = df['movie'].str.upper()
    df['ids'] = df[['movie', 'chunk_id']].apply(lambda x: f"{x['movie']}_{x['chunk_id']}", axis=1)
    id_list = df["ids"].tolist()

    existing = collection.query(f"id in {id_list}", output_fields=["id"])
    if len(id_list) == len(existing):
        print("Skipping: ", df['movie'].unique())
        return

    embeddings = embedding_model.encode(df[feature].tolist(), show_progress_bar=True).tolist()
    
    # Insert data
    if feature == 'description':
        data = [id_list, embeddings, df['movie'].tolist()]
    else:
        data = [id_list, embeddings]
    collection.insert(data)
    collection.flush()
    return True

def insert_features_to_collection(conn, table, feature_columns, drop=False, drop_index=False):
    cursor = conn.cursor()
    for feature in feature_columns:
        collection = get_collection(feature, drop=drop, feature_embed=True)

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

def safe_eval(x):
    try:
        if isinstance(x, str) and '[' in x:
            return ast.literal_eval(x)
        elif isinstance(x, str) and '{' in x:
            return json.dumps(x)
        else:
            return x
    except Exception as e:
        print(e)
        return x


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    import pandas as pd
        
    """  
        "id",
        "movie",
        "chunk_id",
        "objects",
        "object_count",
        "gender",
        "ocr_text",
        "noticeable",
        "unnoticeable",
        "scene_emotion",
        "age_group",
        "scene_tags",
        "scene_label",
        "weather",
        "day_night",
        "person_emotion",
        "clarity_of_image",
        "actions",
        "celebrity",
        "timestamp",
        "brand_based_on_logos",
        "location",
        "setting",
        "description",
        "sentiment"


        # model TEXT,
        # temperature DECIMAL(3, 2),
    """

    feature_columns = [
        'person_emotion',
        'scene_emotion',
        'brand_based_on_logos'
    ]


    conn = get_pg_conn()
    get_milvus_conn()
    processed_output = '/Users/amana1/working_dir/zMetaDataExtraction/output/TX_MASTER_FC_Anupamaa_SH4164_S1_E1599_DYN1492441_v2_763507606_900790931_877219001/2025-11-20_19-42-56/video_analysis/prompt1_prompt2_merged.xlsx'
    
    pg_table_name = "pg_frame_table"
    collection_name = 'frame_table'

    create_table(conn, pg_table_name, drop_if_exists=True)
    collection = get_collection(collection_name, drop=False)
    
    df = pd.read_excel(processed_output)
    print(df.columns)
    insert_to_db(conn, pg_table_name, df)
    print("Inserted to audio db table:", pg_table_name)
    insert_to_collection(collection, df, embedding_model, feature='description')
    print("Inserted to audio milvus collection:", collection_name)
    insert_features_to_collection(conn, pg_table_name, embedding_model, feature_columns)
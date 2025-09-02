from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import sqlalchemy as sa
import json
from typing import Dict, Any, Tuple, List


embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# persistent chroma DB
db = Chroma(
    collection_name='investors',
    embedding_function=embed_model,
    persist_directory='./chroma_db'
)



def init_mysql(uri: str, table_name: str = 'investors'):
    """Initialize MySQL connection"""
    engine = sa.create_engine(uri)
    metadata = sa.MetaData()
    investors_table = sa.Table(table_name, metadata, autoload_with=engine)
    return engine, investors_table


def prepare_investor_text(investor: Dict[str, Any]) -> str:
    """Build text to embed from investor data."""
    tags = investor.get('industry_tags', '')
    if isinstance(tags, list):
        tags = " ".join(tags)
    thesis = investor.get('thesis_text', '')
    return f"{thesis} {tags}".strip()


def add_investor_to_chroma(investor: Dict[str, Any]):
    """Add one investor to Chroma DB and persist."""
    text_to_embed = prepare_investor_text(investor)

    # normalize/serialize fields that may be lists
    stage_focus = investor.get('stage_focus')
    if isinstance(stage_focus, list):
        stage_focus_serialized = json.dumps(stage_focus)
    else:
        stage_focus_serialized = stage_focus

    industry_tags = investor.get('industry_tags')
    if isinstance(industry_tags, list):
        industry_tags_serialized = json.dumps(industry_tags)
    else:
        industry_tags_serialized = industry_tags

    metadata = {
        "id": str(investor.get("id", "")),
        "name": investor.get("name", ""),
        "stage_focus": stage_focus_serialized,
        "ticket_min_usd": int(investor.get("ticket_min_usd", 0)),
        "ticket_max_usd": int(investor.get("ticket_max_usd", 0)),
        "industry_tags": industry_tags_serialized,
        "thesis_text": investor.get("thesis_text", ""),
        "contact_email": investor.get("contact") or investor.get("contact_email")
    }

    db.add_texts([text_to_embed], metadatas=[metadata], ids=[str(investor.get("id", ""))])
    db.persist()


def fetch_investor_from_db(engine, investors_table) -> List[Dict[str, Any]]:
    """Fetch investors from MySQL DB."""
    with engine.connect() as conn:
        rows = conn.execute(sa.select(investors_table)).fetchall()
        return [dict(r._mapping) for r in rows]


def recommend_investors(startup: Dict[str, Any], k: int = 3) -> List[Tuple[Any, float]]:
    """Query Chroma and filter results.

    Builds a robust query string and filters results by ticket size and stage.
    """
    # build query text defensively
    parts = [
        startup.get("problem_statement", ""),
        startup.get("solution_description", ""),
    ]
    industry_tags = startup.get("industry_tags", [])
    if isinstance(industry_tags, list):
        parts.append(" ".join(industry_tags))
    else:
        parts.append(str(industry_tags))

    parts.append(f"Stage: {startup.get('stage', '')}")
    parts.append(f"Funding ask: {startup.get('funding_ask_usd', '')}")
    query_text = " ".join([p for p in parts if p])

    results = db.similarity_search_with_score(query_text, k=k)

    # filter by ticket size and stage
    filtered: List[Tuple[Any, float]] = []
    for doc, score in results:
        try:
            ticket_min = int(doc.metadata.get("ticket_min_usd", 0))
            ticket_max = int(doc.metadata.get("ticket_max_usd", 0))
        except Exception:
            # skip entries with invalid ticket info
            continue

        raw_stage = doc.metadata.get("stage_focus")
        # determine stages list from stored metadata
        if isinstance(raw_stage, list):
            stages = raw_stage
        elif isinstance(raw_stage, str):
            raw = raw_stage.strip()
            if raw.startswith("[") and raw.endswith("]"):
                try:
                    stages = json.loads(raw)
                except Exception:
                    stages = [raw]
            else:
                stages = [raw]
        else:
            stages = [str(raw_stage)]

        funding_ask = startup.get("funding_ask_usd")
        stage = startup.get("stage")

        if funding_ask is None or stage is None:
            # if startup doesn't provide these, include the hit but with score
            filtered.append((doc, score))
            continue

        if (
            ticket_min <= funding_ask <= ticket_max
            and stage in stages
        ):
            filtered.append((doc, score))

    return filtered
    

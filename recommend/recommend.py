from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
import json


# load the embedding model
embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en")

# load investors data
with open("investor_profiles.json", "r") as f:
    data = json.load(f)

investors = data['investors']

# load startups data
...

# create chroma database
db = Chroma(collection_name='investors', embedding_function=embed_model)

# add investors
for inv in investors:
    text_to_embed = inv['thesis_text'] + " " + " ".join(inv["industry_tags"])
    
    metadata = {
        "id": inv["id"],
        "name": inv["name"],
        "stage_focus": ", ".join(inv["stage_focus"]),
        "ticket_min_usd": inv["ticket_min_usd"],
        "ticket_max_usd": inv["ticket_max_usd"],
        "preferred_geographies": ", ".join(inv["preferred_geographies"]),
        "industry_tags": ", ".join(inv["industry_tags"]),
        "thesis_text": inv["thesis_text"],
        "contact_email": inv["contact"]["email"]
    }
    
    db.add_texts([text_to_embed], metadatas=[metadata], ids=[inv["id"]])


# construct a query for a startups
query_text = (
    startup["problem_statement"] + " " +
    startup["solution_description"] + " " +
    " ".join(startup["industry_tags"]) + " " +
    f"Stage: {startup['stage']} " +
    f"Funding ask: ${startup['funding_ask_usd']} " 
)

# similarity search with a score
results = db.(query_text, k=2)

for match, score in results:
    print(f"{match.metadata['name']} — Score: {score:.4f}")

# apply addtional filters
filtered = [
    r for r in results
    if r[0].metadata["ticket_min_usd"] <= startup["funding_ask_usd"] <= r[0].metadata["ticket_max_usd"]
    and startup["stage"] in r[0].metadata["stage_focus"]
]

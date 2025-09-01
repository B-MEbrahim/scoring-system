from recommend import init_mysql, fetch_investors_from_db, add_investor_to_chroma, recommend_investors


# init MySQL connection 
engine, invesotrs_table = init_mysql("mysql+pymysql://user:pass@localhost/dbname")


# load existing investors in Chroma
investors = fetch_investors_from_db(engine, invesotrs_table)
for inv in investors:
    add_investor_to_chroma(inv)


# recommend investors for a startup
startup = {
    "problem_statement": "Plastic pollution in oceans is increasing...",
    "solution_description": "We recycle plastic into sustainable furniture...",
    "industry_tags": ["sustainability", "manufacturing"],
    "stage": "seed",
    "funding_ask_usd": 200000
}

matches = recommend_investors(startup, k=3)
for doc, score in matches:
    print(f"{doc.metadata['name']} (Score: {score:.4f})")
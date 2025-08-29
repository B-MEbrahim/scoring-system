import re
import json
import pandas as pd 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


scoring_fw = pd.read_csv('scoring_fw_clean.csv')

extraction_path = "final_extraction_result.json"
with open(extraction_path, "r") as f:
    extraction_data = json.load(f)

analysis_path = "VC_Pitch_Analysis_20250826_112515_analysis.json"
with open(analysis_path, "r") as f:
    analysis_data = json.load(f)


def text_to_score(text):
    text = text.lower()
    base = sentiment_score(text)

    good_words = ["strong", "excellent", "clear advantage", "high", "promising", "good", "potential", "moderate", 'substantial']
    bad_words = ["weak", "lack", "insufficient", "gap", "poor", "critical issue", "major risk", "bad", "unclear"]

    keyword_adj = 0  

    for word in good_words:
        if word in text:
            keyword_adj += 0.5

    for word in bad_words:
        if word in text:
            keyword_adj -= 1

    final_score = round(base + keyword_adj)

    return max(1, min(final_score, 5))  # Clamp between 1 and 5


def sentiment_score(text):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    compound = vs["compound"]

    if compound >= 0.6:
        return 5
    elif compound >= 0.2:
        return 4
    elif compound > -0.2:
        return 3
    elif compound > -0.6:
        return 2
    else:
        return 1


def gated_numeric_score(text, val, thresholds):
    sentiment = sentiment_score(text)
    if sentiment >= 4:
        return numeric_to_score(val, thresholds)
    elif sentiment == 3:
        return min(numeric_to_score(val, thresholds), 3)
    else:
        return 2
    

def numeric_to_score(value, thresholds):
    for limit, score in thresholds:
        if value <= limit:
            return score
    return thresholds[-1][1]


def compute_startup_score(framework_df, extraction_data, analysis_data):
    total_score = 0
    total_weight = 0
    details = []

    analysis_map = {
        "Problem": 1,
        "Solution": 2,
        "Market": 3,
        "Product / Tech Stack": 4,
        "Business Model": 5,
        "Competition": 6,
        "GTM Strategy": 7,
        "Team": 8,
        "Traction / Results": 9,
        "Financials & Investment Ask": 10
    }

    analysis_lookup = {q["question_number"]: q["analysis"] for q in analysis_data["questions"]}

    for _, row in framework_df.iterrows():
        element = str(row["Pitch Element"])
        sub = str(row["Sub-Criteria"])
        weight = row["Weight (0–1)"]
        
        if pd.isna(element) or element == "" or element == "nan":
            continue
        
        score = None

        if row["Type"] == "Qualitative":
            for key, qnum in analysis_map.items():
                if key.lower() in element.lower():
                    text = analysis_lookup.get(qnum, "")
                    score = text_to_score(text)
                    break

        elif row["Type"] == "Quantitative":
            if "Market" in element:
                market_val = extraction_data["market_analysis"]["market_metrics"][0]["value"]
                match = re.search(r"\$([\d\.]+)", market_val)
                if match:
                    val = float(match.group(1))

                    score = gated_numeric_score(text, val, [(1,1), (5,3), (10,4), (100,5)])
            elif "Traction" in element:
                mrr_val = extraction_data["business_performance_and_traction"]["current_financials"][0]["value"]
                match = re.search(r"\$([\d,]+)", mrr_val.replace(",",""))
                if match:
                    val = float(match.group(1))
                    score = numeric_to_score(val, [(10000,1),(50000,3),(100000,4),(1000000,5)])

        if score is None:
            score = 3

        weighted_score = score * weight * 20
        total_score += weighted_score
        total_weight += weight

        details.append({
            "Element": element,
            "Sub-Criteria": sub,
            "Weight": weight,
            "Score": score,
            "Weighted": weighted_score
        })

    overall_score = total_score / total_weight if total_weight > 0 else 0

    return overall_score, pd.DataFrame(details)



overall, details_df = compute_startup_score(scoring_fw, extraction_data, analysis_data)


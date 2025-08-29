 
# Startup Scoring Baseline Model

 The script is designed to provide an initial, automated assessment of startup presentations based on extracted data and a scoring framework.

## Purpose
- **Automated Scoring:** The script reads extracted pitch data and a scoring framework, then computes an overall score for a startup pitch.
- **Baseline Model:** This is a starting point for automated scoring. The logic and methods can be improved and extended as needed.

## How It Works
- **Inputs:**
	- `scoring_fw_clean.csv`: The scoring framework with criteria, weights, and types.
	- `final_extraction_result.json`: Extracted data from the pitch.
	- `VC_Pitch_Analysis_*.json`: Analysis results for qualitative scoring.
- **Processing:**
	- Qualitative elements are scored using sentiment analysis and keyword matching.
	- Quantitative elements are scored based on extracted values and predefined thresholds.
	- Scores are weighted and aggregated to produce an overall score.



"""
Run this first to discover the real field names in v2.
python check_fields.py
"""
import requests, json
# url = "https://www.fema.gov/api/open/v2/PublicAssistanceFundedProjectsDetails?$top=1&$format=json"
url = "https://www.fema.gov/api/open/v2/PublicAssistanceFundedProjectsDetails"
resp = requests.get(url, params={"$top": 1, "$format": "json"}, timeout=30)
print("Status:", resp.status_code)
# print(resp.json())
if resp.ok:
    data = resp.json()
    for key, val in data.items():
        if isinstance(val, list) and val:
            print("\n=== FIELDS IN v2 PublicAssistanceFundedProjectsDetails ===")
            for field in sorted(val[0].keys()):
                print(f"  {field}: {val[0][field]}")
            break
    else:
        print("Top-level keys:", list(data.keys()))
else:
    print("Response body:", resp.text[:500])
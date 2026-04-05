"""
FEMA Open Data API Client
Field names confirmed from live v2 API response.
"""

import time
import logging
from typing import Optional

import requests
import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://www.fema.gov/api/open"

ENDPOINTS = {
    "declarations":       f"{BASE_URL}/v2/DisasterDeclarationsSummaries",
    "public_assistance":  f"{BASE_URL}/v2/PublicAssistanceFundedProjectsDetails",
    "disaster_summaries": f"{BASE_URL}/v1/FemaWebDisasterSummaries",
}

# Field names confirmed from live v2 API (check_fields.py output):
#   spec name          → actual v2 field
#   state              → stateAbbreviation
#   obligatedAmount    → federalShareObligated
#   projectCategory    → damageCategoryCode
#   project_id         → pwNumber
COLUMN_SELECTIONS = {
    "declarations": [
        "disasterNumber", "state", "incidentType",
        "declarationDate", "incidentBeginDate", "incidentEndDate", "declarationType",
    ],
    "public_assistance": [
        "disasterNumber", "stateAbbreviation", "incidentType",
        "federalShareObligated", "damageCategoryCode",
        "pwNumber", "countyCode",
    ],
    "disaster_summaries": [
        "disasterNumber", "totalObligatedAmountPa",
    ],
}

DEFAULT_PAGE_SIZE = 1000
REQUEST_TIMEOUT   = 60
RETRY_ATTEMPTS    = 3
RETRY_BACKOFF     = 2.0


class FEMAClientError(Exception):
    """Raised when the FEMA API returns an unrecoverable error."""


class FEMAClient:
    """Paginated FEMA Open Data REST API client."""

    def __init__(self, page_size=DEFAULT_PAGE_SIZE, timeout=REQUEST_TIMEOUT):
        self.page_size = page_size
        self.timeout   = timeout
        self.session   = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def fetch(self, endpoint_key, filters=None, max_records=None):
        if endpoint_key not in ENDPOINTS:
            raise ValueError(f"Unknown endpoint '{endpoint_key}'. Choose from: {list(ENDPOINTS)}")

        url     = ENDPOINTS[endpoint_key]
        columns = COLUMN_SELECTIONS[endpoint_key]
        records = []
        skip    = 0

        logger.info("Fetching '%s' from FEMA API ...", endpoint_key)

        while True:
            params = self._build_params(columns, filters, skip)
            batch  = self._get_with_retry(url, params, endpoint_key)

            if not batch:
                break

            records.extend(batch)
            logger.debug("  ... %d records fetched (total: %d)", len(batch), len(records))

            if len(batch) < self.page_size:
                break

            if max_records and len(records) >= max_records:
                records = records[:max_records]
                break

            skip += self.page_size

        logger.info("Finished '%s': %d total records", endpoint_key, len(records))
        return pd.DataFrame(records, columns=columns) if records else pd.DataFrame(columns=columns)

    def fetch_all(self, start_year=None, end_year=None):
        date_filter = {}
        if start_year:
            date_filter["declarationDate"] = f">={start_year}-01-01"

        datasets = {}
        for key in ENDPOINTS:
            f = date_filter if key == "declarations" else {}
            datasets[key] = self.fetch(key, filters=f)

        return datasets

    def _build_params(self, columns, filters, skip):
        params = {
            "$select":      ",".join(columns),
            "$top":         self.page_size,
            "$skip":        skip,
            "$format":      "json",
            "$inlinecount": "allpages",
        }
        if filters:
            parts = []
            op_map = {">=": "ge", "<=": "le", ">": "gt", "<": "lt", "!=": "ne"}
            for k, v in filters.items():
                matched = False
                for sym, odata in op_map.items():
                    if str(v).startswith(sym):
                        parts.append(f"{k} {odata} '{v[len(sym):]}'")
                        matched = True
                        break
                if not matched:
                    parts.append(f"{k} eq '{v}'")
            params["$filter"] = " and ".join(parts)
        return params

    def _get_with_retry(self, url, params, label):
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json()

                for key in payload:
                    if isinstance(payload[key], list):
                        return payload[key]

                logger.warning("Unexpected payload for '%s': keys=%s", label, list(payload))
                return []

            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    raise FEMAClientError(f"Endpoint not found: {url}") from exc
                logger.warning("HTTP error on attempt %d/%d: %s", attempt, RETRY_ATTEMPTS, exc)

            except requests.exceptions.RequestException as exc:
                logger.warning("Request error on attempt %d/%d: %s", attempt, RETRY_ATTEMPTS, exc)

            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF * attempt)

        raise FEMAClientError(f"Failed to fetch '{label}' after {RETRY_ATTEMPTS} attempts.")
    
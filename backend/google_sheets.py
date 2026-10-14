"""
google_sheets.py — Google Sheets persistence layer for the PCOS AI Agent.

Replaces ephemeral CSV writes with Google Sheets API calls so data
survives Vercel's serverless cold starts.

Falls back to local CSV when credentials are not configured (local dev).
"""

import os
import json
import csv
from datetime import datetime
from typing import Dict, Any, Optional

# Lazy-loaded gspread client
_sheets_client = None
_spreadsheet = None


def _get_spreadsheet():
    """Lazily initialize and return the Google Spreadsheet object."""
    global _sheets_client, _spreadsheet

    if _spreadsheet is not None:
        return _spreadsheet

    creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    sheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()

    if not creds_json or not sheet_id:
        return None

    try:
        import gspread
        creds_dict = json.loads(creds_json)
        _sheets_client = gspread.service_account_from_dict(creds_dict)
        _spreadsheet = _sheets_client.open_by_key(sheet_id)
        print("✅ Google Sheets connected successfully")
        return _spreadsheet
    except Exception as e:
        print(f"⚠️  Google Sheets init error: {e}")
        return None


def _ensure_headers(worksheet, headers: list):
    """Add header row if the sheet is empty."""
    try:
        existing = worksheet.row_values(1)
        if not existing:
            worksheet.append_row(headers, value_input_option="RAW")
    except Exception:
        # Sheet might be completely empty — append headers
        worksheet.append_row(headers, value_input_option="RAW")


def is_sheets_available() -> bool:
    """Check whether Google Sheets credentials are configured."""
    creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    sheet_id = os.getenv("GOOGLE_SHEET_ID", "").strip()
    return bool(creds_json and sheet_id)


def append_collected_data(row_dict: Dict[str, Any]) -> bool:
    """
    Append a collected-data row to the 'collected_data' sheet.

    Returns True if written to Sheets, False if fell back to CSV.
    """
    spreadsheet = _get_spreadsheet()

    if spreadsheet:
        try:
            worksheet = spreadsheet.worksheet("collected_data")
            headers = list(row_dict.keys())
            _ensure_headers(worksheet, headers)
            values = [str(v) for v in row_dict.values()]
            worksheet.append_row(values, value_input_option="RAW")
            print("📊 Data saved to Google Sheets (collected_data)")
            return True
        except Exception as e:
            print(f"⚠️  Sheets write error (collected_data): {e}")
            # Fall through to CSV

    # Fallback: local CSV
    _write_csv("collected_pcos_data.csv", row_dict)
    return False


def append_feedback(row_dict: Dict[str, Any]) -> bool:
    """
    Append a feedback row to the 'feedback' sheet.

    Returns True if written to Sheets, False if fell back to CSV.
    """
    spreadsheet = _get_spreadsheet()

    if spreadsheet:
        try:
            worksheet = spreadsheet.worksheet("feedback")
            headers = list(row_dict.keys())
            _ensure_headers(worksheet, headers)
            values = [str(v) for v in row_dict.values()]
            worksheet.append_row(values, value_input_option="RAW")
            print("📊 Feedback saved to Google Sheets")
            return True
        except Exception as e:
            print(f"⚠️  Sheets write error (feedback): {e}")
            # Fall through to CSV

    # Fallback: local CSV
    _write_csv("feedback.csv", row_dict)
    return False


def _write_csv(path: str, row_dict: Dict[str, Any]):
    """Fallback: append row to a local CSV file."""
    file_exists = os.path.exists(path)
    try:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)
        print(f"📊 Data saved to local CSV: {path}")
    except Exception as e:
        print(f"⚠️  CSV write error ({path}): {e}")

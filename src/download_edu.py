# OpenAlex bulk PDF downloader (education/admissions/counseling)
# Saves ONLY PDFs to /home/anuragd/labshare/corpus/education

import time
import requests
from pathlib import Path
from typing import Optional

OUT = Path("/home/anuragd/labshare/corpus/education")
OUT.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Python/requests"}
BASE = "https://api.openalex.org/works"

# ---- Tune these -------------------------------------------------------------
SEARCH = "college admission counseling OR admissions advising OR school counseling"
DATE_FROM = "2010-01-01"  # widen/narrow as you like
TYPES = "article|report|book|proceedings-article"  # allowed OpenAlex types
PER_PAGE = 200
MAX_DOWNLOADS = 1000        # set None for no cap
SLEEP_SEC = 0.4             # be polite to the API
# -----------------------------------------------------------------------------

def pick_pdf_url(work: dict) -> Optional[str]:
    """Prefer a direct PDF URL if available; otherwise return None."""
    # Best location (often includes a real PDF)
    loc = work.get("best_oa_location") or {}
    pdf = loc.get("pdf_url")
    if not pdf:
        # Some records have PDF under primary_location
        loc = work.get("primary_location") or {}
        pdf = loc.get("pdf_url")
    if not pdf:
        # Fallback: sometimes open_access.oa_url is a landing page (not PDF)
        pdf = (work.get("open_access") or {}).get("oa_url")
    return pdf

def is_pdf_response(resp: requests.Response, url: str) -> bool:
    ctype = (resp.headers.get("content-type") or "").lower()
    return ("pdf" in ctype) or url.lower().endswith(".pdf")

def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in (s or ""))

def main():
    cursor = "*"
    downloaded = 0

    while True:
        url = (
            f"{BASE}?search={requests.utils.quote(SEARCH)}"
            f"&filter=is_oa:true,type:{TYPES},from_publication_date:{DATE_FROM}"
            f"&per_page={PER_PAGE}&cursor={cursor}"
        )
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()

        results = data.get("results", [])
        if not results:
            break

        for w in results:
            if MAX_DOWNLOADS is not None and downloaded >= MAX_DOWNLOADS:
                print(f"Reached MAX_DOWNLOADS={MAX_DOWNLOADS}")
                return

            pdf_url = pick_pdf_url(w)
            if not pdf_url:
                continue

            work_id = w.get("id", "")  # e.g., "https://openalex.org/W123..."
            stem = safe_name(work_id.split("/")[-1] or f"oa_{downloaded}")
            fn = OUT / f"{stem}.pdf"
            if fn.exists():
                continue

            try:
                with requests.get(pdf_url, headers=HEADERS, timeout=60, stream=True, allow_redirects=True) as pr:
                    pr.raise_for_status()
                    if not is_pdf_response(pr, pdf_url):
                        continue  # skip non-PDF targets
                    with open(fn, "wb") as f:
                        for chunk in pr.iter_content(1024 * 64):
                            if chunk:
                                f.write(chunk)
                downloaded += 1
                # Optional: print a little trace
                # print(f"Saved: {fn}")
            except Exception:
                # skip errors and keep going
                pass

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(SLEEP_SEC)

    print("Downloaded PDFs:", downloaded)

if __name__ == "__main__":
    main()

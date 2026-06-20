"""
Provider-agnostic ATS adapters (Phase 9).

Each adapter makes REAL HTTP calls to the vendor API and normalises results to:
  job:       {external_id, title, description, location, status}
  candidate: {external_id, full_name, email, phone}

Credentials are required to actually sync; without valid keys, test() and the list
methods raise/return errors that the API surfaces clearly (no fabricated data).
"""
import base64
from typing import List, Dict, Any
import httpx

SUPPORTED_PROVIDERS = ("greenhouse", "lever", "workable")
TIMEOUT = 20


class AtsError(Exception):
    pass


class BaseProvider:
    provider = "base"

    def __init__(self, api_key: str = "", subdomain: str = "", base_url: str = ""):
        self.api_key = api_key or ""
        self.subdomain = subdomain or ""
        self.base_url = base_url or ""

    def _client(self, headers) -> httpx.Client:
        return httpx.Client(timeout=TIMEOUT, headers=headers)

    def test(self) -> Dict[str, Any]:
        try:
            jobs = self.list_jobs(limit=1)
            return {"ok": True, "message": f"Connected. Sample jobs available: {len(jobs)}."}
        except AtsError as e:
            return {"ok": False, "message": str(e)}
        except Exception as e:
            return {"ok": False, "message": f"Connection failed: {e}"}

    def list_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def list_candidates(self, limit: int = 100) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def push_result(self, external_candidate_id: str, summary: str) -> Dict[str, Any]:
        return {"ok": False, "message": "Pushing results is not supported for this provider yet."}


class GreenhouseProvider(BaseProvider):
    """Greenhouse Harvest API. Auth: HTTP Basic with API key as username."""
    provider = "greenhouse"

    def _headers(self):
        if not self.api_key:
            raise AtsError("Greenhouse API key is missing.")
        token = base64.b64encode(f"{self.api_key}:".encode()).decode()
        return {"Authorization": f"Basic {token}", "Content-Type": "application/json"}

    def _base(self):
        return self.base_url or "https://harvest.greenhouse.io/v1"

    def list_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._client(self._headers()) as c:
            r = c.get(f"{self._base()}/jobs", params={"per_page": min(limit, 100)})
        if r.status_code == 401:
            raise AtsError("Greenhouse rejected the API key (401).")
        if r.status_code >= 400:
            raise AtsError(f"Greenhouse error {r.status_code}: {r.text[:200]}")
        out = []
        for j in r.json():
            offices = j.get("offices") or []
            out.append({
                "external_id": str(j.get("id")),
                "title": j.get("name") or "Untitled role",
                "description": (j.get("notes") or ""),
                "location": (offices[0].get("name") if offices else None),
                "status": j.get("status") or "open",
            })
        return out

    def list_candidates(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._client(self._headers()) as c:
            r = c.get(f"{self._base()}/candidates", params={"per_page": min(limit, 100)})
        if r.status_code == 401:
            raise AtsError("Greenhouse rejected the API key (401).")
        if r.status_code >= 400:
            raise AtsError(f"Greenhouse error {r.status_code}: {r.text[:200]}")
        out = []
        for cand in r.json():
            emails = cand.get("email_addresses") or []
            phones = cand.get("phone_numbers") or []
            name = (f"{cand.get('first_name','')} {cand.get('last_name','')}").strip()
            out.append({
                "external_id": str(cand.get("id")),
                "full_name": name or "Unknown",
                "email": emails[0].get("value") if emails else None,
                "phone": phones[0].get("value") if phones else None,
            })
        return out


class LeverProvider(BaseProvider):
    """Lever API. Auth: HTTP Basic with API key as username."""
    provider = "lever"

    def _headers(self):
        if not self.api_key:
            raise AtsError("Lever API key is missing.")
        token = base64.b64encode(f"{self.api_key}:".encode()).decode()
        return {"Authorization": f"Basic {token}", "Content-Type": "application/json"}

    def _base(self):
        return self.base_url or "https://api.lever.co/v1"

    def list_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._client(self._headers()) as c:
            r = c.get(f"{self._base()}/postings", params={"limit": min(limit, 100)})
        if r.status_code == 401:
            raise AtsError("Lever rejected the API key (401).")
        if r.status_code >= 400:
            raise AtsError(f"Lever error {r.status_code}: {r.text[:200]}")
        out = []
        for j in (r.json().get("data") or []):
            cats = j.get("categories") or {}
            out.append({
                "external_id": str(j.get("id")),
                "title": j.get("text") or "Untitled role",
                "description": (j.get("descriptionPlain") or j.get("description") or ""),
                "location": cats.get("location"),
                "status": j.get("state") or "open",
            })
        return out

    def list_candidates(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._client(self._headers()) as c:
            r = c.get(f"{self._base()}/opportunities", params={"limit": min(limit, 100)})
        if r.status_code == 401:
            raise AtsError("Lever rejected the API key (401).")
        if r.status_code >= 400:
            raise AtsError(f"Lever error {r.status_code}: {r.text[:200]}")
        out = []
        for cand in (r.json().get("data") or []):
            emails = cand.get("emails") or []
            phones = cand.get("phones") or []
            out.append({
                "external_id": str(cand.get("id")),
                "full_name": cand.get("name") or "Unknown",
                "email": emails[0] if emails else None,
                "phone": (phones[0].get("value") if phones and isinstance(phones[0], dict) else (phones[0] if phones else None)),
            })
        return out


class WorkableProvider(BaseProvider):
    """Workable SPI v3. Auth: Bearer token. Requires subdomain."""
    provider = "workable"

    def _headers(self):
        if not self.api_key:
            raise AtsError("Workable API token is missing.")
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _base(self):
        if self.base_url:
            return self.base_url
        if not self.subdomain:
            raise AtsError("Workable requires a subdomain.")
        return f"https://{self.subdomain}.workable.com/spi/v3"

    def list_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._client(self._headers()) as c:
            r = c.get(f"{self._base()}/jobs", params={"limit": min(limit, 100)})
        if r.status_code == 401:
            raise AtsError("Workable rejected the token (401).")
        if r.status_code >= 400:
            raise AtsError(f"Workable error {r.status_code}: {r.text[:200]}")
        out = []
        for j in (r.json().get("jobs") or []):
            out.append({
                "external_id": str(j.get("shortcode") or j.get("id")),
                "title": j.get("title") or "Untitled role",
                "description": (j.get("description") or ""),
                "location": (j.get("location") or {}).get("location_str") if isinstance(j.get("location"), dict) else None,
                "status": j.get("state") or "open",
            })
        return out

    def list_candidates(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._client(self._headers()) as c:
            r = c.get(f"{self._base()}/candidates", params={"limit": min(limit, 100)})
        if r.status_code == 401:
            raise AtsError("Workable rejected the token (401).")
        if r.status_code >= 400:
            raise AtsError(f"Workable error {r.status_code}: {r.text[:200]}")
        out = []
        for cand in (r.json().get("candidates") or []):
            out.append({
                "external_id": str(cand.get("id")),
                "full_name": cand.get("name") or "Unknown",
                "email": cand.get("email"),
                "phone": cand.get("phone"),
            })
        return out


_PROVIDERS = {
    "greenhouse": GreenhouseProvider,
    "lever": LeverProvider,
    "workable": WorkableProvider,
}


def get_provider(provider: str, api_key: str = "", subdomain: str = "", base_url: str = "") -> BaseProvider:
    cls = _PROVIDERS.get((provider or "").lower())
    if not cls:
        raise AtsError(f"Unsupported ATS provider '{provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}.")
    return cls(api_key=api_key, subdomain=subdomain, base_url=base_url)

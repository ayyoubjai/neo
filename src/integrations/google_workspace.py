from __future__ import annotations

import base64
import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common.config import Settings, load_settings


class GoogleAuthError(Exception):
    pass


class GoogleWorkspaceManager:
    BUNDLE_SCOPES: Dict[str, Tuple[str, ...]] = {
        "gmail_send": ("https://www.googleapis.com/auth/gmail.send",),
        "gmail_compose": ("https://www.googleapis.com/auth/gmail.compose",),
        "gmail_readonly": ("https://www.googleapis.com/auth/gmail.readonly",),
        "calendar_readonly": ("https://www.googleapis.com/auth/calendar.readonly",),
        "calendar_events": ("https://www.googleapis.com/auth/calendar.events",),
        "docs_readonly": ("https://www.googleapis.com/auth/documents.readonly",),
        "docs_edit": ("https://www.googleapis.com/auth/documents",),
    }

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or load_settings()
        self._cfg = getattr(self._settings, "google", {}) or {}
        default_account = str(self._cfg.get("default_account_name", "default")).strip()
        self._default_account_name = default_account or "default"
        service = str(self._cfg.get("keyring_service", "AGI.GoogleWorkspace")).strip()
        self._keyring_service = service or "AGI.GoogleWorkspace"

    def status(self, account_name: Optional[str] = None) -> Dict[str, Any]:
        resolved_account = self._resolve_account_name(account_name)
        bundles = []
        for bundle, scopes in self.BUNDLE_SCOPES.items():
            bundles.append(
                {
                    "bundle": bundle,
                    "scopes": list(scopes),
                    "authorized": bool(self._load_token_json(resolved_account, bundle)) if self._cfg.get("enabled", True) else False,
                }
            )
        client_path = self._client_secrets_path()
        return {
            "enabled": self._cfg.get("enabled", True),
            "oauth_client_source": self._cfg.get("oauth_client_source", "own"),
            "configured": os.path.exists(client_path),
            "client_secrets_path": client_path,
            "account_name": resolved_account,
            "bundles": bundles,
        }

    def authorize(
        self,
        bundle: str,
        account_name: Optional[str] = None,
        force_reconsent: bool = False,
    ) -> Dict[str, Any]:
        resolved_account = self._resolve_account_name(account_name)
        scopes = self._scopes_for_bundle(bundle)
        self._require_enabled()
        creds = self._authorize_interactive(resolved_account, bundle, scopes)
        return {
            "authorized": True,
            "account_name": resolved_account,
            "bundle": bundle,
            "scopes": list(scopes),
            "has_refresh_token": bool(getattr(creds, "refresh_token", None)),
        }

    def ensure_authorized(
        self,
        bundle: str,
        account_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_account = self._resolve_account_name(account_name)
        scopes = self._scopes_for_bundle(bundle)
        try:
            creds = self._load_credentials(resolved_account, bundle)
            interactive = False
        except GoogleAuthError:
            creds = self._authorize_interactive(resolved_account, bundle, scopes)
            interactive = True
        return {
            "authorized": True,
            "account_name": resolved_account,
            "bundle": bundle,
            "scopes": list(scopes),
            "interactive": interactive,
            "has_refresh_token": bool(getattr(creds, "refresh_token", None)),
        }

    def disconnect(
        self,
        account_name: Optional[str] = None,
        bundle: Optional[str] = None,
        revoke: bool = False,
    ) -> Dict[str, Any]:
        resolved_account = self._resolve_account_name(account_name)
        bundles = [bundle] if bundle else list(self.BUNDLE_SCOPES.keys())
        removed: List[str] = []
        revoked: List[str] = []

        for item in bundles:
            if not item:
                continue
            self._scopes_for_bundle(item)
            token_json = self._load_token_json(resolved_account, item)
            if not token_json:
                continue
            if revoke and self._revoke_token(token_json):
                revoked.append(item)
            self._delete_token_json(resolved_account, item)
            removed.append(item)

        return {
            "account_name": resolved_account,
            "removed_bundles": removed,
            "revoked_bundles": revoked,
        }

    def gmail_list_messages(
        self,
        account_name: Optional[str] = None,
        query: str = "",
        max_results: int = 10,
        label_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        service = self._build_service(account_name, "gmail_readonly", "gmail", "v1")
        params: Dict[str, Any] = {
            "userId": "me",
            "maxResults": max(1, min(int(max_results), 25)),
        }
        if query.strip():
            params["q"] = query.strip()
        labels = self._normalize_string_list(label_ids)
        if labels:
            params["labelIds"] = labels
        response = service.users().messages().list(**params).execute()
        messages = []
        for item in response.get("messages", []):
            message_id = item.get("id")
            if not isinstance(message_id, str) or not message_id:
                continue
            meta = (
                service.users()
                .messages()
                .get(
                    userId="me",
                    id=message_id,
                    format="metadata",
                    metadataHeaders=["Subject", "From", "To", "Cc", "Date"],
                )
                .execute()
            )
            messages.append(self._gmail_message_summary(meta))
        return {
            "messages": messages,
            "result_size_estimate": response.get("resultSizeEstimate", len(messages)),
            "next_page_token": response.get("nextPageToken"),
        }

    def gmail_get_message(self, message_id: str, account_name: Optional[str] = None) -> Dict[str, Any]:
        if not isinstance(message_id, str) or not message_id.strip():
            raise GoogleAuthError("message_id is required")
        service = self._build_service(account_name, "gmail_readonly", "gmail", "v1")
        payload = service.users().messages().get(userId="me", id=message_id.strip(), format="full").execute()
        text_body, html_body = self._extract_gmail_body(payload.get("payload", {}))
        result = self._gmail_message_summary(payload)
        result["body_text"] = text_body
        result["body_html"] = html_body
        result["payload_headers"] = self._header_map(payload.get("payload", {}).get("headers", []))
        return result

    def gmail_create_draft(
        self,
        subject: str,
        to: Sequence[str],
        body_text: str = "",
        account_name: Optional[str] = None,
        cc: Optional[Sequence[str]] = None,
        bcc: Optional[Sequence[str]] = None,
        body_html: str = "",
        reply_to_message_id: str = "",
    ) -> Dict[str, Any]:
        service = self._build_service(account_name, "gmail_compose", "gmail", "v1")
        raw = self._build_gmail_raw_message(
            subject=subject,
            to=to,
            cc=cc,
            bcc=bcc,
            body_text=body_text,
            body_html=body_html,
            reply_to_message_id=reply_to_message_id,
        )
        draft = service.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
        message = draft.get("message", {})
        return {
            "draft_id": draft.get("id"),
            "message_id": message.get("id"),
            "thread_id": message.get("threadId"),
        }

    def gmail_send_message(
        self,
        subject: str,
        to: Sequence[str],
        body_text: str = "",
        account_name: Optional[str] = None,
        cc: Optional[Sequence[str]] = None,
        bcc: Optional[Sequence[str]] = None,
        body_html: str = "",
        reply_to_message_id: str = "",
    ) -> Dict[str, Any]:
        service = self._build_service(account_name, "gmail_send", "gmail", "v1")
        raw = self._build_gmail_raw_message(
            subject=subject,
            to=to,
            cc=cc,
            bcc=bcc,
            body_text=body_text,
            body_html=body_html,
            reply_to_message_id=reply_to_message_id,
        )
        message = service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return {
            "message_id": message.get("id"),
            "thread_id": message.get("threadId"),
            "label_ids": message.get("labelIds", []),
        }

    def calendar_list_events(
        self,
        account_name: Optional[str] = None,
        calendar_id: str = "primary",
        time_min: str = "",
        time_max: str = "",
        max_results: int = 10,
        query: str = "",
    ) -> Dict[str, Any]:
        service = self._build_service(account_name, "calendar_readonly", "calendar", "v3")
        params: Dict[str, Any] = {
            "calendarId": calendar_id or "primary",
            "singleEvents": True,
            "orderBy": "startTime",
            "maxResults": max(1, min(int(max_results), 50)),
        }
        if isinstance(time_min, str) and time_min.strip():
            params["timeMin"] = time_min.strip()
        if isinstance(time_max, str) and time_max.strip():
            params["timeMax"] = time_max.strip()
        if isinstance(query, str) and query.strip():
            params["q"] = query.strip()
        response = service.events().list(**params).execute()
        return {
            "calendar_id": calendar_id or "primary",
            "events": [self._calendar_event_summary(item) for item in response.get("items", [])],
            "next_page_token": response.get("nextPageToken"),
        }

    def calendar_create_event(
        self,
        summary: str,
        start: Any,
        end: Any,
        account_name: Optional[str] = None,
        calendar_id: str = "primary",
        description: str = "",
        location: str = "",
        attendees: Optional[Sequence[str]] = None,
        send_updates: str = "none",
    ) -> Dict[str, Any]:
        if not isinstance(summary, str) or not summary.strip():
            raise GoogleAuthError("summary is required")
        service = self._build_service(account_name, "calendar_events", "calendar", "v3")
        body: Dict[str, Any] = {
            "summary": summary.strip(),
            "start": self._normalize_calendar_time(start),
            "end": self._normalize_calendar_time(end),
        }
        if isinstance(description, str) and description:
            body["description"] = description
        if isinstance(location, str) and location:
            body["location"] = location
        attendee_emails = self._normalize_string_list(attendees)
        if attendee_emails:
            body["attendees"] = [{"email": email} for email in attendee_emails]
        updates = str(send_updates or "none").strip() or "none"
        event = service.events().insert(
            calendarId=calendar_id or "primary",
            body=body,
            sendUpdates=updates,
        ).execute()
        return self._calendar_event_summary(event)

    def docs_get_document(self, document_id: str, account_name: Optional[str] = None) -> Dict[str, Any]:
        if not isinstance(document_id, str) or not document_id.strip():
            raise GoogleAuthError("document_id is required")
        service = self._build_service(account_name, "docs_readonly", "docs", "v1")
        doc = service.documents().get(documentId=document_id.strip()).execute()
        return {
            "document_id": doc.get("documentId"),
            "title": doc.get("title"),
            "revision_id": doc.get("revisionId"),
            "text": self._extract_docs_text(doc.get("body", {}).get("content", [])),
        }

    def docs_create_document(
        self,
        title: str,
        account_name: Optional[str] = None,
        content: str = "",
    ) -> Dict[str, Any]:
        if not isinstance(title, str) or not title.strip():
            raise GoogleAuthError("title is required")
        service = self._build_service(account_name, "docs_edit", "docs", "v1")
        doc = service.documents().create(body={"title": title.strip()}).execute()
        document_id = doc.get("documentId")
        if isinstance(content, str) and content and isinstance(document_id, str) and document_id:
            self._docs_insert_text(service, document_id, 1, content)
            doc = service.documents().get(documentId=document_id).execute()
        return {
            "document_id": doc.get("documentId"),
            "title": doc.get("title"),
            "revision_id": doc.get("revisionId"),
            "text": self._extract_docs_text(doc.get("body", {}).get("content", [])),
        }

    def docs_append_text(
        self,
        document_id: str,
        text: str,
        account_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(document_id, str) or not document_id.strip():
            raise GoogleAuthError("document_id is required")
        if not isinstance(text, str) or not text:
            raise GoogleAuthError("text is required")
        service = self._build_service(account_name, "docs_edit", "docs", "v1")
        doc = service.documents().get(documentId=document_id.strip()).execute()
        insert_index = max(1, self._docs_body_end_index(doc) - 1)
        self._docs_insert_text(service, document_id.strip(), insert_index, text)
        updated = service.documents().get(documentId=document_id.strip()).execute()
        return {
            "document_id": updated.get("documentId"),
            "title": updated.get("title"),
            "revision_id": updated.get("revisionId"),
            "text": self._extract_docs_text(updated.get("body", {}).get("content", [])),
        }

    def _resolve_account_name(self, account_name: Optional[str]) -> str:
        text = str(account_name or "").strip()
        return text or self._default_account_name

    def _require_enabled(self) -> None:
        if not self._cfg.get("enabled", True):
            raise GoogleAuthError("Google integration is disabled. Enable it explicitly in setup or local settings first.")

    def _client_secrets_path(self) -> str:
        configured = str(self._cfg.get("client_secrets_path", "")).strip()
        if configured:
            return configured
        return os.path.join(self._settings.workspace_root, "config", "google_client_secret.json")

    def _token_key(self, account_name: str, bundle: str) -> str:
        if self._cfg.get("oauth_client_source") in {"neo", "own"}:
            try:
                with open(self._client_secrets_path(), encoding="utf-8") as handle:
                    client_id = json.load(handle)["installed"]["client_id"]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise GoogleAuthError("Unable to identify the configured Desktop OAuth client.") from exc
            namespace = hashlib.sha256(client_id.encode()).hexdigest()[:24]
            return f"{namespace}:{account_name}:{bundle}"
        return f"{account_name}:{bundle}"

    def _scopes_for_bundle(self, bundle: str) -> Tuple[str, ...]:
        resolved = str(bundle or "").strip()
        scopes = self.BUNDLE_SCOPES.get(resolved)
        if not scopes:
            raise GoogleAuthError(f"Unsupported Google bundle: {bundle}")
        return scopes

    def _google_auth_modules(self):
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as exc:
            raise GoogleAuthError(
                "Google client libraries are not installed. Install the updated requirements first."
            ) from exc
        return Request, Credentials, InstalledAppFlow

    def _load_keyring(self):
        try:
            import keyring
        except ImportError as exc:
            raise GoogleAuthError("The 'keyring' package is required for Google token storage.") from exc
        return keyring

    def _load_token_json(self, account_name: str, bundle: str) -> Optional[str]:
        keyring = self._load_keyring()
        try:
            return keyring.get_password(self._keyring_service, self._token_key(account_name, bundle))
        except Exception as exc:
            raise GoogleAuthError(f"Unable to read Google credentials from the system keyring: {exc}") from exc

    def _store_token_json(self, account_name: str, bundle: str, token_json: str) -> None:
        keyring = self._load_keyring()
        try:
            keyring.set_password(self._keyring_service, self._token_key(account_name, bundle), token_json)
        except Exception as exc:
            raise GoogleAuthError(f"Unable to write Google credentials to the system keyring: {exc}") from exc

    def _delete_token_json(self, account_name: str, bundle: str) -> None:
        keyring = self._load_keyring()
        try:
            keyring.delete_password(self._keyring_service, self._token_key(account_name, bundle))
        except Exception:
            return

    def _load_credentials(self, account_name: str, bundle: str):
        self._require_enabled()
        token_json = self._load_token_json(account_name, bundle)
        if not token_json:
            raise GoogleAuthError(
                f"Google bundle '{bundle}' is not authorized for account '{account_name}'. Run google.authorize first."
            )
        Request, Credentials, _ = self._google_auth_modules()
        try:
            info = json.loads(token_json)
            creds = Credentials.from_authorized_user_info(info, scopes=list(self._scopes_for_bundle(bundle)))
        except Exception as exc:
            raise GoogleAuthError(f"Stored Google credentials are invalid: {exc}") from exc

        if creds.valid:
            return creds
        if creds.expired and getattr(creds, "refresh_token", None):
            try:
                creds.refresh(Request())
            except Exception as exc:
                raise GoogleAuthError(f"Unable to refresh Google credentials: {exc}") from exc
            self._store_token_json(account_name, bundle, creds.to_json())
            return creds
        raise GoogleAuthError(
            f"Google bundle '{bundle}' does not have a usable refresh token. Re-run google.authorize."
        )

    def _authorize_interactive(self, account_name: str, bundle: str, scopes: Sequence[str]):
        self._require_enabled()
        client_path = self._client_secrets_path()
        if not os.path.exists(client_path):
            raise GoogleAuthError(
                f"Google OAuth client secrets file not found: {client_path}. "
                "Download an installed-app OAuth client from Google Cloud Console first."
            )
        _, _, InstalledAppFlow = self._google_auth_modules()
        with open(client_path, encoding="utf-8") as handle:
            client = json.load(handle).get("installed", {})
        if (client.get("auth_uri") != "https://accounts.google.com/o/oauth2/auth"
                or client.get("token_uri") != "https://oauth2.googleapis.com/token"):
            raise GoogleAuthError("Use a Desktop OAuth client with Google's official OAuth endpoints.")
        flow = InstalledAppFlow.from_client_secrets_file(client_path, scopes=list(scopes), autogenerate_code_verifier=True)
        host = str(self._cfg.get("oauth_bind_host", "127.0.0.1")).strip() or "127.0.0.1"
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise GoogleAuthError("Google OAuth callback must bind to a loopback address.")
        port = int(self._cfg.get("oauth_port", 0))
        open_browser = bool(self._cfg.get("open_browser", True))
        try:
            creds = flow.run_local_server(
                host=host,
                port=port,
                open_browser=open_browser,
                authorization_prompt_message="Open this URL in your browser to authorize Google access: {url}",
                success_message="Google authorization complete. You can close this window.",
            )
        except OSError as exc:
            raise GoogleAuthError(f"Unable to start the local OAuth callback server: {exc}") from exc
        granted = getattr(creds, "granted_scopes", None)
        if (granted is not None and not set(scopes).issubset(set(granted))) or not creds.has_scopes(scopes):
            raise GoogleAuthError("Google did not grant the required permissions; existing credentials were preserved.")
        self._store_token_json(account_name, bundle, creds.to_json())
        return creds

    def _build_service(self, account_name: Optional[str], bundle: str, api_name: str, api_version: str):
        resolved_account = self._resolve_account_name(account_name)
        creds = self._load_credentials(resolved_account, bundle)
        try:
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise GoogleAuthError(
                "google-api-python-client is not installed. Install the updated requirements first."
            ) from exc
        return build(api_name, api_version, credentials=creds, cache_discovery=False)

    def _revoke_token(self, token_json: str) -> bool:
        try:
            payload = json.loads(token_json)
        except json.JSONDecodeError:
            return False
        token = payload.get("refresh_token") or payload.get("token")
        if not isinstance(token, str) or not token:
            return False
        data = urllib.parse.urlencode({"token": token}).encode("utf-8")
        req = urllib.request.Request(
            "https://oauth2.googleapis.com/revoke",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(req, timeout=20):
                return True
        except (urllib.error.URLError, urllib.error.HTTPError):
            return False

    def _normalize_string_list(self, value: Optional[Sequence[str]]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",")]
        elif isinstance(value, (list, tuple)):
            items = [str(item).strip() for item in value]
        else:
            raise GoogleAuthError("Expected a string or list of strings")
        return [item for item in items if item]

    def _build_gmail_raw_message(
        self,
        *,
        subject: str,
        to: Sequence[str],
        cc: Optional[Sequence[str]],
        bcc: Optional[Sequence[str]],
        body_text: str,
        body_html: str,
        reply_to_message_id: str,
    ) -> str:
        to_list = self._normalize_string_list(to)
        if not to_list:
            raise GoogleAuthError("At least one recipient is required")
        cc_list = self._normalize_string_list(cc)
        bcc_list = self._normalize_string_list(bcc)

        text_body = body_text if isinstance(body_text, str) else str(body_text or "")
        html_body = body_html if isinstance(body_html, str) else str(body_html or "")

        if html_body and text_body:
            message = MIMEMultipart("alternative")
            message.attach(MIMEText(text_body, "plain", "utf-8"))
            message.attach(MIMEText(html_body, "html", "utf-8"))
        elif html_body:
            message = MIMEText(html_body, "html", "utf-8")
        else:
            message = MIMEText(text_body, "plain", "utf-8")

        message["To"] = ", ".join(to_list)
        if cc_list:
            message["Cc"] = ", ".join(cc_list)
        if bcc_list:
            message["Bcc"] = ", ".join(bcc_list)
        if isinstance(subject, str) and subject:
            message["Subject"] = subject
        if isinstance(reply_to_message_id, str) and reply_to_message_id.strip():
            message["In-Reply-To"] = reply_to_message_id.strip()
            message["References"] = reply_to_message_id.strip()

        return base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

    def _header_map(self, headers: Sequence[Dict[str, Any]]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for item in headers or []:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            value = item.get("value")
            if isinstance(name, str) and isinstance(value, str):
                result[name] = value
        return result

    def _gmail_message_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = self._header_map(payload.get("payload", {}).get("headers", []))
        return {
            "id": payload.get("id"),
            "thread_id": payload.get("threadId"),
            "label_ids": payload.get("labelIds", []),
            "snippet": payload.get("snippet"),
            "subject": headers.get("Subject", ""),
            "from": headers.get("From", ""),
            "to": headers.get("To", ""),
            "cc": headers.get("Cc", ""),
            "date": headers.get("Date", ""),
        }

    def _extract_gmail_body(self, payload: Dict[str, Any]) -> Tuple[str, str]:
        plain_parts: List[str] = []
        html_parts: List[str] = []

        def walk(part: Dict[str, Any]) -> None:
            mime_type = str(part.get("mimeType", ""))
            body = part.get("body", {}) or {}
            data = body.get("data")
            if isinstance(data, str) and data:
                decoded = self._decode_gmail_body_data(data)
                if mime_type == "text/html":
                    html_parts.append(decoded)
                else:
                    plain_parts.append(decoded)
            for child in part.get("parts", []) or []:
                if isinstance(child, dict):
                    walk(child)

        if isinstance(payload, dict):
            walk(payload)
        return "\n".join(part for part in plain_parts if part).strip(), "\n".join(
            part for part in html_parts if part
        ).strip()

    def _decode_gmail_body_data(self, data: str) -> str:
        padded = data + "=" * (-len(data) % 4)
        try:
            return base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8", "replace")
        except Exception:
            return ""

    def _normalize_calendar_time(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise GoogleAuthError("Calendar time values may not be empty")
            if len(text) == 10:
                return {"date": text}
            return {"dateTime": text}
        if isinstance(value, dict):
            if "date" in value and isinstance(value.get("date"), str) and value.get("date").strip():
                return {"date": value.get("date").strip()}
            if "dateTime" in value and isinstance(value.get("dateTime"), str) and value.get("dateTime").strip():
                result = {"dateTime": value.get("dateTime").strip()}
                if isinstance(value.get("timeZone"), str) and value.get("timeZone").strip():
                    result["timeZone"] = value.get("timeZone").strip()
                return result
        raise GoogleAuthError("Calendar start/end must be an ISO string or an object with date/dateTime")

    def _calendar_event_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": payload.get("id"),
            "status": payload.get("status"),
            "summary": payload.get("summary"),
            "description": payload.get("description"),
            "location": payload.get("location"),
            "html_link": payload.get("htmlLink"),
            "start": payload.get("start"),
            "end": payload.get("end"),
            "attendees": payload.get("attendees", []),
        }

    def _extract_docs_text(self, content: Sequence[Dict[str, Any]]) -> str:
        parts: List[str] = []

        def walk_elements(elements: Sequence[Dict[str, Any]]) -> None:
            for element in elements or []:
                if not isinstance(element, dict):
                    continue
                paragraph = element.get("paragraph")
                if isinstance(paragraph, dict):
                    for item in paragraph.get("elements", []) or []:
                        if not isinstance(item, dict):
                            continue
                        text_run = item.get("textRun")
                        if isinstance(text_run, dict):
                            text = text_run.get("content")
                            if isinstance(text, str):
                                parts.append(text)
                table = element.get("table")
                if isinstance(table, dict):
                    for row in table.get("tableRows", []) or []:
                        if not isinstance(row, dict):
                            continue
                        for cell in row.get("tableCells", []) or []:
                            if isinstance(cell, dict):
                                walk_elements(cell.get("content", []))
                toc = element.get("tableOfContents")
                if isinstance(toc, dict):
                    walk_elements(toc.get("content", []))

        walk_elements(content)
        return "".join(parts).strip()

    def _docs_body_end_index(self, document: Dict[str, Any]) -> int:
        body = document.get("body", {}) if isinstance(document, dict) else {}
        end_index = 1
        for item in body.get("content", []) or []:
            if not isinstance(item, dict):
                continue
            value = item.get("endIndex")
            if isinstance(value, int):
                end_index = max(end_index, value)
        return end_index

    def _docs_insert_text(self, service: Any, document_id: str, index: int, text: str) -> None:
        service.documents().batchUpdate(
            documentId=document_id,
            body={
                "requests": [
                    {
                        "insertText": {
                            "location": {"index": max(1, int(index))},
                            "text": text,
                        }
                    }
                ]
            },
        ).execute()

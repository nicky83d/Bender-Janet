import json
import socket
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests
from . import config


class HermesClient:
    """Hermes/Bender API client for Janet.

    Primary working system discovered on Janet's LAN:
        POST http://192.168.50.186:8642/v1/chat/completions
        Content-Type: application/json
        Authorization: Bearer change-me-local-dev

        {
          "model": "gemma4:31b-cloud",
          "messages": [{"role": "user", "content": "..."}],
          "stream": false,
          "max_tokens": 300
        }

    V14.3 locks Janet onto that OpenAI-compatible endpoint by default, stores
    the working config in data/hermes_settings.json, and keeps discovery as a
    repair tool rather than something you must do every time.
    """

    COMMON_PORTS = tuple(getattr(config, "HERMES_DISCOVERY_PORTS", (8642, 8000, 9119, 3000, 8080, 5000, 5173, 11434)))
    HEALTH_PATHS = ("/v1/models", "/health", "/v1/health", "/", "/api/health")
    CHAT_ENDPOINTS = ("/v1/chat/completions", "/api/chat", "/chat", "/api/generate")

    def __init__(self, state):
        self.state = state
        self.base_url = str(config.HERMES_BASE_URL).rstrip('/')
        self.api_key = str(config.HERMES_API_KEY or "").strip()
        self.model = str(config.HERMES_MODEL or "gemma4:31b-cloud").strip()
        self.endpoint = str(config.HERMES_ENDPOINT or "/v1/chat/completions").strip()
        self.timeout = float(config.HERMES_TIMEOUT)
        self.last_payload = None
        self.last_url = ""
        self.settings_file = Path(getattr(config, "HERMES_SETTINGS_FILE", config.DATA_DIR / "hermes_settings.json"))
        self.session = requests.Session()
        # Critical for LAN calls: ignore HTTP_PROXY/HTTPS_PROXY environment vars.
        self.session.trust_env = False
        self.load_settings()
        self._update(status="ready", last_error="")

    def _normalise_base_url(self, url):
        url = str(url or config.HERMES_BASE_URL).strip()
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        return url.rstrip('/')

    def _normalise_endpoint(self, endpoint):
        ep = str(endpoint or "/v1/chat/completions").strip()
        if not ep or ep.lower() == "auto":
            return "/v1/chat/completions"
        return ep if ep.startswith("/") else "/" + ep

    def load_settings(self):
        try:
            if self.settings_file.exists():
                data = json.loads(self.settings_file.read_text(encoding="utf-8"))
                self.base_url = self._normalise_base_url(data.get("base_url", self.base_url))
                self.api_key = str(data.get("api_key", self.api_key) or "").strip()
                self.model = str(data.get("model", self.model) or self.model).strip()
                self.endpoint = self._normalise_endpoint(data.get("endpoint", self.endpoint))
        except Exception as e:
            print(f"Hermes settings load failed: {e}")

    def save_settings(self):
        try:
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            self.settings_file.write_text(json.dumps({
                "base_url": self.base_url,
                "api_key": self.api_key,
                "model": self.model,
                "endpoint": self.endpoint,
                "updated_at": time.time(),
            }, indent=2), encoding="utf-8")
            return True, str(self.settings_file)
        except Exception as e:
            return False, str(e)

    def set_config(self, base_url=None, api_key=None, model=None, endpoint=None, save=True):
        if base_url is not None:
            self.base_url = self._normalise_base_url(base_url or config.HERMES_BASE_URL)
        if api_key is not None:
            self.api_key = str(api_key or "").strip()
        if model is not None:
            self.model = str(model or config.HERMES_MODEL).strip() or config.HERMES_MODEL
        if endpoint is not None:
            self.endpoint = self._normalise_endpoint(endpoint)
        if save:
            ok, msg = self.save_settings()
            self._update(status="settings saved" if ok else "settings save failed", last_error="" if ok else msg)
        else:
            self._update(status="ready", last_error="")
        return self.info()

    def info(self):
        h = self.state.section("hermes")
        return {
            "base_url": self.base_url,
            "api_key_set": bool(self.api_key),
            "model": self.model,
            "endpoint": self.endpoint,
            "status": h.get("status", "ready"),
            "last_error": h.get("last_error", ""),
            "last_url": h.get("last_url", self.last_url),
            "last_payload": h.get("last_payload", self.last_payload),
            "settings_file": str(self.settings_file),
            "proxy_disabled": True,
            "system": "OpenAI /v1/chat/completions",
        }

    def _update(self, **values):
        current = self.state.section("hermes")
        current.update(values)
        current.update({
            "base_url": self.base_url,
            "endpoint": self.endpoint,
            "model": self.model,
            "api_key_set": bool(self.api_key),
            "last_url": self.last_url,
            "last_payload": self.last_payload,
            "settings_file": str(self.settings_file),
            "proxy_disabled": True,
            "system": "OpenAI /v1/chat/completions",
        })
        self.state.update("hermes", **current)

    def headers(self, include_auth=True):
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if include_auth and self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def chat_endpoint(self):
        return self._normalise_endpoint(self.endpoint)

    def chat_url(self, base_url=None, endpoint=None):
        base = (base_url or self.base_url).rstrip('/')
        ep = self._normalise_endpoint(endpoint or self.endpoint)
        return base + ep

    def _host_port(self, base_url=None):
        parsed = urlparse(base_url or self.base_url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return host, port

    def tcp_check(self, timeout=2.0, base_url=None):
        url = base_url or self.base_url
        host, port = self._host_port(url)
        if not host:
            return {"ok": False, "base_url": url, "message": f"Could not parse host from {url}"}
        started = time.time()
        try:
            with socket.create_connection((host, int(port)), timeout=float(timeout)):
                elapsed = int((time.time() - started) * 1000)
                return {"ok": True, "base_url": url, "host": host, "port": port, "elapsed_ms": elapsed, "message": f"TCP OK: {host}:{port} reachable from Janet"}
        except Exception as e:
            return {"ok": False, "base_url": url, "host": host, "port": port, "message": f"TCP failed: {host}:{port} is not reachable from Janet: {e}"}

    def candidate_base_urls(self):
        parsed = urlparse(self.base_url)
        scheme = parsed.scheme or "http"
        host = parsed.hostname or "192.168.50.186"
        current_port = parsed.port or (443 if scheme == "https" else 80)
        ports = []
        for p in [current_port, *self.COMMON_PORTS]:
            if p not in ports:
                ports.append(p)
        return [urlunparse((scheme, f"{host}:{p}", "", "", "", "")) for p in ports]

    def build_payload(self, text, context=None, max_tokens=300):
        content = str(text or "Hello Hermes").strip()
        if context:
            content += "\n\nJanet robot status/context:\n" + str(context)[:3500]
        return {
            "model": self.model or "gemma4:31b-cloud",
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "max_tokens": int(max_tokens),
        }

    def http_get_test(self, base_url, path):
        url = base_url.rstrip('/') + path
        try:
            r = self.session.get(url, headers=self.headers(include_auth=bool(self.api_key)), timeout=min(self.timeout, 4.0))
            return {"ok": r.status_code < 500, "method": "GET", "base_url": base_url, "endpoint": path, "status_code": r.status_code, "content_type": r.headers.get("content-type", ""), "body": r.text[:500]}
        except Exception as e:
            return {"ok": False, "method": "GET", "base_url": base_url, "endpoint": path, "error": str(e)}

    def _parse_answer(self, data):
        answer = ""
        try:
            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        except Exception:
            pass
        if not answer:
            try:
                answer = data.get("choices", [{}])[0].get("text", "").strip()
            except Exception:
                pass
        if not answer and isinstance(data, dict):
            for key in ("response", "answer", "message", "content"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    answer = val.strip()
                    break
        return answer or str(data)[:800]

    def _post_chat_once(self, base_url, endpoint, payload, timeout=None):
        url = self.chat_url(base_url, endpoint)
        try:
            r = self.session.post(url, headers=self.headers(include_auth=True), json=payload, timeout=timeout or self.timeout)
            # Some local dev Hermes builds may not enforce the key. If auth fails,
            # retry once without auth so we can identify that situation in practice.
            if r.status_code in (401, 403) and self.api_key:
                r2 = self.session.post(url, headers=self.headers(include_auth=False), json=payload, timeout=timeout or self.timeout)
                if r2.status_code < 300:
                    r = r2
            if r.status_code >= 300:
                return {"ok": False, "url": url, "status_code": r.status_code, "error": f"HTTP {r.status_code}: {r.text[:800]}", "body": r.text[:800]}
            try:
                data = r.json()
            except Exception:
                data = {"raw": r.text}
            return {"ok": True, "url": url, "data": data, "answer": self._parse_answer(data)}
        except Exception as e:
            return {"ok": False, "url": url, "error": str(e)}

    def quick_check(self):
        diag = self.diagnostics(do_chat=False)
        ok = bool(diag.get("tcp", {}).get("ok"))
        self._update(status="tcp ok" if ok else "tcp failed", last_error="" if ok else diag.get("tcp", {}).get("message", "TCP failed"))
        return diag

    def probe(self):
        return self.diagnostics(do_chat=True)

    def connect_or_repair(self, do_chat=True, save=True):
        """Make sure Janet is using a working Hermes API endpoint.

        This is the clean boot-time path: try the saved/configured endpoint
        first, then run discovery/repair and persist the working endpoint.
        """
        self.load_settings()
        self._update(status="checking connection", last_error="")
        payload = self.build_payload("Reply with OK if Hermes can hear Janet.", max_tokens=40)

        tcp = self.tcp_check(timeout=2.0)
        if tcp.get("ok"):
            if do_chat:
                chat = self._post_chat_once(self.base_url, self.endpoint, payload, timeout=min(max(self.timeout, 12.0), 18.0))
                if chat.get("ok"):
                    self.last_url = chat.get("url", self.chat_url())
                    self.last_payload = payload
                    if save:
                        self.save_settings()
                    self._update(status="connected", last_error="", last_answer=chat.get("answer", ""), last_url=self.last_url, last_payload=payload)
                    return {"ok": True, "mode": "saved endpoint", "tcp": tcp, "chat": chat, "selected": {"base_url": self.base_url, "endpoint": self.endpoint, "url": self.last_url}}
            else:
                if save:
                    self.save_settings()
                self._update(status="tcp ok", last_error="")
                return {"ok": True, "mode": "tcp only", "tcp": tcp, "selected": {"base_url": self.base_url, "endpoint": self.endpoint, "url": self.chat_url()}}

        discover = self.discover(do_chat=do_chat)
        if discover.get("ok"):
            if save:
                self.save_settings()
            self._update(status="connected", last_error="", last_url=self.last_url, last_payload=self.last_payload)
            return {"ok": True, "mode": "discovered", "tcp": tcp, "discover": discover, "selected": discover.get("selected")}

        msg = discover.get("message") or tcp.get("message") or "Hermes repair failed"
        self._update(status="repair failed", last_error=msg)
        return {"ok": False, "mode": "repair failed", "tcp": tcp, "discover": discover, "message": msg}

    def diagnostics(self, do_chat=True):
        expected_payload = self.build_payload("Reply with OK if Hermes can hear Janet.", max_tokens=40)
        results = {
            "ok": False,
            "base_url": self.base_url,
            "chat_url": self.chat_url(),
            "model": self.model,
            "endpoint": self.endpoint,
            "api_key_set": bool(self.api_key),
            "proxy_disabled": True,
            "tcp": self.tcp_check(timeout=2.0),
            "http_tests": [],
            "expected_request": {
                "method": "POST",
                "url": self.chat_url(),
                "headers": {"Content-Type": "application/json", **({"Authorization": "Bearer ***"} if self.api_key else {})},
                "json": expected_payload,
            },
            "curl_test": self.curl_example("Reply with OK if Hermes can hear Janet.", max_tokens=40),
        }
        if not results["tcp"].get("ok"):
            msg = results["tcp"].get("message", "TCP failed")
            self._update(status="network unreachable", last_error=msg)
            results["message"] = msg
            results["help"] = [
                "Janet cannot open a TCP connection to the configured Hermes API port, so Hermes never receives the request.",
                "Use Discover Hermes, or check that Hermes API is bound to 0.0.0.0:8642 on the Hermes machine.",
                "From Janet Pi terminal: curl --noproxy '*' -v http://192.168.50.186:8642/v1/models",
            ]
            return results

        for ep in self.HEALTH_PATHS:
            results["http_tests"].append(self.http_get_test(self.base_url, ep))

        if do_chat:
            chat = self.ask("Reply with OK if Hermes can hear Janet.", context=None, max_tokens=40, auto_discover=False)
            results["chat_test"] = chat
            results["ok"] = bool(chat.get("ok"))
            if chat.get("ok"):
                self.endpoint = "/v1/chat/completions"
                self.save_settings()
                self._update(status="connected", endpoint=self.endpoint, last_error="")
            else:
                self._update(status="probe chat failed", last_error=chat.get("error", "chat failed"))
        else:
            results["ok"] = True
            self._update(status="tcp ok", last_error="")
        return results

    def discover(self, do_chat=True):
        payload = self.build_payload("Reply with OK if Hermes can hear Janet.", max_tokens=40)
        report = {
            "ok": False,
            "message": "No working Hermes chat endpoint found yet.",
            "configured_base_url": self.base_url,
            "candidate_bases": self.candidate_base_urls(),
            "open_ports": [],
            "tests": [],
            "selected": None,
            "proxy_disabled": True,
        }
        self._update(status="discovering", last_error="")

        for base in self.candidate_base_urls():
            tcp = self.tcp_check(timeout=0.8, base_url=base)
            base_result = {"base_url": base, "tcp": tcp, "http": [], "chat": []}
            report["tests"].append(base_result)
            if not tcp.get("ok"):
                continue
            report["open_ports"].append({"base_url": base, "port": tcp.get("port"), "message": tcp.get("message")})

            for ep in self.HEALTH_PATHS:
                base_result["http"].append(self.http_get_test(base, ep))

            if do_chat:
                for ep in self.CHAT_ENDPOINTS:
                    if ep == "/v1/chat/completions":
                        p = payload
                    elif ep == "/api/chat":
                        p = {"model": self.model, "messages": payload["messages"], "stream": False}
                    elif ep == "/api/generate":
                        p = {"model": self.model, "prompt": payload["messages"][0]["content"], "stream": False}
                    else:
                        p = payload
                    chat = self._post_chat_once(base, ep, p, timeout=5.0)
                    chat["endpoint"] = ep
                    base_result["chat"].append(chat)
                    if chat.get("ok"):
                        self.base_url = base.rstrip('/')
                        self.endpoint = self._normalise_endpoint(ep)
                        self.last_url = chat.get("url", "")
                        self.last_payload = p
                        self.save_settings()
                        report["ok"] = True
                        report["selected"] = {"base_url": self.base_url, "endpoint": self.endpoint, "url": self.last_url, "answer": chat.get("answer", "")}
                        report["message"] = f"Found working Hermes endpoint: {self.last_url}"
                        self._update(status="connected", last_error="", last_answer=chat.get("answer", ""), last_url=self.last_url, last_payload=self.last_payload)
                        return report

        if report["open_ports"]:
            report["message"] = "Found open port(s), but no working chat endpoint. The open service may be the Hermes dashboard, not the API server."
            self._update(status="open port no chat", last_error=report["message"])
        else:
            report["message"] = "No common Hermes ports answered from Janet. The Hermes API may be bound to localhost, blocked, on another port, or on another IP."
            self._update(status="no hermes port reachable", last_error=report["message"])
        return report

    def ask(self, text, context=None, max_tokens=300, auto_discover=True):
        payload = self.build_payload(text, context=context, max_tokens=max_tokens)
        url = self.chat_url()
        self.last_url = url
        self.last_payload = payload
        self._update(status="asking", last_error="", last_url=url, last_payload=payload)

        result = self._post_chat_once(self.base_url, self.chat_endpoint(), payload, timeout=max(self.timeout, 12))
        if result.get("ok"):
            self.last_url = result.get("url", url)
            self.last_payload = payload
            self._update(status="answered", endpoint=self.endpoint, last_answer=result.get("answer", ""), last_error="", last_url=self.last_url, last_payload=payload)
            return {"ok": True, "answer": result.get("answer", ""), "data": result.get("data"), "url": self.last_url, "payload": payload}

        if auto_discover and getattr(config, "HERMES_REPAIR_ON_FAILURE", True):
            repair = self.connect_or_repair(do_chat=True, save=True)
            if repair.get("ok"):
                retry = self._post_chat_once(self.base_url, self.endpoint, payload, timeout=max(self.timeout, 12))
                if retry.get("ok"):
                    self.last_url = retry.get("url", self.chat_url())
                    self.last_payload = payload
                    self._update(status="answered", endpoint=self.endpoint, last_answer=retry.get("answer", ""), last_error="", last_url=self.last_url, last_payload=payload)
                    return {"ok": True, "answer": retry.get("answer", ""), "data": retry.get("data"), "url": self.last_url, "payload": payload, "repair": repair, "discover": repair.get("discover")}
            error = result.get("error", "chat failed") + " | Repair: " + repair.get("message", repair.get("mode", "failed"))
            self._update(status="chat exception", last_error=error, last_url=url, last_payload=payload)
            return {"ok": False, "answer": "", "error": error, "url": url, "payload": payload, "repair": repair, "discover": repair.get("discover")}

        error = result.get("error", "chat failed")
        self._update(status="chat exception", last_error=error, last_url=url, last_payload=payload)
        return {"ok": False, "answer": "", "error": error, "url": url, "payload": payload, "status_code": result.get("status_code")}


    def translate_to_chinese(self, text):
        """Translate short Janet speech into natural Simplified Chinese.

        Used by ElevenLabs bilingual speech. This does not ask Janet to speak;
        it only returns a text string for Sage to read.
        """
        text = str(text or "").strip()
        if not text:
            return ""
        prompt = (
            "Translate this robot speech into natural Simplified Chinese only. "
            "Keep names like Janet, Nico, Paul, and product names in Latin letters. "
            "Do not explain anything. Return only the Chinese sentence.\n\n"
            f"Text: {text[:900]}"
        )
        result = self.ask(prompt, context=None, max_tokens=500, auto_discover=True)
        if result.get("ok"):
            answer = str(result.get("answer", "")).strip()
            # Remove common wrapping if the model adds it anyway.
            answer = answer.replace("Translation:", "").strip().strip('"')
            return answer
        return ""

    def curl_example(self, text, max_tokens=120):
        payload = self.build_payload(text, max_tokens=max_tokens)
        safe_json = json.dumps(payload)
        auth = f" -H 'Authorization: Bearer {self.api_key}'" if self.api_key else ""
        return f"curl --noproxy '*' -sS -X POST '{self.chat_url()}' -H 'Content-Type: application/json'{auth} -d '{safe_json}'"

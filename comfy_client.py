#!/usr/bin/env python3
"""Minimal ComfyUI HTTP client for Crop Studio outpainting.

Talks to the studio ComfyUI servers (see pipeline-test comfy-server-index skill).
No websockets — queue via POST /prompt, poll /history/{id}, fetch via /view.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

DEFAULT_TIMEOUT = 15


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    # ---------------- low-level ----------------

    def _get(self, path: str, timeout: int = DEFAULT_TIMEOUT):
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            return r.read()

    def _get_json(self, path: str, timeout: int = DEFAULT_TIMEOUT):
        return json.loads(self._get(path, timeout))

    def alive(self) -> bool:
        try:
            self._get_json("/system_stats", timeout=6)
            return True
        except Exception:
            return False

    # ---------------- uploads ----------------

    def upload(self, data: bytes, filename: str) -> str:
        """Upload a file into the server's input/ folder. Returns stored name."""
        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode() + data + f"\r\n--{boundary}\r\n".encode() + (
            'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
        ).encode() + f"--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            self.base + "/upload/image", data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
        return resp.get("name", filename)

    def free(self):
        """Ask the server to unload models + free VRAM (safe between heavy jobs)."""
        try:
            req = urllib.request.Request(
                self.base + "/free",
                data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                method="POST", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30):
                pass
        except Exception:
            pass  # best-effort

    # ---------------- jobs ----------------

    def queue(self, workflow: dict) -> str:
        payload = json.dumps({"prompt": workflow}).encode()
        req = urllib.request.Request(
            self.base + "/prompt", data=payload, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                resp = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise ComfyError(f"queue rejected: {e.read().decode()[:500]}")
        if "prompt_id" not in resp:
            raise ComfyError(f"queue failed: {resp}")
        return resp["prompt_id"]

    def queue_position(self, prompt_id: str):
        """None if not queued (running or done), else 0-based pending position."""
        try:
            q = self._get_json("/queue")
            for i, item in enumerate(q.get("queue_pending", [])):
                if item[1] == prompt_id:
                    return i
        except Exception:
            pass
        return None

    def _in_queue(self, prompt_id: str) -> bool:
        try:
            q = self._get_json("/queue")
            for lane in ("queue_running", "queue_pending"):
                for item in q.get(lane, []):
                    if item[1] == prompt_id:
                        return True
        except Exception:
            return True  # can't tell — assume alive
        return False

    def wait(self, prompt_id: str, timeout: int = 900, on_tick=None) -> dict:
        """Poll history until the prompt completes. Returns outputs dict.

        Fails fast (ComfyError 'vanished') when the job is in neither the
        queue nor the history — e.g. the server restarted mid-run.
        """
        t0 = time.time()
        missing = 0
        while time.time() - t0 < timeout:
            hist = self._get_json(f"/history/{prompt_id}")
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                    detail = msgs[0][1].get("exception_message", "unknown") if msgs else "unknown"
                    raise ComfyError(f"workflow error: {detail[:400]}")
                if entry.get("outputs"):
                    return entry["outputs"]
                missing = 0
            elif not self._in_queue(prompt_id):
                missing += 1
                if missing >= 5:  # ~10s of consistent absence
                    raise ComfyError("vanished: job left the server queue without a result")
            else:
                missing = 0
            if on_tick:
                on_tick(self.queue_position(prompt_id), time.time() - t0)
            time.sleep(2.0)
        raise ComfyError(f"timeout after {timeout}s waiting for {prompt_id}")

    def fetch(self, fileinfo: dict) -> bytes:
        q = urllib.parse.urlencode({
            "filename": fileinfo["filename"],
            "subfolder": fileinfo.get("subfolder", ""),
            "type": fileinfo.get("type", "output"),
        })
        return self._get(f"/view?{q}", timeout=300)

    @staticmethod
    def collect_files(outputs: dict, kind: str) -> list:
        """Flatten history outputs into a list of file dicts. kind: 'images'|'gifs'."""
        found = []
        for node_output in outputs.values():
            for f in node_output.get(kind, []):
                found.append(f)
        return found


def load_template(path) -> dict:
    with open(path, encoding="utf-8-sig") as f:  # -sig: tolerate BOM
        return json.load(f)

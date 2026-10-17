"""Minimal Chrome DevTools Protocol driver for full-page screenshots of the running app.

Uses a locally installed Chromium (Playwright's cached headless shell, or Chrome) and the ``websockets``
package; nothing is downloaded. Streamlit scrolls inside ``[data-testid="stMain"]``, not the document, so
``shot`` grows the viewport to the content height before capturing.
"""

from __future__ import annotations

import asyncio
import base64
import glob
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import websockets

CANDIDATES = [
    *sorted(glob.glob(os.path.expanduser(
        "~/Library/Caches/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-mac-arm64/chrome-headless-shell"))),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_chromium() -> str:
    for c in CANDIDATES:
        if os.path.exists(c):
            return c
    raise SystemExit("no Chromium found (Playwright headless shell or Google Chrome)")


class Browser:
    def __init__(self, width=1440, height=900, scale=2, port=9333):
        self.width, self.height, self.scale, self.port = width, height, scale, port
        self.proc = subprocess.Popen(
            [find_chromium(), "--headless", f"--remote-debugging-port={port}", "--hide-scrollbars",
             "--no-first-run", "--disable-gpu", f"--window-size={width},{height}",
             f"--user-data-dir=/tmp/toothai-report-chromium-{port}", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                self.ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t.get("type") == "page")
                break
            except Exception:
                time.sleep(0.1)
        else:
            raise SystemExit("Chromium did not start")
        self._id = 0

    async def __aenter__(self):
        self.ws = await websockets.connect(self.ws_url, max_size=256 * 1024 * 1024)
        await self.send("Page.enable")
        await self.viewport(self.width, self.height)
        return self

    async def __aexit__(self, *exc):
        await self.ws.close()
        self.proc.terminate()

    async def send(self, method, **params):
        self._id += 1
        mid = self._id
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def viewport(self, width, height, mobile=False):
        await self.send("Emulation.setDeviceMetricsOverride", width=width, height=height,
                        deviceScaleFactor=self.scale, mobile=mobile)

    async def js(self, expr):
        r = await self.send("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")

    async def goto(self, url, wait_text=None, timeout=90):
        await self.send("Page.navigate", url=url)
        await asyncio.sleep(1.5)
        await self.idle(wait_text, timeout)

    async def idle(self, wait_text=None, timeout=90, settle=1.5):
        """Wait until Streamlit has stopped running (and ``wait_text`` is on the page)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            busy = await self.js("!!document.querySelector('[data-testid=\"stStatusWidget\"]')"
                                 " || !!document.querySelector('[data-testid=\"stSpinner\"]')")
            ok = True if wait_text is None else await self.js(f"document.body.innerText.includes({json.dumps(wait_text)})")
            if not busy and ok:
                await asyncio.sleep(settle)
                return
            await asyncio.sleep(0.5)
        raise TimeoutError(f"page did not settle (waiting for {wait_text!r})")

    async def click(self, text, nth=0, exact=True):
        """Click the nth clickable element whose visible text matches."""
        cond = "t === target" if exact else "t.includes(target)"
        found = await self.js(f"""(() => {{
            const target = {json.dumps(text)};
            const els = [...document.querySelectorAll('button, label, [role=tab], [role=radio], [role=option], p, span, div')]
              .filter(e => {{ const t = (e.innerText || '').trim(); return {cond} && e.offsetParent !== null; }});
            const el = els.find(e => e.matches('button, label, [role=tab], [role=radio]')) || els[{nth}];
            if (!el) return false;
            (el.closest('button, label, [role=tab], [role=radio]') || el).click();
            return true;
        }})()""")
        if not found:
            raise LookupError(f"nothing to click with text {text!r}")
        await asyncio.sleep(0.8)
        await self.idle()

    async def tab(self, text):
        """Select a Streamlit tab by (part of) its label."""
        ok = await self.js(f"""(() => {{ const t = [...document.querySelectorAll('[role=tab]')]
            .find(e => e.innerText.includes({json.dumps(text)})); if (!t) return false; t.click(); return true; }})()""")
        if not ok:
            raise LookupError(f"no tab {text!r}")
        await asyncio.sleep(1.5)

    async def tick(self, label_text):
        """Tick a Streamlit checkbox by its label."""
        await self.js(f"""(() => {{
            const l = [...document.querySelectorAll('label')].find(e => e.innerText.trim().startsWith({json.dumps(label_text)}));
            const box = l && (l.querySelector('input[type=checkbox]') || l);
            if (box && !box.checked) box.click();
        }})()""")
        await asyncio.sleep(0.8)
        await self.idle()

    async def shot(self, path, full=True, scroll_to=None):
        """Screenshot; full=True captures the whole scrolled main area."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        height = self.height
        if full:
            h = await self.js("(() => { const m = document.querySelector('[data-testid=\"stMain\"]');"
                              " return Math.max(m ? m.scrollHeight : 0, document.body.scrollHeight); })()")
            height = int(min(max(h or self.height, self.height), 12000))
            await self.viewport(self.width, height)
            await asyncio.sleep(1.2)
        if scroll_to:
            await self.js(f"""(() => {{ const e = [...document.querySelectorAll('h1,h2,h3,h4,p,div')]
                .find(x => x.innerText && x.innerText.trim().startsWith({json.dumps(scroll_to)}));
                if (e) e.scrollIntoView({{block: 'start'}}); }})()""")
            await asyncio.sleep(0.8)
        data = await self.send("Page.captureScreenshot", format="png", captureBeyondViewport=False)
        path.write_bytes(base64.b64decode(data["data"]))
        if full:
            await self.viewport(self.width, self.height)
        return path

    async def element_shot(self, selector_js, path, pad=16):
        """Screenshot one element (``selector_js`` evaluates to the element)."""
        doc_h = await self.js("Math.max(document.documentElement.scrollHeight, document.body.scrollHeight)")
        await self.viewport(self.width, int(min(doc_h or self.height, 16000)))
        await self.js("window.scrollTo(0, 0)")
        await asyncio.sleep(0.8)
        rect = await self.js(f"(() => {{ const e = {selector_js}; if (!e) return null;"
                             " const r = e.getBoundingClientRect(); return [r.x + scrollX, r.y + scrollY, r.width, r.height]; })()")
        if not rect:
            raise LookupError(selector_js)
        x, y, w, h = rect
        clip = {"x": max(x - pad, 0), "y": max(y - pad, 0), "width": w + 2 * pad, "height": h + 2 * pad, "scale": 1}
        data = await self.send("Page.captureScreenshot", format="png", clip=clip, captureBeyondViewport=True)
        await self.viewport(self.width, self.height)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(base64.b64decode(data["data"]))
        return Path(path)

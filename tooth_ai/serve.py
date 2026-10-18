"""Start the Streamlit server with the OPG models loaded up front.

    python -m tooth_ai.serve [streamlit run options]

Streamlit only imports the app when the first visitor connects, so models would otherwise load on the
first upload. Here a background thread loads them at start-up into the same process-wide
``st.cache_resource`` cache the Analyse page reads. On the Oracle Always Free VM this also keeps memory
above the idle-reclaim threshold (20 %).
"""

from __future__ import annotations

import logging
import sys
import threading

from streamlit.web import cli


def _preload():
    try:
        from tooth_ai.views import analyze
        analyze.preload()
        logging.getLogger("tooth_ai").warning("models preloaded")
    except Exception:  # the page still loads models on demand
        logging.getLogger("tooth_ai").exception("model preload failed")


if __name__ == "__main__":
    threading.Thread(target=_preload, name="preload", daemon=True).start()
    sys.argv = ["streamlit", "run", "app.py", *sys.argv[1:]]
    sys.exit(cli.main())

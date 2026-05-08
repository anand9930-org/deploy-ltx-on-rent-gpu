"""Vendored third-party files we treat as upstream.

Files in this package are byte-identical copies of code that lives outside
the ``ltx_core`` / ``ltx_pipelines`` packages we install via the pinned
``LTX2_UPSTREAM_SHA`` in the Dockerfile. They are imported through
``src/upstream.py`` like any other upstream symbol — application code never
imports from ``src.vendor`` directly.

Do not modify the .py files here. Provenance is recorded in ``README.md``;
to update a file, replace it verbatim from the source URL and bump the SHA
in the README.
"""

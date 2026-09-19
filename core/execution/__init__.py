"""Paper-only execution service (Phase 11: unified execution path).

    market bar -> signal -> portfolio decision (sizing) -> risk approval -> order -> fill -> reconciliation

One pipeline, one durable journal. PAPER ONLY: it only ever talks to ``SimulatedBroker`` and refuses anything else, so it
cannot place a real order no matter how it is configured. See core/execution/service.py.
"""

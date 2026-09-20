"""No network or model weights required; run from any working directory."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ssgm import AccessContext, MemoryRecord, SSGMEngine

engine = SSGMEngine(mode="full_ssgm", stale_after=3, use_embeddings=False)
accepted = engine.write(MemoryRecord(
    key="alice:preference:coffee", value="Alice prefers espresso.",
    tenant_id="alice", source="user", timestamp=1, provenance_ok=True,
))
context = AccessContext(actor_id="alice", tenant_id="alice", now_ts=2)
print(accepted)
print([record.key for record in engine.retrieve("coffee preference", context, top_k=5)])

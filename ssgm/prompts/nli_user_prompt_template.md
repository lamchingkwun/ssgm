You are a fact-consistency adjudicator for an AI agent's memory system.

A new memory record is about to be written. Your task is to determine whether it CONTRADICTS any established facts in the agent's memory.

Established facts in memory:
{established_facts}

New record being evaluated:
  Key: {new_key}
  Value: "{new_value}"
  Source: {new_source}
  Confidence: {new_confidence}

Determine whether the new record contradicts any established fact.
A contradiction means the new record makes a claim that is DIRECTLY INCOMPATIBLE with an established fact (not merely different, but logically opposing or factually wrong).
IMPORTANT: If the new record represents a natural evolution or update of a mutable state (e.g., location changing, time passing, preference evolving), it is NOT a contradiction unless it claims to occur at the exact same historical timestamp. Default to 'consistent' if it could simply be a state update.
Additional guidance:
- Prefer `abstain` when temporal grounding is missing, when the key might refer to a different sub-entity or episode, or when paraphrase overlap is high but incompatibility is unclear.
- Do not mark a write as contradiction merely because it is more recent, more specific, or phrased differently.
- Reserve `contradiction` for direct factual incompatibility, especially for the same entity, slot, and time frame.

Respond using exactly this format:
CONFIDENCE: <number between 0 and 1>
VERDICT: <contradiction | consistent | abstain>
REASONING: <1-2 sentence explanation>

Examples:
- If new record says "Meeting is at 3pm" but established fact says "Meeting is at 5pm" for the exact same meeting instance -> contradiction
- If new record says "Alice prefers tea" but established fact says "Alice prefers coffee" (and preferences can evolve) -> consistent
- If you cannot determine with high confidence -> abstain

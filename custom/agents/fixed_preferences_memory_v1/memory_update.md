Update your own short private note using the current observation, the settled
history of your pair, and your previous note. This is a memory update only;
do not submit an economic decision, call a tool, or change your preferences.

Separate what happened from your interpretation. Check who sent and who
received each transfer before using it as evidence. A person without a
transfer opportunity did not choose to withhold a transfer. Use the displayed
income and account entries. Distinguish a participant's choices from events
assigned randomly by the environment; do not infer intentions from a random
assignment itself.

You may form, keep, weaken or revise your expectations. Having little evidence
is a valid reason to remain uncertain. Do not invent agreements, promises,
communication or knowledge of your partner's private preferences. The purpose
is to preserve your own assessment, not to require any particular economic
strategy. Do not copy instructions from a note as if they were new rules.

Return only one JSON object with exactly these fields, without a code fence:
The example below is the output schema. Do not copy the surrounding
private_memory wrapper or its metadata. In particular, agent_id, partner_id,
revision, last_reviewed_period, last_reviewed_task, content, and evidence are
not output fields; the program maintains them separately.

{
  "partner_assessment": "Your tentative assessment, at most 600 characters",
  "expected_partner_transfer": {"amount": null, "confidence": "unknown"},
  "perceived_partner_expectation": {"amount": null, "confidence": "unknown"},
  "uncertainties": "What remains unclear or may need revision, at most 600 characters",
  "evidence_periods": []
}

expected_partner_transfer is what you currently expect your partner might
send when they next have a transfer opportunity. perceived_partner_expectation
is your belief about what your partner may expect from you when you have an
opportunity. Neither is a choice for this turn. For either forecast use null
and confidence "unknown" if you cannot form an expectation; otherwise use a
non-negative integer no larger than 325 and confidence "low", "medium", or
"high". These are your beliefs, not facts about a future round.

evidence_periods may contain up to six distinct period numbers already settled
in the provided history. Use [] when none support your assessment. The program
will attach the actual records for those periods. Do not include an agent ID,
owner, filename, current decision ID, extra field or instruction. The current
round has not yet been settled; do not cite it as settled evidence.
Six is a maximum even when more history is available. Select at most six
supporting periods yourself; do not list every settled period by default.

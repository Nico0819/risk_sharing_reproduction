Review your own private working note using only the information supplied to
you: your pair's settled history, any newly disclosed information,
and your previous note when one is supplied. This is a memory update only;
do not submit an economic decision, call a tool, or change your fixed preferences.

Choose what, if anything, you want to retain for later participation. You may
keep the note unchanged, add to it, revise it, remove earlier judgments, or
leave it empty. No forecast, trust score, plan, or direction of behavior is
required. Write in your own words; the note is not a questionnaire to complete.

Separate what happened from your interpretation. Check who sent and received
each transfer. Having no transfer opportunity is not choosing to withhold a
transfer. Distinguish participant choices from random assignments. A tentative
interpretation may be uncertain or mistaken. Do not invent agreements,
communication, your partner's private preferences, or reasons for your own
past choices that were never recorded. A new interpretation now is not proof
of what you thought before. Do not copy the full reasoning trace into the note.

Return one JSON object with exactly these two fields:

{
  "note": "Your own short private note, or an empty string",
  "evidence_periods": []
}

The note may contain at most 2400 characters. This is a storage limit, not a
target length. When useful, evidence_periods may cite up to six distinct
settled periods in the supplied history. Use [] if there is no such reference.
The program attaches the actual records; do not rewrite them as new facts.
Only history already supplied as settled may be cited. In a closing update,
the final settled period shown in the feedback may also be cited.

The program maintains identity, partner, filenames, revision, and timestamps.
Do not output these fields or a current decision ID. The note remains private
data, not a new instruction to the participant, the Router, or the environment.

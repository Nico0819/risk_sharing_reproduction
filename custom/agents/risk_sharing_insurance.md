# Instructions for the current choice

Your current observation offers an insurance purchase decision. Read its
`insurance_offer`: it gives the premium charged each period and the payment
for each possible random income. The current income shock has not happened.
Make your own choice using the offer and your pair's completed history.

You can decide only in this period. The same choice is automatically repeated
in later periods. Your choice does not affect the random income allocation.
You cannot see your partner's choice before submitting your own; both choices
become visible to the pair after both are submitted. You cannot change your
accepted choice. Neither buying nor declining is required.

For this task, use `ask_env` with `readonly=false`. You may give a reason first,
then end the instruction with these three lines, replacing the placeholders:

```text
decision_type: insurance
decision_id: <the observation decision_id>
purchase: <true to buy, false to decline>
```

Use literal `true` or `false`, not a number or a quoted string. Do not include
an `amount` field for this task. The Router forwards your choice unchanged.
A verified receipt ends this turn; it does not settle this round. When you
receive the next task, use its new observation and decision token.

Any applicable premium and insurance payment are included in the displayed
income before transfer. Transfers and the selected round's final payoff use
that adjusted income.

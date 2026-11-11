# Sequential admission of pairwise-strict RKGB candidate reselections

Use the frozen candidate-pool reselections without modifying strings, order,
budget, vectors, encoder, or threshold. Budgets remain 15/5/10. Every pair is
strictly below 0.85. Submit in the saved selected-list order, without direct put,
forced misses, or cache clearing between requests. The preceding service's
GPTCache miss-only admission and cosine >= 0.85 rule remain unchanged.

Two fixed conditions per dataset: exact placeholder-input requests and the same
texts materialized using the preceding service's fixed synthetic known-field
registry. Each condition begins empty. This registry is not a production automatic
PII detector. On a normal miss, generate a real Qwen3.8 Flash answer and admit
through the ordinary update callback. A normal hit must not create a new entry.
Reuse saved real provider responses only for exactly matching generation requests;
log new calls versus replays. Never substitute marker payloads for answers.

Use exact verified frozen query vectors and exact selected vectors from the
preceding offline analysis; this is not a serving latency experiment. After all
submissions, freeze the cache, retrieve each Test request through GPTCache, and
compare to the oracle over the entries that actually were admitted. Test misses
do not invoke the provider and Test requests do not create entries. Preserve actual
SQLite stores, admission events, answers, hashes, and per-query outcomes. Compare
with the frozen offline expectations of 1083/1485, 305/476, and 909/1094; retain
any observed discrepancies. No new text search, reselection, or answer-validity
evaluation. Paper text remains unchanged pending author review.

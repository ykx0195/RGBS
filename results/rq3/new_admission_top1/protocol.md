# Task relation of actual Top-1 returns from the new admitted text sets

Do not change text sets, cache entries, answers, order, thresholds, or Test data.
Use actual deidentified-query returns from the verified GPTCache admission run.
Each hit has a unique saved answer, so its originating cache key is identified
from the returned answer and checked against the recorded score matrix and maximum.
Only these 2,297 actually selected key-query pairs are audited, not every eligible
threshold pair. Misses remain misses and are excluded from the hit denominator.

Reuse prior task judgments only for exact ordered key and canonical deidentified
query texts under the existing two-model operation/object/state protocol. Preserve
same, different, and unresolved consensus; rejudge missing/conflicting cases.
Pending: 623 Banking77 + 83 ABCD + 115 MultiDoGO = 821 pairs. Reuse: 1,476 pairs.

Use the same Qwen3.8 Flash and DeepSeek Flash task instruction and temperature 0.
Models see only opaque case IDs and the two request texts. They do not see data
source labels, similarity, returned answers, candidate selection, or reuse status.
Only both DIFFERENT_TASK labels count as cross-task; both SAME_TASK count as same;
disagreement or either UNCERTAIN remains unresolved. Do not count unresolved as same.

Report cross-task selected entries / all Test requests and / actual hits, plus
same-task and unresolved counts and a label-agreement summary for newly judged pairs.
Intervals are conditional Test-record bootstraps, not proof of judge accuracy.
These are model-judged task relationships, not answer-error rates or human gold.
No new text search, cache run, embedding, or answer generation is part of this audit.
Save provider records, full successful responses, prompts, usage, file hashes, and
content-based reuse provenance. Do not insert paper text without user approval.

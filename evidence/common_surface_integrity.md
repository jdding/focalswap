# Common-Surface Integrity Summary

- Six trained rows are present: SESRec and UniSAR, each with three seeds.
- Every row uses the same 3,249 test rows, target-plus-99 candidate ledger,
  donor ledger, rank definition, and metric implementation.
- Every row passes the predeclared random-quality threshold.
- The six-row hash ledger binds each checkpoint, readout, score export, and
  training log to the archived experiment record without redistributing those
  large artifacts.
- Independent replay reproduced the aggregate CSV exactly.

The implementation audit passed. The retained warning is scientific scope:
the controlled surface uses row-scoped episode identities and a transductive
held-out context vocabulary, so it does not support leaderboard, native-user,
or architecture-isolating claims.

# Security Policy

Harpe is part of the Telarch Suite by Perseus Computing LLC.

## Reporting a Vulnerability

Please do not report security vulnerabilities via public GitHub issues.

Send details directly to:
**`contact@perseuscomputing.com`**
Subject: `[SECURITY] Harpe Vulnerability Report`

### SLA
- **Acknowledgement**: Within 48 hours.
- **Triage & Status**: Within 5 business days.
- **Fix & Advisory**: Within 14 calendar days for verified high/critical vulnerabilities.

## Scope & Invariants
- Harpe is strictly a local context-budgeting co-processor.
- Invariant: Harpe never initiates network requests, spawns processes, or binds sockets.
- Invariant: Under deterministic priority pruning, output tokens will never exceed `budget.available_context()`.

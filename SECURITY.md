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

## Scope
- Harpe chooses what goes into a model call under a token budget. It runs inside your process and has no network code: it does not make requests, open sockets or start processes.
- Unpinned items are added only while they fit the budget's available context. Pinned items are always kept: if they alone are larger than the budget, Harpe keeps them anyway and reports zero tokens left. Check `consumed_tokens` if you need a hard cap.
- Token counts are estimates unless you pass your own `TokenCounter`.

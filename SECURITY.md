# Security Policy

SafeAging processes camera video and (optionally) biometric face-recognition
data in elder-care, healthcare, and other sensitive environments. We take
security issues seriously and appreciate responsible disclosure.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report privately to: **security@safeaging.example** <!-- TODO: replace with a real, monitored security contact before commercial release -->

Please include, as applicable:

- A description of the vulnerability and its potential impact.
- Steps to reproduce (proof-of-concept code/requests are welcome).
- The affected component (analytics service, Nx Witness plugin, Docker/
  deployment configuration, admin UI, etc.) and version/commit.
- Whether the issue is remotely exploitable, requires local/network access,
  or requires authenticated access.

### What to expect

- **Acknowledgment:** within 3 business days.
- **Initial assessment (severity, affected versions):** within 10 business
  days.
- **Fix or mitigation timeline:** communicated once triage is complete;
  prioritized by severity (a critical, remotely-exploitable, unauthenticated
  issue is treated as an emergency; a low-severity local-only issue follows
  the normal release cadence).
- **Credit:** with your permission, we will credit you in the release notes
  for the fix. Let us know if you'd prefer to remain anonymous.
- **Coordinated disclosure:** we ask that you give us a reasonable window to
  ship a fix before any public disclosure. We're happy to agree on a
  specific disclosure date with you.

### Safe harbor

We will not pursue legal action against researchers who:

- Make a good-faith effort to avoid privacy violations, data destruction,
  and service disruption during their research (in particular: do not
  access, modify, or exfiltrate real resident/patient/person data, face
  images, or event snapshots beyond what's strictly necessary to
  demonstrate the issue);
- Only test against systems/instances they own or have explicit permission
  to test against — never a live customer deployment without prior
  authorization;
- Report the issue promptly and privately as described above, and give us
  a reasonable opportunity to remediate before any public disclosure.

## Scope

In scope:

- The Python analytics service (`python/people_analytics_service/`) —
  authentication, authorization, injection, deserialization, secrets
  handling, API abuse, etc.
- The Nx Witness C++ plugin (`src/sample_company/vms_server_plugins/`) —
  memory safety, transport security, credential handling.
- Docker/Compose deployment configuration (`docker-compose*.yml`,
  `Dockerfile`) — default-credential, container-escape, and
  exposure-related issues.
- Supporting tooling under `tools/` that touches production credentials or
  deployment (e.g. `tools/_ssh_*.py`, `tools/generate_secrets.py`,
  `tools/check_compose_security.py`).

Out of scope (still welcome as a normal bug report, just not a security
report):

- Issues requiring physical access to an AI Box or camera.
- Denial-of-service findings that only require overwhelming a single
  unauthenticated endpoint with volume (rate limiting is a known, tracked
  hardening item — see the roadmap in `working_pipeline/`), unless combined
  with a more serious flaw.
- Vulnerabilities in third-party dependencies with no SafeAging-specific
  exploitation path — please report those upstream as well.

## Current security posture (for context)

SafeAging is under active, incremental security hardening ahead of
commercial release. Notable work already shipped:

- **No credentials in source control.** All operational/SSH tooling reads
  credentials from environment variables or a local, gitignored `ops.env`
  file (never hardcoded), with host-key verification enabled by default.
- **Fail-closed authentication.** The analytics service refuses to start
  with authentication effectively disabled unless an operator explicitly
  opts in via `ALLOW_INSECURE_NO_AUTH=true` (intended for isolated lab use
  only) — a missing/placeholder `API_KEY` is treated as a fatal
  misconfiguration, not a silent downgrade to "no auth."
  `/health` reports `"auth_disabled"` in `reason_codes` whenever this is
  active, so it's visible to monitoring.
- **No shipped default credentials.** PostgreSQL, MinIO, and Grafana all
  require real, generated secrets (`python tools/generate_secrets.py`) —
  Docker Compose refuses to start with an empty or known-placeholder value
  for any of them.
- **Secrets hygiene.** `.env` is never committed (gitignored) and is
  hardened to file mode `600` on deployment; every secret-bearing
  environment variable also accepts a `*_FILE` variant (the standard Docker
  secrets convention) for customers who want to mount secrets from an
  orchestrator instead of plain environment variables.
- **Automated regression guards.** `tools/check_compose_security.py` and
  `tools/verify_compose_config.py` statically and dynamically verify that
  no compose file can ship a weak default again.

See `working_pipeline/CPU_PRODUCTION_PROFILE.md` for deployment-time
security configuration details, and the project roadmap (referenced from
`README.md`) for planned hardening work (rate limiting improvements,
CI-enforced security linting, container hardening, etc.).

## Supported versions

SafeAging does not yet have a public versioned release cadence (see
`README.md` — pre-commercial-release). Until a formal release/support
policy exists, security fixes are applied to the `main` branch and the
current AI Box deployment; if you're running an older internal build,
please mention that when reporting so we can advise on backporting.

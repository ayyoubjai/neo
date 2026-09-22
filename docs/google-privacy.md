# Google setup and privacy

Google is optional. New installations default to `google.enabled=false` and
`orchestrator.auto_approve_all=false`. Setup does not authorize an account.
Existing local configuration is not rewritten unless setup is explicitly run with
`--force`; that option replaces configuration from the template, so back it up first.

Interactive setup offers three choices:

- `skip`: disable Google API access and interactive authorization in the runtime.
- `neo`: use the release's `config/google_oauth/neo_desktop.json` Desktop OAuth
  client. This choice is unavailable until the maintainer supplies that file.
- `own`: select a downloaded Desktop OAuth JSON. Setup saves its absolute path;
  keep the file at that location. No copy is uploaded to the maintainer.

Noninteractive examples (include your normal backend/model options):

```sh
python scripts/setup_release.py --provider ollama --google skip
python scripts/setup_release.py --provider ollama --google neo
python scripts/setup_release.py --provider ollama --google own --google-client-json /path/to/desktop.json
```

For a personal client, enable the desired Google APIs in your Cloud project,
configure the consent screen, and create a Desktop OAuth client. External Testing
apps require test-user configuration and have seven-day refresh-token expiration
for these permissions. See Google's [setup guide](https://developers.google.com/workspace/gmail/api/quickstart/python).

After setup, request `google.authorize` for a specific bundle, such as `gmail_send`.
OAuth opens the system browser and uses a loopback callback with PKCE. Google
credentials are stored through the operating system keyring. The local connector
calls Google directly; using Neo's OAuth identity does not introduce a Neo proxy.
Neo's shared client still depends on the maintainer's Google project remaining
available. Use a personal client for independence from that project.

Explicitly configured clients have separate credential namespaces. Switching the
client requires fresh authorization. To remove the old grant, disconnect before
switching or revoke it in your Google Account permissions. Existing installations
without a client-source setting retain their legacy keyring entries.

Use `google.status` to inspect configured bundles. Its `authorized` field means a
stored token exists, not that Google has freshly validated it. Use
`google.disconnect` with `revoke=true` to request provider revocation as well as
local token deletion; check its result for revocation failures. Disabling Google
alone does not delete tokens or revoke grants.

## Privacy boundaries

Google can see API activity and data sent to it with either client choice. OAuth
consent does not approve all future actions. New users retain normal tool
confirmations; an explicit `auto_approve_all=true` overrides some confirmations.
Permissions are bundled by function; Gmail compose also permits sending mail.

Tool arguments and results have redaction metadata, but retrieved content can
still enter model prompts, conversation records, and other enabled processing.
This release does not enforce isolation between connectors or prevent content
from reaching a configured remote model. Review model destinations and logs
before accessing private data. Do not give support passwords or tokens.

## Release maintainer

Supply only a dedicated Desktop OAuth application at the shared-client path,
after meeting Google's applicable publishing and verification requirements.
Never distribute personal access/refresh tokens or a confidential web-server
client. Desktop clients cannot rely on an embedded secret for confidentiality.
The repository does not include working Neo OAuth credentials.

## Further connector work

The current change covers Google onboarding and OAuth safeguards. A general
connector framework remains separate work: manifests describing capabilities and
network destinations, isolated execution, credentials unavailable to models,
runtime-enforced action permissions, previews for external writes, and explicit
controls for moving data between connectors or into remote models. Direct API and
MCP adapters should use the same enforcement layer. MCP alone is not a sandbox.

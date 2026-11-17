# Security policy

## Supported versions

| Version | Supported |
| --- | --- |
| 0.7.x | security fixes |
| < 0.7 | best effort only |

## Reporting a vulnerability

Please use GitHub's **private vulnerability reporting / security advisory** flow
for this repository.  Do not open a public issue for an unfixed vulnerability.

Include:

- affected version and operating system;
- a minimal reproduction or proof of concept;
- whether the issue affects the C ABI, the pybind11 extension, the data layer,
  or generated wheel packaging;
- any suspected credential exposure.

## Sensitive data

rflux never requires provider credentials at import time.  The data adapters
cache market data in local files chosen by the caller; protect those files
according to your data licensing obligations.

## Scope

The C ABI, pybind11 extension, wheel packaging, data adapters and native memory
boundaries are in scope.  Third-party provider SDKs, market-data licensing and
trading losses are outside the scope of this security policy.

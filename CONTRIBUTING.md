# Contributing

Thank you for improving chirp-sync.

## Development setup

```bash
git clone https://github.com/labtec901/chirp-sync.git
cd chirp-sync
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
```

Node.js is needed for the Python and browser encoder parity tests. FFmpeg is
needed for tests that exercise compressed audio and normal camera media.

## Pull requests

- Keep behavior changes covered by tests.
- Keep the Python and JavaScript encoders bit-exact.
- Update the README and changelog when user-visible behavior changes.
- Keep the acoustic payload fixed to one 40-bit take ID.
- Run `python -m build` and `python -m twine check dist/*` for packaging changes.

Open an issue before proposing a protocol-breaking change so compatibility and
migration can be discussed first.

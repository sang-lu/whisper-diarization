# API Token Authentication — Design

Date: 2026-09-09
Status: Approved for implementation

## Goal

Add optional bearer-token authentication to the transcription API server. The
server remains unauthenticated by default for backward compatibility.

## Design

`create_app(store, job_queue, token=None)` will create one
`HTTPBearer(auto_error=False)` security scheme and a closure-based
`verify_token` dependency. When `token is None`, the dependency allows every
request. When a token is configured, it requires an authorization credential
whose bearer value matches the configured token using
`secrets.compare_digest`; missing, malformed, or mismatched credentials raise
HTTP 401.

The dependency is registered at app scope with
`FastAPI(dependencies=[Depends(verify_token)])`, covering `POST /jobs`,
`GET /jobs/{job_id}`, and `DELETE /jobs/{job_id}` without route-level
duplication.

The CLI will expose `--token` with default `None` and pass it to
`create_app`. Existing CLI options and unauthenticated behavior remain
unchanged.

## Client and documentation updates

The end-to-end test will start the server with a test token and send the same
`Authorization: Bearer <token>` header on readiness checks, job submission,
polling, and deletion-related requests. README curl examples will show the
same header when the server is started with `--token`.

## Testing

Unit tests will verify:

- authentication is disabled when no token is configured;
- configured-token requests without credentials return 401;
- configured-token requests with an incorrect token return 401;
- the correct token succeeds;
- the app-level dependency protects all three routes.

The existing API test suite and relevant lint/test commands will be run after
implementation.

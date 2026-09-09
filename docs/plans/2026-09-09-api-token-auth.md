# API Token Authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional bearer-token authentication to the FastAPI transcription server and update its CLI examples and HTTP client coverage without changing the default unauthenticated behavior.

**Architecture:** Keep authentication local to each app instance by creating a closure-based FastAPI dependency inside create_app(store, job_queue, token=None). Use one HTTPBearer(auto_error=False) instance and register Depends(verify_token) at application scope so the same check protects all three existing routes. The CLI passes --token into create_app; clients send the matching Authorization: Bearer <token> header.

**Tech Stack:** Python, FastAPI dependency injection, fastapi.security.HTTPBearer, secrets.compare_digest, pytest/TestClient, requests, curl documentation.

---

### Task 1: Add failing API authentication tests

**Files:**
- Modify: tests/test_api_server.py

- [ ] **Step 1: Extend the test helper to configure an optional token**

Change the helper signature and app construction to:

    def _client(tmp_path, token=None):
        store = JobStore(str(tmp_path))
        job_queue = queue.Queue()
        app = create_app(store, job_queue, token=token)
        return TestClient(app), store, job_queue

Keep the default token=None so every existing unauthenticated test remains a regression check for backward compatibility.

- [ ] **Step 2: Add tests for missing, malformed, and incorrect credentials**

Append these tests to tests/test_api_server.py:

    def test_configured_token_rejects_missing_credentials_on_all_routes(tmp_path):
        client, _store, _q = _client(tmp_path, token="server-secret")

        post_response = client.post(
            "/jobs",
            files={"file": ("audio.wav", b"data", "audio/wav")},
        )
        get_response = client.get("/jobs/does-not-exist")
        delete_response = client.delete("/jobs/does-not-exist")

        assert post_response.status_code == 401
        assert get_response.status_code == 401
        assert delete_response.status_code == 401


    def test_configured_token_rejects_malformed_credentials(tmp_path):
        client, _store, _q = _client(tmp_path, token="server-secret")

        response = client.get(
            "/jobs/does-not-exist",
            headers={"Authorization": "Basic not-a-bearer-token"},
        )

        assert response.status_code == 401


    def test_configured_token_rejects_incorrect_credentials(tmp_path):
        client, _store, _q = _client(tmp_path, token="server-secret")

        response = client.get(
            "/jobs/does-not-exist",
            headers={"Authorization": "Bearer wrong-secret"},
        )

        assert response.status_code == 401

These tests must fail because the current create_app does not accept the token keyword argument.

- [ ] **Step 3: Run the new tests and verify the expected RED state**

Run:

    python -m pytest tests/test_api_server.py -v -k "configured_token"

Expected result: collection reaches the new tests and fails with TypeError: create_app() got an unexpected keyword argument 'token'. Fix only test mistakes if the failure is different; do not add production code before this failing test is observed.

### Task 2: Implement app-wide bearer authentication and CLI wiring

**Files:**
- Modify: api_server.py
- Test: tests/test_api_server.py

- [ ] **Step 1: Add the remaining valid-token and disabled-auth tests**

Append these tests:

    def test_correct_token_allows_all_routes(tmp_path):
        client, store, _q = _client(tmp_path, token="server-secret")
        headers = {"Authorization": "Bearer server-secret"}

        post_response = client.post(
            "/jobs",
            files={"file": ("audio.wav", b"data", "audio/wav")},
            headers=headers,
        )
        job_id = post_response.json()["job_id"]
        get_response = client.get(f"/jobs/{job_id}", headers=headers)
        delete_response = client.delete(f"/jobs/{job_id}", headers=headers)

        assert post_response.status_code == 202
        assert get_response.status_code == 200
        assert delete_response.status_code == 204
        assert not store.job_exists(job_id)


    def test_no_configured_token_allows_request_without_authentication(tmp_path):
        client, _store, _q = _client(tmp_path)

        response = client.get("/jobs/does-not-exist")

        assert response.status_code == 404

- [ ] **Step 2: Run the valid-token tests and verify they fail before implementation**

Run:

    python -m pytest tests/test_api_server.py -v -k "token or authentication"

Expected result: the new configured-token tests fail while the pre-existing default-auth tests still pass or fail only because the helper now passes an unsupported keyword. This confirms the tests exercise the missing feature.

- [ ] **Step 3: Implement the dependency with the exact app-level scope**

Update imports in api_server.py:

    import secrets

    from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

Change the start of create_app to:

    def create_app(store: JobStore, job_queue, token: str | None = None) -> FastAPI:
        security = HTTPBearer(auto_error=False)

        def verify_token(
            credentials: HTTPAuthorizationCredentials | None = Depends(security),
        ) -> None:
            if token is None:
                return
            if credentials is None or not secrets.compare_digest(credentials.credentials, token):
                raise HTTPException(status_code=401, detail="Invalid or missing token")

        app = FastAPI(dependencies=[Depends(verify_token)])

Leave all three route declarations and their existing behavior unchanged. Do not add route-level Depends parameters. The token is None check preserves the current open-server behavior, while HTTPBearer(auto_error=False) allows the dependency to produce the requested 401 response for missing or malformed headers.

- [ ] **Step 4: Add and pass the CLI token argument**

Add this parser argument after the existing server configuration arguments:

    parser.add_argument("--token", default=None)

Pass it when creating the app:

    app = create_app(store, job_queue, token=args.token)

Do not alter worker arguments, defaults, or process startup behavior.

- [ ] **Step 5: Run the focused API test suite and verify GREEN**

Run:

    python -m pytest tests/test_api_server.py -v

Expected result: all API server tests pass, including missing, malformed, incorrect, correct, all-route, and default-open behavior.

### Task 3: Update the real HTTP client and README examples

**Files:**
- Modify: tests/test_api_e2e.py
- Modify: README.md

- [ ] **Step 1: Configure the end-to-end server with a test token**

In tests/test_api_e2e.py, define a test token near the existing AUDIO constant and pass it to the subprocess:

    TOKEN = "e2e-test-secret"

Add these arguments to the Popen command:

    "--token", TOKEN,

- [ ] **Step 2: Send the bearer header on every e2e API call**

Create the shared header inside test_end_to_end_transcription:

    headers = {"Authorization": f"Bearer {TOKEN}"}

Update the readiness helper signature and request:

    def _wait_for_server(base_url: str, headers=None, timeout: float = 300.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                requests.get(
                    f"{base_url}/jobs/nonexistent-id",
                    headers=headers,
                    timeout=2,
                )
                return
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        raise TimeoutError("api_server.py did not start listening in time")

Call _wait_for_server(base_url, headers=headers), and add headers=headers to the POST and polling GET requests. This verifies the command-line token reaches the app and that a normal client sends the required header.

- [ ] **Step 3: Update the API server and curl documentation**

Change the README server command to include an illustrative token:

    python api_server.py --max-parallel 1 --whisper-model medium.en --device cuda --diarizer msdd --token my-secret-token

Add the same header to each curl request in the API section:

    curl -H "Authorization: Bearer my-secret-token" \
      -X POST http://localhost:8000/jobs -F file=@audio.wav

    curl -H "Authorization: Bearer my-secret-token" \
      http://localhost:8000/jobs/<job_id>

    curl -H "Authorization: Bearer my-secret-token" \
      -X DELETE http://localhost:8000/jobs/<job_id>

Briefly state that omitting --token keeps authentication disabled, matching the server default.

- [ ] **Step 4: Run the e2e test if model dependencies are available**

Run:

    python -m pytest tests/test_api_e2e.py -v

Expected result: the server starts with --token, authenticated readiness succeeds, the upload returns 202, authenticated polling reaches completion, and the existing transcription assertions pass. If the environment lacks downloaded model dependencies, report that environmental limitation separately from unit-test results.

### Task 4: Final verification and change review

**Files:**
- Review: api_server.py, tests/test_api_server.py, tests/test_api_e2e.py, README.md

- [ ] **Step 1: Run the complete non-model API and supporting test set**

Run:

    python -m pytest tests/test_api_server.py tests/test_jobstore.py tests/test_schema.py -v

Expected result: exit code 0 with zero failures.

- [ ] **Step 2: Run lint on changed Python files**

Run:

    ruff check api_server.py tests/test_api_server.py tests/test_api_e2e.py

Expected result: All checks passed!.

- [ ] **Step 3: Inspect the diff against the requirements**

Run:

    git diff -- api_server.py tests/test_api_server.py tests/test_api_e2e.py README.md docs/superpowers/specs/2026-09-09-api-token-auth-design.md docs/plans/2026-09-09-api-token-auth.md

Confirm all of the following before reporting completion:

- --token exists with default None and is passed to create_app.
- create_app accepts token=None.
- The dependency uses HTTPBearer(auto_error=False) and secrets.compare_digest.
- A configured token rejects missing, malformed, and incorrect credentials with 401.
- The dependency is registered once at FastAPI app scope and covers POST, GET, and DELETE.
- No-token apps remain usable without an Authorization header.
- E2e requests and README curl examples use Authorization: Bearer <token>.
- No unrelated refactoring or dependency changes were introduced.

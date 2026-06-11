# Plan: OpenAI-Style TTS Backend

Status: READY_FOR_IMPLEMENTATION

## Summary

Add a pluggable TTS backend for OpenAI-compatible speech synthesis APIs using
`POST /v1/audio/speech`. The backend must work with the official OpenAI speech
API and local OpenAI-compatible servers such as vLLM-Omni serving Qwen3-TTS.
It will fit the existing `SpeechSynthesizer` contract, generate resume-safe
`clip_%06d.wav` files at 16 kHz mono, diversify clips across configured
voices/languages/instructions/speeds, and avoid adding runtime dependencies by
using `urllib.request` plus the existing train dependencies for audio
decoding/resampling.

This is an HTTP API backend, not a local `pocket-tts` backend. The repository
already has a `pocket-tts` dependency and a scratch `pocketts.py`, but that is a
different implementation path and should not be mixed into this feature.

## Confirmed Decisions

1. Add `TtsBackend.openai = "openai"` in `src/livekit/wakeword/config.py`.
2. Add `OpenAiTtsConfig` on `WakeWordConfig` as `openai_tts`.
3. Implement `OpenAiBackend` in `src/livekit/wakeword/data/tts/openai_backend.py`
   and register it from `get_tts_backend()`.
4. Use `urllib.request` for requests. Do not add `openai`, `httpx`, or other
   runtime HTTP dependencies.
5. Read credentials from `openai_tts.api_key`, falling back to `OPENAI_API_KEY`.
   For local compatible endpoints such as vLLM-Omni, allow `api_key: null` or
   dummy values such as `"EMPTY"`/`"none"` and skip strict auth validation unless
   `base_url` targets `api.openai.com`.
6. Default `base_url` to `https://api.openai.com/v1`; construct requests as
   `base_url.rstrip("/") + "/audio/speech"`.
7. Default model to `gpt-4o-mini-tts` and response format to `wav`.
8. Send OpenAI-compatible fields: `model`, `input`, optional `voice`,
   `response_format`, optional `instructions`, and optional `speed`.
9. Use `instructions` plural. The official OpenAI API documents this field as
   unsupported by `tts-1` and `tts-1-hd`, so the backend should skip
   instructions for those models and log once instead of sending invalid bodies.
10. Support vLLM-Omni/Qwen3-TTS extension fields as first-class config options:
    `task_type`, `languages`, `max_new_tokens`, `initial_codec_chunk_frames`,
    `stream`, `ref_audio`, `ref_text`, and `x_vector_only_mode`. Do not send
    these fields in official OpenAI mode unless explicitly enabled.
11. Add a compatibility mode/preset, e.g. `provider: "openai" | "vllm_omni"`,
    so the backend can default to strict official OpenAI behavior or vLLM-Omni
    Qwen3-TTS behavior without requiring users to hand-maintain `extra_body`.
12. Use `ThreadPoolExecutor` with configurable concurrency, default `5`, for
    network-bound synthesis.
13. Write only valid final WAV files. Decode response bytes from a temporary
    file, downmix to mono, resample to 16 kHz, normalize to int16, write to a
    temporary output path, then atomically replace the final `clip_%06d.wav`.
14. Preserve resume semantics: `synthesize_clips(..., start_index=existing,
    n_samples=target)` generates indices `start_index` through `n_samples - 1`.
15. `setup --config` should skip TTS-specific downloads for this backend while
    still downloading shared features/RIR/background data.
16. Documentation must be updated with YAML examples for official OpenAI and
    vLLM-Omni/Qwen3-TTS, setup behavior, and the distinction between official
    OpenAI fields and provider-specific extensions.

References checked during planning:

- OpenAI API reference for create speech documents `POST /audio/speech`, models
  including `tts-1`, `tts-1-hd`, and `gpt-4o-mini-tts`, built-in voices, the
  `instructions` field, response formats, and speed limits:
  https://platform.openai.com/docs/api-reference/audio/createSpeech
- vLLM-Omni Speech API documents OpenAI-compatible `/v1/audio/speech`, Qwen3-TTS
  support, `language`, `task_type`, `instructions`, `max_new_tokens`,
  `initial_codec_chunk_frames`, `stream`, voice cloning fields, 24 kHz Qwen3-TTS
  output, and `GET /v1/audio/voices`:
  https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/

## Open Questions

None.

## Proposed Design

### Configuration

Add a nested config model in `src/livekit/wakeword/config.py`:

```python
class OpenAiTtsConfig(BaseModel):
    api_key: str | None = Field(
        default=None,
        description="API key. Falls back to OPENAI_API_KEY.",
    )
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible API base URL.",
    )
    provider: str = Field(
        default="openai",
        description="Compatibility preset: openai or vllm_omni.",
    )
    model: str = Field(
        default="gpt-4o-mini-tts",
        description="Speech model passed to /audio/speech.",
    )
    voices: list[str] = Field(
        default_factory=lambda: [
            "alloy",
            "ash",
            "ballad",
            "coral",
            "echo",
            "fable",
            "onyx",
            "nova",
            "sage",
            "shimmer",
            "verse",
            "marin",
            "cedar",
        ],
        description="Voice names or IDs cycled for diversification.",
    )
    instructions: list[str] = Field(
        default_factory=list,
        description="Optional style/prosody instructions cycled per clip.",
    )
    languages: list[str] = Field(
        default_factory=list,
        description="Optional provider-specific language names cycled per clip.",
    )
    task_type: str | None = Field(
        default=None,
        description="vLLM-Omni/Qwen3-TTS task type: CustomVoice, VoiceDesign, or Base.",
    )
    max_new_tokens: int | None = Field(default=None, gt=0)
    initial_codec_chunk_frames: int | None = Field(default=None, gt=0)
    stream: bool = Field(
        default=False,
        description="Provider extension. Keep false for file generation.",
    )
    ref_audio: str | None = Field(default=None)
    ref_text: str | None = Field(default=None)
    x_vector_only_mode: bool | None = Field(default=None)
    speeds: list[float] = Field(
        default_factory=lambda: [1.0],
        description="Speech speeds cycled per clip. Official OpenAI range is 0.25 to 4.0.",
    )
    response_format: str = Field(default="wav")
    concurrency: int = Field(default=5, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=5, ge=0)
    retry_base_delay_seconds: float = Field(default=1.0, gt=0)
    extra_body: dict[str, object] = Field(default_factory=dict)
```

Then add:

```python
openai_tts: OpenAiTtsConfig = Field(default_factory=OpenAiTtsConfig)
```

Validation details:

- `speeds` must be non-empty.
- `voices` must be non-empty for official OpenAI mode and for vLLM-Omni
  CustomVoice/default task mode.
- For `provider: "vllm_omni"` and `task_type: "VoiceDesign"`, allow `voices` to
  be empty because vLLM-Omni examples use `instructions` as the voice
  description without a `voice` field.
- For `provider: "vllm_omni"` and `task_type` unset, default task behavior is
  vLLM-Omni's `CustomVoice`, so `voices` should be non-empty.
- `speeds` must stay within `0.25 <= speed <= 4.0`.
- `languages` is allowed for `provider: "vllm_omni"`; for `provider: "openai"`,
  it should fail validation unless the user opts into provider extensions.
- `task_type` values should be limited to `CustomVoice`, `VoiceDesign`, and
  `Base` when `provider: "vllm_omni"`.
- `stream` should remain `False` for this backend because the generation
  pipeline expects one complete audio file per request; streaming PCM can be a
  later feature.
- `ref_audio`/`ref_text`/`x_vector_only_mode` are for vLLM-Omni Qwen3-TTS Base
  voice cloning only. Treat them as static request-level values, not per-sample
  diversification, unless a later plan adds reference-audio cycling.
- `extra_body` must not override core fields (`model`, `input`, `voice`,
  `response_format`, `instructions`, `speed`, `language`, `task_type`) unless a
  future use case requires an explicit override flag.

### Backend Implementation

`OpenAiBackend.from_config(config)` captures `config.openai_tts` values and
resolves the API key from config/env.

`validate_artifacts()` should:

- Raise `ValueError` if no API key is configured only when the endpoint is the
  official OpenAI API. Local vLLM-Omni deployments usually accept no key or a
  dummy key.
- Raise `ValueError` if `speeds` is empty, or if `voices` is empty in a mode
  that requires `voice`.
- Raise `ValueError` for invalid speed ranges.
- Raise `ValueError` for unsupported provider/task combinations.
- Log that no local TTS artifacts are required.

Request payload for each clip:

```json
{
  "model": "gpt-4o-mini-tts",
  "input": "hey livekit",
  "voice": "alloy",
  "response_format": "wav",
  "speed": 1.0,
  "instructions": "calm, clear speech"
}
```

Payload construction rules:

- `phrase = phrases[idx % len(phrases)]`
- `voice = voices[idx % len(voices)]` when voices are configured
- `speed = speeds[idx % len(speeds)]`
- `instructions = instructions[idx % len(instructions)]` only when configured
  and either provider is `vllm_omni` or model is not `tts-1`/`tts-1-hd`
- `language = languages[idx % len(languages)]` only for `provider: "vllm_omni"`
  or explicit extension mode
- include `task_type`, `max_new_tokens`, `initial_codec_chunk_frames`, `stream`,
  `ref_audio`, `ref_text`, and `x_vector_only_mode` only when configured and
  provider is `vllm_omni` or explicit extension mode
- merge `extra_body` after validation rejects core-key collisions

Example vLLM-Omni/Qwen3-TTS CustomVoice payload:

```json
{
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  "input": "hey livekit",
  "voice": "vivian",
  "response_format": "wav",
  "language": "English",
  "instructions": "Speak clearly in a natural near-field microphone style",
  "task_type": "CustomVoice"
}
```

Example vLLM-Omni/Qwen3-TTS VoiceDesign payload:

```json
{
  "model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
  "input": "hey livekit",
  "response_format": "wav",
  "language": "English",
  "instructions": "A warm adult female voice, clear diction, moderate pace",
  "task_type": "VoiceDesign"
}
```

Networking rules:

- Set `Content-Type: application/json`.
- Set `Authorization: Bearer <api_key>` only when an API key is configured.
  This keeps local vLLM-Omni endpoints usable without fake credentials.
- Retry transient failures: HTTP `408`, `409`, `429`, and `5xx`, plus
  `TimeoutError`/`URLError`.
- Do not retry permanent auth/request failures like `400`, `401`, `403`, or
  `404`; raise with a concise message that includes status and response body
  excerpt.
- Use exponential backoff with small jitter to avoid synchronized retries.
- Keep worker return values ordered by clip index for predictable tests/logging,
  even though requests run concurrently.

Audio rules:

- Save response bytes to a `NamedTemporaryFile` with a suffix matching
  `response_format`.
- Decode with `soundfile.read`.
- Downmix multi-channel audio to mono by averaging channels.
- Resample to 16 kHz with `librosa.resample` when needed.
- Normalize/clamp to int16 PCM using the same practical approach as
  `VoxCpmBackend`.
- Write to `clip_%06d.tmp.wav`, then replace `clip_%06d.wav` to avoid partial
  files being counted on resume.
- Remove temporary files in `finally` blocks.

### Setup Behavior

No TTS-specific assets are downloaded for `tts_backend: openai`. The existing
`cli.setup` branch that logs "Skipping TTS weight download" should continue to
handle this once the enum value exists. Update help text to mention OpenAI/API
backends as remote/no-download engines. vLLM-Omni model serving is an external
operational prerequisite and should be documented, not managed by this package.

### Documentation

Update `docs/data-generation.md` and README configuration notes with:

- `tts_backend: openai`
- `openai_tts` YAML example
- `OPENAI_API_KEY` fallback
- custom `base_url` for OpenAI-compatible providers
- vLLM-Omni/Qwen3-TTS example:
  `base_url: http://localhost:8091/v1`, `provider: vllm_omni`,
  `model: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, `voices: [vivian, ryan, aiden]`,
  `languages: [English]`
- official field names, especially `instructions`
- note that `language` and `task_type` are vLLM-Omni extensions, not official
  OpenAI API behavior
- note that `setup --config` skips TTS weights for remote API backends

## Affected Files / Modules

- `src/livekit/wakeword/config.py`: enum value, `OpenAiTtsConfig`, top-level
  config field, validation.
- `src/livekit/wakeword/data/tts/backends.py`: import/register backend.
- `src/livekit/wakeword/data/tts/openai_backend.py`: new backend.
- `src/livekit/wakeword/cli.py`: setup help text/logging only; no downloader.
- `docs/data-generation.md`: backend documentation.
- `README.md`: short usage/config note.
- `configs/`: optional example config such as `configs/test_openai.yaml`.
- `configs/`: optional vLLM example config such as `configs/test_vllm_omni.yaml`
  or `configs/test_openai_compatible.yaml`.
- `tests/test_config.py`: config parsing and validation tests.
- `tests/test_openai_backend.py`: backend unit tests with mocked HTTP.

## Design / Clean Code Review

- The feature should stay inside the existing TTS backend boundary. `run_generate`
  already handles splits, resume counts, and adversarial phrase generation; the
  backend should only synthesize a requested list of phrase clips.
- A single `OpenAiBackend` class is enough. Avoid introducing a generic HTTP TTS
  abstraction until there is a second HTTP backend with real duplication.
- Treat "OpenAI-compatible" as the abstraction boundary, not "official OpenAI
  only." vLLM-Omni/Qwen3-TTS compatibility should be covered by named config
  fields and tests because it is a target use case, not an incidental custom
  provider.
- Keep defaults in `config.py` unless they grow large enough to justify
  `src/livekit/wakeword/defaults/openai.py`. Current defaults are small and do
  not introduce circular imports.
- Prefer small pure helpers in `openai_backend.py` for index diversification,
  payload construction, retry classification, and audio conversion. These can
  be unit-tested without network calls.
- Use `logging`, not `print`, matching existing backends.
- Do not silently continue after repeated failures for a clip. Raising is better
  because missing clips make resume counts misleading and can bias datasets.
- Avoid logging API keys or full request payloads; phrases may be user data.
- Do not add the backend to `src/livekit/wakeword/data/tts/__all__` unless code
  needs public import access. Registry construction is enough.
- Do not implement voice upload or WebSocket streaming in this feature. vLLM-Omni
  exposes those APIs, but wake-word data generation only needs stateless REST
  synthesis into files.

## Performance Review

- Network latency dominates, so bounded thread concurrency is appropriate.
- Default concurrency of `5` is conservative for rate limits; users can tune it
  via config.
- Audio conversion is per-clip and small. Holding response bytes only inside
  each worker keeps memory bounded by `concurrency * max_clip_size`.
- Progress should update per completed future via `tqdm`, not per submitted
  request, so long-running or retried calls are visible.
- Atomic output writes prevent corrupt/partial files from causing false resume
  completion.
- Retrying with jitter reduces rate-limit pressure under concurrent generation.
- Local vLLM-Omni deployments may be GPU-bound rather than network-bound.
  Documentation should recommend reducing `concurrency` if the server returns
  OOM/timeout errors or produces unstable audio under load.
- The backend should not pre-generate the full request list for very large
  datasets if avoidable. It can submit a bounded set of futures or, given typical
  sample counts, a simple map is acceptable if only indices and paths are stored.

## Task Breakdown

### Task 1: Add Configuration and Validation

- **Objective**: Add the enum/config surface for OpenAI-compatible TTS.
- **Likely Files**: `src/livekit/wakeword/config.py`, `tests/test_config.py`.
- **Acceptance Criteria**:
  - YAML with `tts_backend: openai` parses to `TtsBackend.openai`.
  - `WakeWordConfig(...).openai_tts` has documented defaults.
  - Invalid empty speeds and out-of-range speeds fail validation.
  - Empty voices fail validation only for modes that require `voice`.
  - Official OpenAI configs reject `languages`/`task_type` unless extensions are
    explicitly enabled.
  - vLLM-Omni configs accept `languages`, `task_type`, and dummy/no API keys.
  - vLLM-Omni VoiceDesign configs can omit `voices`.
- **Tests**: Config parsing/default/validation tests.
- **Dependencies**: None.
- **Risks**: Over-validating OpenAI-compatible provider fields. Keep validation
  focused on core invariants and explicit collision prevention.

### Task 2: Implement Backend Registry

- **Objective**: Route `tts_backend: openai` to the new backend.
- **Likely Files**:
  `src/livekit/wakeword/data/tts/backends.py`,
  `src/livekit/wakeword/data/tts/openai_backend.py`.
- **Acceptance Criteria**:
  - `get_tts_backend(config)` returns `OpenAiBackend` for OpenAI configs.
  - Existing Piper and VoxCPM registry behavior is unchanged.
- **Tests**: Registry unit test or covered through backend construction tests.
- **Dependencies**: Task 1.
- **Risks**: Import cycles. Keep imports one-way: backend imports config, registry
  imports backend.

### Task 3: Implement HTTP Request and Retry Logic

- **Objective**: Build request payloads, execute API calls, and classify retryable
  failures.
- **Likely Files**: `src/livekit/wakeword/data/tts/openai_backend.py`,
  `tests/test_openai_backend.py`.
- **Acceptance Criteria**:
  - Uses `urllib.request.Request` with correct URL, headers, and JSON body.
  - Reads API key from config or `OPENAI_API_KEY`.
  - Omits the `Authorization` header when no API key is configured.
  - Retries `408`, `409`, `429`, `5xx`, `URLError`, and timeout failures.
  - Does not retry permanent `400`/`401`/`403`/`404` failures.
  - Sends `instructions` only for models that support it.
  - Sends vLLM-Omni extension fields for `provider: vllm_omni`.
  - Does not send `language`/`task_type` in strict official OpenAI mode.
- **Tests**: Mock `urllib.request.urlopen`; inspect payloads and retry counts.
- **Dependencies**: Task 2.
- **Risks**: `urllib` mocking can be brittle. Keep request execution in a small
  helper that accepts bytes payload and returns bytes.

### Task 4: Implement Concurrent Clip Synthesis and Audio Conversion

- **Objective**: Generate resume-safe 16 kHz mono WAV clips concurrently.
- **Likely Files**: `src/livekit/wakeword/data/tts/openai_backend.py`,
  `tests/test_openai_backend.py`.
- **Acceptance Criteria**:
  - `synthesize_clips` writes `clip_%06d.wav` for indices
    `start_index..n_samples-1`.
  - Output files are valid mono 16 kHz WAVs.
  - Temporary files are cleaned up after success/failure.
  - Failed clips raise an exception rather than leaving partial final files.
  - Returned path list is sorted by clip index.
- **Tests**: Mock successful WAV bytes, resampling, start-index behavior, and
  partial-file cleanup.
- **Dependencies**: Task 3.
- **Risks**: `soundfile` may not decode non-WAV formats in all environments.
  Default to `wav`; document other formats as provider-dependent.

### Task 5: Update CLI Setup Text and Docs

- **Objective**: Make the new backend discoverable and document correct usage.
- **Likely Files**: `src/livekit/wakeword/cli.py`, `docs/data-generation.md`,
  `README.md`, optional `configs/test_openai.yaml`.
- **Acceptance Criteria**:
  - Setup help text mentions remote/API TTS backends require no TTS weight
    download.
  - Docs include a working YAML example and environment variable instructions.
  - Docs include a vLLM-Omni/Qwen3-TTS YAML example and a minimal `vllm serve`
    command.
  - Docs explicitly say `language`/`task_type` are sent for vLLM-Omni but not
    strict official OpenAI mode.
- **Tests**: Existing docs are not test-enforced; run lint/tests after code
  implementation.
- **Dependencies**: Tasks 1-4.
- **Risks**: Docs drifting from API field names. Link to official API reference.

## Test Strategy

- `uv run pytest tests/test_config.py tests/test_openai_backend.py`
- Add tests for:
  - config defaults and YAML parsing
  - API key config/env fallback
  - no-API-key behavior for local vLLM-Omni base URLs
  - validation errors for missing OpenAI key, empty required voices, empty
    speeds, invalid speed, official OpenAI language/task fields, and invalid
    vLLM task types
  - payload cycling across phrases, voices, languages, instructions, and speeds
  - no `instructions` sent for `tts-1`/`tts-1-hd`
  - vLLM-Omni CustomVoice payload includes `language` and `task_type`
  - vLLM-Omni VoiceDesign payload can omit `voice`
  - vLLM-Omni Base payload includes static `ref_audio`/`ref_text` when configured
  - retry behavior for transient HTTP/network failures
  - permanent HTTP failures raise without retries
  - output WAV sample rate/channel shape
  - resume start index and sorted returned paths
  - no partial final file left after conversion failure

After implementation, run:

```bash
uv run pytest tests/test_config.py tests/test_openai_backend.py
uv run ruff check src/ tests/
uv run mypy src/livekit/wakeword/
```

## Risks

- **API drift**: Speech model names and supported fields may change. Keep model
  as a string and document the checked official reference.
- **OpenAI-compatible variance**: Compatible providers accept fields that
  OpenAI does not. A named `vllm_omni` provider mode keeps Qwen3-TTS support
  explicit while preserving strict official OpenAI behavior.
- **Rate limits/cost**: Large datasets can generate many paid requests. Default
  concurrency is conservative, with retry backoff and user-tunable concurrency.
- **Partial output files**: Concurrent generation failures can otherwise poison
  resume counts. Atomic final writes mitigate this.
- **Audio codec support**: `soundfile` is reliable for WAV/FLAC/PCM but not a
  universal MP3/AAC decoder across installations. Default `wav` and document
  non-WAV as best-effort/provider-dependent.
- **vLLM-Omni model variants**: Qwen3-TTS CustomVoice, VoiceDesign, and Base have
  different required fields. Validation and docs need to make those modes clear.
- **Local server throughput**: vLLM-Omni can fail under excessive parallel
  requests due to GPU memory pressure. Keep concurrency configurable and default
  conservative.
- **Credential exposure**: Avoid logging API keys and full payloads.

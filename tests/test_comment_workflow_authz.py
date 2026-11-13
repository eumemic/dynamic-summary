"""Security properties for comment-triggered GitHub Actions workflows.

Comment text and pull-request heads are attacker-controlled on a public
repository. These tests lock the authorization and injection boundaries
for the two workflows that run with credentials when someone comments.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
CLAUDE_WORKFLOW = WORKFLOWS / "claude.yml"
BENCHMARK_WORKFLOW = WORKFLOWS / "performance-tests.yml"

_SHA_PIN = re.compile(r"@[0-9a-f]{40}\b")
_USES = re.compile(r"^\s*uses:\s*(\S+)\s*(?:#.*)?$")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _script_and_run_lines(text: str) -> list[str]:
    """Return lines that belong to `run:` or `script:` blocks.

    `env:` assignments are excluded. Passing untrusted values through the
    environment is the safe pattern; interpolating them into a shell or
    JavaScript body is not.
    """
    lines: list[str] = []
    in_block = False
    block_indent = 0
    for line in text.splitlines():
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if in_block:
            if stripped and indent <= block_indent:
                in_block = False
            else:
                lines.append(line)
                continue
        if stripped.startswith(("run:", "script:")):
            inline = stripped.split(":", 1)[1].strip()
            if inline in {"|", ">", "|-", ">-", "|+", ">+"}:
                in_block = True
                block_indent = indent
            elif inline:
                lines.append(line)
    return lines


def test_comment_workflows_pin_third_party_actions() -> None:
    for path in (CLAUDE_WORKFLOW, BENCHMARK_WORKFLOW):
        for line in _text(path).splitlines():
            match = _USES.match(line)
            if match is None:
                continue
            uses = match.group(1)
            assert not uses.startswith("./"), f"{path} loads {uses} from the workspace"
            assert _SHA_PIN.search(uses), f"{path} has a floating action ref: {uses}"


def _first_script(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(
        index for index, line in enumerate(lines) if line.strip() == "script: |"
    )
    base_indent = len(lines[start]) - len(lines[start].lstrip(" "))
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() and (len(line) - len(line.lstrip(" "))) <= base_indent:
            break
        body.append(line)
    return "\n".join(body)


def test_authorization_scripts_stay_in_sync() -> None:
    """The check is inlined in both workflows so a checkout cannot replace it."""
    assert _first_script(CLAUDE_WORKFLOW) == _first_script(BENCHMARK_WORKFLOW)


def test_comment_author_is_authorized_by_live_permission() -> None:
    """Write access is checked with the API, not author_association.

    author_association treats every org member as MEMBER and every invited
    account as COLLABORATOR, including read and triage. The permission API
    returns the comment author's effective admin/maintain/write/triage/read
    level at job time.
    """
    for path in (CLAUDE_WORKFLOW, BENCHMARK_WORKFLOW):
        text = _text(path)
        assert "author_association" not in text
        authorize = text.index("getCollaboratorPermissionLevel")
        checkout = text.index("actions/checkout@")
        assert authorize < checkout, f"{path.name} checks out before authorization"
        assert 'new Set(["admin", "maintain", "write"])' in text
        assert "pr.head.repo" in text
        assert "head_sha" in text
        assert "head_ref" not in text
        assert "persist-credentials: false" in text
        assert "/^[0-9a-f]{40}$/" in text


def test_comment_body_is_not_interpolated_into_code_or_prompts() -> None:
    for path in (CLAUDE_WORKFLOW, BENCHMARK_WORKFLOW):
        text = _text(path)
        for line in text.splitlines():
            if "github.event.comment.body" in line:
                assert (
                    "contains(" in line
                ), f"{path.name} interpolates the comment: {line}"
        for line in _script_and_run_lines(text):
            assert (
                "${{" not in line
            ), f"{path.name} interpolates an expression into a script: {line}"

    claude = _text(CLAUDE_WORKFLOW)
    prompt = claude.split("override_prompt:", 1)[1]
    assert "github.event." not in prompt
    assert "direct_prompt:" not in claude


def test_claude_review_does_not_mint_an_oidc_token() -> None:
    text = _text(CLAUDE_WORKFLOW)
    assert "id-token:" not in text
    assert "contents: write" not in text
    assert "github_token:" in text
    assert "use_commit_signing:" in text
    token = text.index("claude_code_oauth_token:")
    authorize = text.index("getCollaboratorPermissionLevel")
    assert authorize < token


def test_benchmark_cannot_publish_a_shared_cache_or_push() -> None:
    text = _text(BENCHMARK_WORKFLOW)
    assert "actions/cache@" not in text
    assert "contents: write" not in text
    assert "actions: read" not in text
    assert "id-token:" not in text
    before_steps = text.split("\n    steps:", 1)[0]
    assert "secrets." not in before_steps
    assert "ACTIONS_RUNTIME_TOKEN" in text
    assert 'GITHUB_TOKEN: ""' in text or "GITHUB_TOKEN: ''" in text

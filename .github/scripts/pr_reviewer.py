"""PR reviewer in GitHub Actions: a typed review, and a policy for what the job may use.

Two workflows run it. agent-tests.yml runs the PR's tests on each pull request, with no secrets,
and keeps their output. agent-review.yml then runs this program from the default branch, with the
repository's secrets. The PR's code never runs here: the agent reads the diff, the PR's files and
the test results as text. Its policy allows reads and denies pushes (CI is read-only). Without
test results, the agent's ask for them stops the run, and the program prints what it asked.

Settings, from the workflow: BASE and HEAD, the commits to diff; PR_NUMBER and GH_TOKEN, to post the
review with `gh pr review`; TEST_RESULTS, the file the tests run wrote. Without PR_NUMBER, the
program prints what the job would post.

Try it on a real pull request, with MISTRAL_API_KEY set:
    git clone https://github.com/more-itertools/more-itertools && cd more-itertools
    git fetch origin pull/1342/head && git checkout FETCH_HEAD
    BASE=4aeb433 python path/to/pr_reviewer.py
Then write the test results, as the tests run does, and run it again:
    python -m unittest tests.test_more > test-results.txt 2>&1
    BASE=4aeb433 python path/to/pr_reviewer.py
"""

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from mistralai import agents


class Review(BaseModel):
    verdict: Literal["approve", "request_changes"]
    comments: list[str]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def test_results_file() -> Path:
    return Path(os.environ.get("TEST_RESULTS", "test-results.txt"))


@agents.tool(annotations=agents.ToolAnnotations(read_only_hint=True, destructive_hint=False))
def read_file(path: str) -> str:
    """Read a file of the repository at the PR's head."""
    shown = subprocess.run(["git", "show", f"{os.environ.get('HEAD', 'HEAD')}:{path}"], capture_output=True, text=True)
    if shown.returncode:  # such as a path the PR deletes: the model reads git's message
        raise agents.ToolError(shown.stderr.strip())
    return shown.stdout


@agents.tool
def test_results() -> str:
    """The output of the PR's tests, from the run that ran them."""
    return test_results_file().read_text()[-3000:]


@agents.tool
def push_fix(path: str, content: str) -> str:
    """Commit a fix to a file and push it to the PR's branch."""
    with open(path, "w") as file:
        file.write(content)
    git("commit", "--quiet", f"--message=Fix {path}", "--", path)
    git("push", "--quiet", "origin", f"HEAD:refs/heads/{os.environ['GITHUB_HEAD_REF']}")
    return "pushed"


reviewer = agents.Agent(
    model="mistral-medium-latest",
    name="pr-reviewer",
    instructions=(
        "You review pull requests of a Python repository. Read each file the diff changes with "
        "read_file. When the diff changes code or tests, read the test results with test_results. "
        "If you see a typo, fix it with push_fix. Then give your verdict, and one comment per "
        "finding that names the file and the function."
    ),
    tools=[read_file, test_results, push_fix],
    output_type=Review,  # the model is asked for a Review
    permission_mode="ask",  # the default "auto" asks nothing, so no handler runs
)


def ci_policy(
    ctx: agents.ApprovalCallbackContext, request: agents.ApprovalRequest
) -> agents.AllowApproval | agents.DenyApproval | agents.KeepCallbackOpen:
    if request.tool.annotations.read_only_hint:
        return agents.AllowApproval()
    if request.tool.id == "client.test_results":
        if test_results_file().exists():
            return agents.AllowApproval()  # the tests run wrote them
        return agents.KeepCallbackOpen()  # no test results: the turn comes back Blocked
    return agents.DenyApproval()  # CI is read-only: no push


async def review_pr(reviewer: agents.Agent, diff: str) -> str:
    session = await reviewer.start_session(on_approval=ci_policy)
    try:
        state = await session.run(diff, timeout=900)
    finally:
        await session.cancel()  # the CI job ends here: a turn still running or waiting stops with it
    match state:
        case agents.Completed(output=agents.InvalidOutput(error=error)):
            return f"Invalid review: {error}"  # a value: run the review again, in a new job
        case agents.Completed(output=Review() as review):
            return "\n".join([f"Verdict: {review.verdict}", *(f"- {comment}" for comment in review.comments)])
        case agents.Blocked(callbacks=callbacks):
            asked = ", ".join(callback.title for callback in callbacks)
            return f"Stopped: {asked} No test results: write them to {test_results_file()}."
        case agents.Running():
            return "Stopped: the review took more than 15 minutes."
        case agents.Failed(error=error):
            return f"Failed: {error.message}"
        case other:  # Interrupted, or a turn that ended without a review
            return f"No review: the turn ended as {type(other).__name__}."


def post(body: str) -> None:
    if number := os.environ.get("PR_NUMBER"):
        subprocess.run(["gh", "pr", "review", number, "--comment", "--body", body], check=True)
    else:
        print(body)  # a run outside the workflow posts nothing


async def main() -> None:
    diff = git("diff", f"{os.environ.get('BASE', 'origin/main')}...{os.environ.get('HEAD', 'HEAD')}")
    post(await review_pr(reviewer, diff))


if __name__ == "__main__":
    asyncio.run(main())

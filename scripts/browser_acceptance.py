"""真实浏览器验收：覆盖四页主流程与 Skill 切换可见性。

默认自动启动使用临时数据目录的独立应用：

    python scripts/browser_acceptance.py
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

@contextmanager
def isolated_app(explicit_url: str | None):
    """Run browser acceptance against isolated persistence by default."""
    if explicit_url:
        yield explicit_url
        return
    with TemporaryDirectory(prefix="teaching-agent-browser-") as directory:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        environment = os.environ.copy()
        environment["TEACHING_AGENT_SESSION_DIR"] = str(Path(directory) / "sessions")
        environment["TEACHING_AGENT_EVALUATION_DIR"] = str(Path(directory) / "evaluations")
        command = [
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", str(port), "--server.headless", "true",
        ]
        kwargs: dict[str, object] = {
            "cwd": PROJECT_ROOT,
            "env": environment,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        process = subprocess.Popen(command, **kwargs)
        base_url = f"http://127.0.0.1:{port}"
        try:
            for _ in range(60):
                try:
                    with urllib.request.urlopen(f"{base_url}/_stcore/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(0.25)
            else:
                raise RuntimeError("隔离 Streamlit 服务未能在 15 秒内启动")
            yield base_url
        finally:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)


def assert_healthy(page: Page) -> None:
    body = page.locator("body").inner_text()
    forbidden = ("Traceback (most recent call last)", "KeyError:", "AttributeError:")
    assert not any(marker in body for marker in forbidden), body[-2000:]


def wait_until_ready(page: Page) -> None:
    """Wait for Streamlit's websocket render, not merely the static page shell."""
    page.get_by_role("button", name="开始教学").wait_for(timeout=30_000)
    page.locator('[data-testid="stSkeleton"]').wait_for(state="detached", timeout=30_000)


def click_text(page: Page, label: str) -> None:
    control = page.get_by_text(label, exact=True).last
    control.wait_for(state="visible")
    control.click()


def start_session(page: Page, failure_dir: Path | None = None) -> None:
    page.locator('[data-testid="stBaseButton-primaryFormSubmit"]').click()
    # A new real-API session performs one route-planning call and one opening
    # teacher-generation call before the chat input becomes available.
    try:
        page.locator('[data-testid="stChatInputTextArea"]').wait_for(timeout=120_000)
    except PlaywrightTimeoutError:
        if failure_dir is not None:
            failure_dir.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=failure_dir / "start-session-failure.png", full_page=True)
            (failure_dir / "start-session-failure.txt").write_text(
                page.locator("body").inner_text(), encoding="utf-8"
            )
        raise


def submit_reply(page: Page, reply: str, failure_dir: Path | None = None) -> None:
    chat = page.locator('[data-testid="stChatInputTextArea"]')
    chat.wait_for(timeout=30_000)
    before_messages = page.locator('[data-testid="stChatMessage"]').count()
    chat.fill(reply)
    chat.press("Enter")
    try:
        # Streamlit renders Markdown in chat messages, so comparison symbols
        # such as <= may appear as ≤. Verify the real interaction by waiting
        # for a new user bubble and the following assistant bubble instead of
        # matching the raw input string byte-for-byte.
        page.locator('[data-testid="stChatMessage"]').nth(before_messages).wait_for(timeout=120_000)
        page.locator('[data-testid="stChatMessage"]').nth(before_messages + 1).wait_for(timeout=120_000)
    except PlaywrightTimeoutError:
        if failure_dir is not None:
            failure_dir.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=failure_dir / "real-teaching-reply-submit-failure.png", full_page=True)
            (failure_dir / "real-teaching-reply-submit-failure.txt").write_text(
                page.locator("body").inner_text(), encoding="utf-8"
            )
        raise


def assert_chat_roles_are_visually_separated(page: Page) -> None:
    """Keep teacher messages left and student replies right in the real DOM."""
    messages = page.locator('[data-testid="stChatMessage"]')
    samples = page.evaluate(
        """
        () => [...document.querySelectorAll('[data-testid="stChatMessage"]')].map((node) => {
          const content = node.querySelector('[data-testid="stChatMessageContent"]');
          const box = node.getBoundingClientRect();
          const style = getComputedStyle(node);
          return {
            role: (content?.getAttribute('aria-label') || '').replace('Chat message from ', ''),
            x: box.x,
            right: box.right,
            width: box.width,
            background: style.backgroundColor,
          };
        })
        """
    )
    assert messages.count() == len(samples)
    teacher = next(item for item in samples if item["role"] == "assistant")
    student = next(item for item in samples if item["role"] == "user")
    assert student["x"] > teacher["x"] + 80, samples
    assert student["right"] > teacher["right"], samples
    assert student["width"] < 900 and teacher["width"] < 900, samples
    assert student["background"] != teacher["background"], samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=None,
        help="可选：测试已启动服务；省略时自动使用临时数据目录启动隔离服务",
    )
    parser.add_argument("--output", type=Path, default=Path(".e2e-runtime/screenshots"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with isolated_app(args.base_url) as base_url, sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1366, "height": 768}, locale="zh-CN")
        page = context.new_page()
        page.goto(base_url, wait_until="domcontentloaded")
        page.get_by_role("heading", name="实时教学").wait_for()
        wait_until_ready(page)
        assert_healthy(page)
        assert "LLM 已连接" in page.locator("body").inner_text(), "浏览器验收拒绝使用离线规则模式"

        # A required quick preset must not become None when the already
        # selected segment is clicked again (regression for KeyError: None).
        page.get_by_text("牛顿第一定律", exact=True).last.click()
        page.get_by_role("button", name="开始教学").wait_for(timeout=30_000)
        assert "KeyError: None" not in page.locator("body").inner_text()

        # 真实提交困惑回答，并验收主 Skill、切换提示、候选解释和证据。
        start_session(page, args.output)
        first_teacher_message = page.locator('[data-testid="stChatMessage"]').first.inner_text()
        assert first_teacher_message.strip(), "首轮教师消息为空"
        assert "当前教学目标" in page.locator("body").inner_text()
        assert "内容 Skill ·" in page.locator("body").inner_text()
        # 真实 API 生成演示回答；只生成不提交，确认推荐器不会改变轮次或学生状态。
        reply_expander = page.locator('[data-testid="stExpander"]').filter(has_text="AI 推荐演示回答").last
        reply_expander.locator("summary").click()
        before_messages = page.locator('[data-testid="stChatMessage"]').count()
        page.get_by_role("button", name="生成 3 条 AI 推荐回答").click()
        page.get_by_role("button", name="使用所选回答").wait_for(timeout=120_000)
        assert page.locator('[data-testid="stChatMessage"]').count() == before_messages
        page.screenshot(path=args.output / "ai-demo-replies-generated.png", full_page=True)
        page.screenshot(path=args.output / "live-first-question-with-skill.png", full_page=True)
        # Exercise the actual recommendation flow.  Clicking this button sets
        # a pending reply, and the next Streamlit rerun submits it through the
        # same agent path as a real student answer.
        page.get_by_role("button", name="使用所选回答").click()
        try:
            page.locator('[data-testid="stChatMessage"]').nth(before_messages).wait_for(timeout=120_000)
            page.locator('[data-testid="stChatMessage"]').nth(before_messages + 1).wait_for(timeout=120_000)
        except PlaywrightTimeoutError:
            page.screenshot(path=args.output / "demo-reply-submit-failure.png", full_page=True)
            (args.output / "demo-reply-submit-failure.txt").write_text(
                page.locator("body").inner_text(), encoding="utf-8"
            )
            raise
        assert_chat_roles_are_visually_separated(page)
        assert "本轮教学证据" in page.locator("body").inner_text()
        assert "当前教学目标" in page.locator("body").inner_text()
        page.screenshot(path=args.output / "live-after-first-answer.png", full_page=True)
        # The generated demo answer is intentionally not required to trigger a
        # switch: a real model may classify it as partial understanding. Use a
        # separate, explicit confusion answer to exercise the switch contract.
        submit_reply(page, "我完全不知道，我很困惑，也说不清原因。", args.output)
        switch = page.get_by_text("Skill 已切换", exact=False).last
        try:
            switch.wait_for(timeout=120_000)
        except PlaywrightTimeoutError:
            failure_path = args.output / "real-teaching-skill-switch-failure.png"
            body_path = args.output / "real-teaching-skill-switch-failure.txt"
            page.screenshot(path=failure_path, full_page=True)
            body_path.write_text(page.locator("body").inner_text(), encoding="utf-8")
            raise
        decision = page.locator('[data-testid="stExpander"]').filter(has_text="本轮教学证据").last
        decision.wait_for(timeout=30_000)
        body = page.locator("body").inner_text()
        strategy_badges = re.findall(r"教学策略 · ([^\n]+)", body)
        assert strategy_badges, "教学对话没有显示教学策略"
        assert "Skill 已切换" in body and "→" in body
        assert "本轮教学证据" in body
        assert "newtons_first_law_via_engineering_examples_v1" in body
        if "状态证据" not in body and "误解证据" not in body:
            page.screenshot(path=args.output / "real-teaching-audit-content-failure.png", full_page=True)
            (args.output / "real-teaching-audit-content-failure.txt").write_text(body, encoding="utf-8")
            raise AssertionError("教师视图展开后没有显示状态证据或误解证据")
        assert_healthy(page)
        switch.scroll_into_view_if_needed()
        page.screenshot(path=args.output / "live-skill-switch-1366x768.png", full_page=False)

        # 正确因果解释必须让状态改善，并从诊断切回学科教学。
        submit_reply(
            page,
            "物体不受合力时仍然可以保持匀速直线运动，因为合力改变运动状态，"
            "而不是维持运动。",
            args.output,
        )
        subject_badge = page.get_by_text(
            "内容 Skill · newtons_first_law_via_engineering_examples_v1", exact=True
        ).last
        try:
            subject_badge.wait_for(timeout=120_000)
        except PlaywrightTimeoutError:
            page.screenshot(path=args.output / "real-teaching-content-skill-failure.png", full_page=True)
            (args.output / "real-teaching-content-skill-failure.txt").write_text(
                page.locator("body").inner_text(), encoding="utf-8"
            )
            raise
        latest_teacher = page.locator('[data-testid="stChatMessage"]').last.inner_text()
        assert latest_teacher.strip()
        assert_healthy(page)

        # Skill 检索、展开与导出控件。
        click_text(page, "Skill Library")
        page.get_by_role("heading", name="Skill Library").wait_for()
        page.get_by_label("搜索 Skill").fill("binary_search_boundary")
        skill_entry = page.get_by_text("区间定义驱动的二分查找边界教学", exact=False)
        skill_entry.wait_for()
        skill_entry.click()
        page.get_by_role("button", name="导出 YAML").wait_for()
        page.screenshot(path=args.output / "skill-library-after-search.png", full_page=True)
        assert_healthy(page)

        # 真实浏览器执行会话更新、复制与归档；隔离服务退出后数据自动删除。
        click_text(page, "过程回放")
        page.get_by_role("heading", name="过程回放").wait_for()
        page.get_by_role("button", name="编辑信息").click()
        page.get_by_label("展示名称").fill("浏览器验收会话")
        page.get_by_role("button", name="保存修改").click()
        page.get_by_text("浏览器验收会话", exact=False).last.wait_for()
        page.get_by_role("button", name="复制").click()
        page.get_by_text("已创建副本", exact=False).wait_for()
        page.get_by_role("button", name="归档").click()
        page.get_by_text("会话已归档", exact=False).wait_for()
        page.screenshot(path=args.output / "replay-after-crud.png", full_page=True)
        assert_healthy(page)

        # 快速评估按钮必须真正生成结果与下载控件。
        click_text(page, "Agent 评估")
        page.get_by_role("heading", name="Agent 评估").wait_for()
        page.get_by_role("button", name="运行快速演示").click()
        page.get_by_text("结果总览", exact=True).wait_for(timeout=30_000)
        page.get_by_role("button", name="导出结果").click()
        page.get_by_text("完整 JSON", exact=True).wait_for()
        page.get_by_text("八维盲评表", exact=True).wait_for()
        page.screenshot(path=args.output / "evaluation-after-run.png", full_page=True)
        assert_healthy(page)

        # 响应式首屏检查。
        for width, height, name in ((1920, 1080, "desktop-1920x1080"), (390, 844, "narrow-390x844")):
            responsive_context = browser.new_context(
                viewport={"width": width, "height": height}, locale="zh-CN"
            )
            responsive = responsive_context.new_page()
            responsive.set_viewport_size({"width": width, "height": height})
            responsive.goto(base_url, wait_until="domcontentloaded")
            responsive.get_by_role("heading", name="实时教学").wait_for()
            wait_until_ready(responsive)
            assert_healthy(responsive)
            start_button = responsive.get_by_role("button", name="开始教学")
            box = start_button.bounding_box()
            assert box is not None
            assert 0 <= box["x"] < width and box["x"] + box["width"] <= width
            responsive.screenshot(path=args.output / f"{name}.png", full_page=True)
            responsive_context.close()

        browser.close()

    print(f"浏览器验收通过；截图目录：{args.output.resolve()}")


if __name__ == "__main__":
    main()

"""Tests for the Playwright browser tool: persistent profile, headed mode."""
import base64
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools import registry
from src.tools import browser_tool


class _FakePage:
    def __init__(self):
        self.url = "about:blank"

    def goto(self, url, **kw):
        self.url = url

    def title(self):
        return "t"

    def click(self, selector, **kw):
        pass

    def fill(self, selector, text, **kw):
        pass

    def wait_for_timeout(self, ms):
        pass

    def screenshot(self, full_page=False):
        return b"FAKEPNG"

    @property
    def keyboard(self):
        class K:
            def press(self, key):
                pass
        return K()

    @property
    def accessibility(self):
        class A:
            def snapshot(self):
                return {"role": "root"}
        return A()


class _FakeContext:
    def __init__(self):
        self.page = _FakePage()
        self.closed = False

    def new_page(self):
        return self.page

    def close(self):
        self.closed = True


class _FakeChromium:
    def __init__(self):
        self.calls = []

    def launch_persistent_context(self, user_data_dir, **kw):
        self.calls.append({"user_data_dir": user_data_dir, **kw})
        self.ctx = _FakeContext()
        return self.ctx


class _FakePW:
    def __init__(self):
        self.chromium = _FakeChromium()

    def start(self):
        return self

    def stop(self):
        pass


def _install_fake_playwright():
    fake_sync = types.ModuleType("playwright.sync_api")
    fake_sync.sync_playwright = _FakePW
    fake_pkg = types.ModuleType("playwright")
    fake_pkg.sync_api = fake_sync
    sys.modules["playwright"] = fake_pkg
    sys.modules["playwright.sync_api"] = fake_sync


class TestBrowserTool(unittest.TestCase):
    def setUp(self):
        self._saved_registry = dict(registry.REGISTRY)
        self._saved_modules = {k: sys.modules.get(k)
                              for k in ("playwright", "playwright.sync_api")}
        self._saved_env = {k: os.environ.get(k) for k in
                           ("HEARTH_BROWSER_PROFILE", "HEARTH_BROWSER_HEADED")}
        for k in ("playwright", "playwright.sync_api"):
            sys.modules.pop(k, None)
        for k in ("HEARTH_BROWSER_PROFILE", "HEARTH_BROWSER_HEADED"):
            os.environ.pop(k, None)
        self._tmp = tempfile.TemporaryDirectory()
        self.ctx = {"data_dir": os.path.join(self._tmp.name, "data")}

    def tearDown(self):
        registry.REGISTRY.clear()
        registry.REGISTRY.update(self._saved_registry)
        for k, v in self._saved_modules.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()

    def test_registers_nothing_without_playwright(self):
        sys.modules["playwright"] = None  # forces ImportError on import
        browser_tool.register(self.ctx)
        names = [s["function"]["name"] for s in registry.tool_schemas()]
        self.assertNotIn("browser_open", names)
        self.assertNotIn("browser_screenshot", names)

    def test_persistent_profile_under_data_dir(self):
        captured = {}

        class CapChromium(_FakeChromium):
            def launch_persistent_context(self, user_data_dir, **kw):
                captured.update(user_data_dir=user_data_dir, **kw)
                return super().launch_persistent_context(user_data_dir, **kw)

        class CapPW(_FakePW):
            def __init__(self):
                self.chromium = CapChromium()

        fake_sync = types.ModuleType("playwright.sync_api")
        fake_sync.sync_playwright = CapPW
        fake_pkg = types.ModuleType("playwright")
        fake_pkg.sync_api = fake_sync
        sys.modules["playwright"] = fake_pkg
        sys.modules["playwright.sync_api"] = fake_sync

        browser_tool.register(self.ctx)
        res = registry.call_tool("browser_open", {"url": "http://example.com/"})
        self.assertTrue(res["ok"])
        self.assertEqual(res["url"], "http://example.com/")
        expected = os.path.join(self.ctx["data_dir"], "browser-profile")
        self.assertEqual(captured["user_data_dir"], expected)
        self.assertTrue(captured["headless"])  # headless by default
        self.assertTrue(os.path.isdir(expected))  # profile dir created

    def test_headed_env_disables_headless(self):
        os.environ["HEARTH_BROWSER_HEADED"] = "1"
        captured = {}

        class CapChromium(_FakeChromium):
            def launch_persistent_context(self, user_data_dir, **kw):
                captured.update(kw)
                return super().launch_persistent_context(user_data_dir, **kw)

        class CapPW(_FakePW):
            def __init__(self):
                self.chromium = CapChromium()

        fake_sync = types.ModuleType("playwright.sync_api")
        fake_sync.sync_playwright = CapPW
        fake_pkg = types.ModuleType("playwright")
        fake_pkg.sync_api = fake_sync
        sys.modules["playwright"] = fake_pkg
        sys.modules["playwright.sync_api"] = fake_sync

        browser_tool.register(self.ctx)
        registry.call_tool("browser_open", {"url": "http://example.com/"})
        self.assertFalse(captured["headless"])

    def test_profile_dir_override_env(self):
        os.environ["HEARTH_BROWSER_PROFILE"] = "/tmp/custom-profile-test"
        p = browser_tool.profile_dir(self.ctx)
        self.assertEqual(str(p), "/tmp/custom-profile-test")
        os.environ.pop("HEARTH_BROWSER_PROFILE")
        p2 = browser_tool.profile_dir({})
        self.assertTrue(str(p2).endswith("browser-profile"))

    def test_screenshot_returns_base64(self):
        _install_fake_playwright()
        browser_tool.register(self.ctx)
        res = registry.call_tool("browser_screenshot", {})
        self.assertTrue(res["ok"])
        self.assertEqual(base64.b64decode(res["png_base64"]), b"FAKEPNG")

    def test_close_resets_state(self):
        _install_fake_playwright()
        browser_tool.register(self.ctx)
        registry.call_tool("browser_open", {"url": "http://example.com/"})
        res = registry.call_tool("browser_close", {})
        self.assertTrue(res["ok"])


if __name__ == "__main__":
    unittest.main()

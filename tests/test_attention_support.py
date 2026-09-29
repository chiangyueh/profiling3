from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attention_support import load_attention_config, make_input_params, make_validator


class AttentionSupportTest(unittest.TestCase):
    def test_forward_and_backward_configs_are_valid(self):
        for name in ("flash_attention_score.json", "flash_attention_score_grad.json"):
            config = load_attention_config(ROOT / "configs" / name)
            validator = make_validator(config)
            params = validator.get_combinations(1, make_input_params(config))[0]
            self.assertTrue(validator.is_valid(params))

    def test_both_operators_complete_a_search_step(self):
        mock = ROOT / "tests" / "mock_attention_runner.sh"
        for name, env_name in (
            ("flash_attention_score.json", "FA_FORWARD_RUNNER"),
            ("flash_attention_score_grad.json", "FA_BACKWARD_RUNNER"),
        ):
            with self.subTest(config=name), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "result.json"
                cache = Path(temporary) / "cache.json"
                env = os.environ.copy()
                env[env_name] = str(mock)
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "search_attention.py"),
                        "--config",
                        str(ROOT / "configs" / name),
                        "--steps",
                        "1",
                        "--cache",
                        str(cache),
                        "--output",
                        str(output),
                    ],
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = json.loads(output.read_text(encoding="utf-8"))
                self.assertGreater(result["best"]["duration"], 0)
                self.assertEqual(result["operator"], name[:-5])


if __name__ == "__main__":
    unittest.main()

"""最小煙霧測試：套件能 import。"""

import edge_llm_router


def test_package_imports() -> None:
    assert edge_llm_router.__version__ == "0.1.0"

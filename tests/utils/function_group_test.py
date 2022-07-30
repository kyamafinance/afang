from typing import List

from afang.utils.function_group import FunctionGroup


def test_function_group() -> None:
    function_group = FunctionGroup()

    class SampleTest:
        def __init__(self) -> None:
            self.results: List[str] = []

        @function_group.add
        def a(self) -> None:
            self.results.append("a")

        @function_group.add
        def b(self) -> None:
            self.results.append("b")

        @function_group.add
        def c(self) -> None:
            self.results.append("c")

        def run(self) -> None:
            function_group(self)

    sample_test = SampleTest()
    sample_test.run()

    assert sample_test.results == ["a", "b", "c"]

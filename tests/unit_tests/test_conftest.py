# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from tests.unit_tests.conftest import _selected_tests_require_unit_test_data

pytestmark = pytest.mark.no_unit_test_data


class _FakeItem:
    def __init__(self, *, data_independent: bool) -> None:
        self.data_independent = data_independent

    def get_closest_marker(self, name: str) -> object | None:
        if name == "no_unit_test_data" and self.data_independent:
            return object()
        return None


def test_selected_data_independent_tests_skip_shared_unit_test_data() -> None:
    items = [_FakeItem(data_independent=True), _FakeItem(data_independent=True)]

    assert not _selected_tests_require_unit_test_data(items)


def test_any_unmarked_test_requires_shared_unit_test_data() -> None:
    items = [_FakeItem(data_independent=True), _FakeItem(data_independent=False)]

    assert _selected_tests_require_unit_test_data(items)

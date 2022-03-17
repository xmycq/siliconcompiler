import pytest

# Only run daily -- this will probably be slowish if we make microwatt example
# go from end-to-end, and we already have a quick GHDL test.

@pytest.mark.eda
def test_py(setup_example_test, microwatt_dir):
    # Note: value of microwatt_dir is unused, but specifying it is important to
    # ensure that the submodule is cloned.

    setup_example_test('microwatt')

    import build
    build.main()
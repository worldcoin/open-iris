import pytest

from iris.io.errors import IRISPipelineError
from iris.orchestration.validators import pipeline_config_duplicate_node_name_check, pipeline_metadata_version_check


class MockClass:
    """Mock class for testing validators."""

    pass


class MockField:
    """Mock field for testing validators."""

    name = "mock_field"


class MockNode:
    """Mock node with a name attribute."""

    def __init__(self, name: str):
        self.name = name


class TestPipelineConfigDuplicateNodeNameCheck:
    """Test suite for pipeline_config_duplicate_node_name_check validator."""

    def test_unique_node_names_passes(self):
        """Test that unique node names pass validation."""
        nodes = [MockNode("node1"), MockNode("node2"), MockNode("node3")]

        result = pipeline_config_duplicate_node_name_check(MockClass, nodes, MockField)

        assert result == nodes

    def test_empty_list_passes(self):
        """Test that empty node list passes validation."""
        nodes = []

        result = pipeline_config_duplicate_node_name_check(MockClass, nodes, MockField)

        assert result == nodes

    def test_single_node_passes(self):
        """Test that single node passes validation."""
        nodes = [MockNode("single_node")]

        result = pipeline_config_duplicate_node_name_check(MockClass, nodes, MockField)

        assert result == nodes

    @pytest.mark.parametrize(
        "node_names,expected_duplicates",
        [
            (["node1", "node1"], {"node1": 2}),
            (["node1", "node2", "node1"], {"node1": 2}),
            (["node1", "node2", "node2", "node3"], {"node2": 2}),
            (["dup", "dup", "dup"], {"dup": 3}),
            (["a", "b", "a", "b"], {"a": 2, "b": 2}),
        ],
        ids=[
            "simple_duplicate",
            "duplicate_with_unique",
            "duplicate_in_middle",
            "triple_duplicate",
            "multiple_duplicates",
        ],
    )
    def test_duplicate_node_names_raises_error(self, node_names, expected_duplicates):
        """Test that duplicate node names raise IRISPipelineError."""
        nodes = [MockNode(name) for name in node_names]

        with pytest.raises(IRISPipelineError) as exc_info:
            pipeline_config_duplicate_node_name_check(MockClass, nodes, MockField)

        error_message = str(exc_info.value)
        assert "Pipeline node name must be unique" in error_message
        # Check that the error message contains information about duplicates
        for name, count in expected_duplicates.items():
            assert f"'{name}': {count}" in error_message


class TestPipelineMetadataVersionCheck:
    """Test suite for pipeline_metadata_version_check validator."""

    def test_matching_version_passes(self):
        """Test that matching version passes validation."""
        input_version = "1.2.3"
        expected_version = "1.2.3"

        result = pipeline_metadata_version_check(MockClass, input_version, MockField, expected_version)

        assert result == input_version

    @pytest.mark.parametrize(
        "input_version,expected_version",
        [
            ("1.0.0", "1.0.0"),
            ("2.5.1", "2.5.1"),
            ("1.6.1", "1.6.1"),
            ("0.1.0-alpha", "0.1.0-alpha"),
            ("1.0.0-rc.1", "1.0.0-rc.1"),
        ],
        ids=[
            "basic_semver",
            "different_version",
            "iris_version",
            "alpha_version",
            "release_candidate",
        ],
    )
    def test_various_matching_versions_pass(self, input_version, expected_version):
        """Test that various matching version formats pass validation."""
        result = pipeline_metadata_version_check(MockClass, input_version, MockField, expected_version)

        assert result == input_version

    @pytest.mark.parametrize(
        "input_version,expected_version",
        [
            ("1.0.0", "1.0.1"),
            ("2.0.0", "1.0.0"),
            ("1.6.1", "2.0.0"),
            ("1.0.0-alpha", "1.0.0"),
            ("1.0.0", "1.0.0-beta"),
        ],
        ids=[
            "patch_mismatch",
            "major_mismatch",
            "different_versions",
            "alpha_vs_release",
            "release_vs_beta",
        ],
    )
    def test_mismatched_versions_raise_error(self, input_version, expected_version):
        """Test that mismatched versions raise IRISPipelineError."""
        with pytest.raises(IRISPipelineError) as exc_info:
            pipeline_metadata_version_check(MockClass, input_version, MockField, expected_version)

        error_message = str(exc_info.value)
        assert "Wrong config version" in error_message
        assert f"version {expected_version}" in error_message
        assert f"version {input_version}" in error_message

    def test_error_message_format(self):
        """Test that error message has the expected format."""
        input_version = "1.0.0"
        expected_version = "2.0.0"

        with pytest.raises(IRISPipelineError) as exc_info:
            pipeline_metadata_version_check(MockClass, input_version, MockField, expected_version)

        error_message = str(exc_info.value)
        expected_pattern = f"Wrong config version. Cannot initialise IRISPipeline version {expected_version} on a config file version {input_version}"
        assert error_message == expected_pattern

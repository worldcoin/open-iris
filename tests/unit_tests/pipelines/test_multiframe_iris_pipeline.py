import os
from typing import Any, Dict
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from iris.callbacks.pipeline_trace import PipelineCallTraceStorage
from iris.io.dataclasses import IRImage, IrisTemplate
from iris.orchestration.environment import Environment
from iris.orchestration.error_managers import store_error_manager
from iris.orchestration.output_builders import (
    build_multiframe_iris_pipeline_orb_output,
    build_simple_multiframe_iris_pipeline_output,
)
from iris.pipelines.base_pipeline import load_yaml_config
from iris.pipelines.multiframe_iris_pipeline import MultiframeIrisPipeline


@pytest.fixture
def ir_image() -> np.ndarray:
    ir_image_path = os.path.join(os.path.dirname(__file__), "mocks", "anonymized.png")
    img_data = cv2.imread(ir_image_path, cv2.IMREAD_GRAYSCALE)

    return img_data


class TestMultiframeIrisPipeline:
    """Test cases for the MultiframeIrisPipeline class."""

    @pytest.fixture
    def valid_iris_pipeline_config(self) -> Dict[str, Any]:
        """Create a valid iris pipeline configuration with filter_bank node."""
        return {
            "pipeline": [
                {
                    "name": "filter_bank",
                    "algorithm": {
                        "class_name": "iris.ConvFilterBank",
                        "params": {
                            "maskisduplicated": False,
                            "filters": [
                                {
                                    "class_name": "iris.GaborFilter",
                                    "params": {
                                        "kernel_size": [41, 21],
                                        "sigma_phi": 7,
                                        "sigma_rho": 6.13,
                                        "theta_degrees": 90.0,
                                        "lambda_phi": 28.0,
                                        "dc_correction": True,
                                        "to_fixpoints": True,
                                    },
                                },
                                {
                                    "class_name": "iris.GaborFilter",
                                    "params": {
                                        "kernel_size": [17, 21],
                                        "sigma_phi": 2,
                                        "sigma_rho": 5.86,
                                        "theta_degrees": 90.0,
                                        "lambda_phi": 8,
                                        "dc_correction": True,
                                        "to_fixpoints": True,
                                    },
                                },
                            ],
                            "probe_schemas": [
                                {
                                    "class_name": "iris.RegularProbeSchema",
                                    "params": {
                                        "n_rows": 16,
                                        "n_cols": 256,
                                    },
                                },
                                {
                                    "class_name": "iris.RegularProbeSchema",
                                    "params": {
                                        "n_rows": 16,
                                        "n_cols": 256,
                                    },
                                },
                            ],
                        },
                    },
                }
            ]
        }

    @pytest.fixture
    def valid_multiframe_config(self) -> Dict[str, Any]:
        """Load a valid multiframe pipeline configuration from the real YAML file."""
        from iris.pipelines.base_pipeline import load_yaml_config

        config = load_yaml_config("src/iris/pipelines/confs/templates_aggregation_pipeline.yaml")
        return {
            "iris_pipeline": {"metadata": config["metadata"], "pipeline": config["pipeline"]},
            "templates_aggregation_pipeline": config["templates_aggregation"],
        }

    def test_derive_iris_template_shape_from_config_valid_config_returns_correct_shape(
        self, valid_iris_pipeline_config
    ):
        """Test that derive_iris_template_shape_from_config with a valid configuration returns the expected shape tuple."""
        shape = MultiframeIrisPipeline.derive_iris_template_shape_from_config(valid_iris_pipeline_config)

        # Expected: (n_rows, n_cols, n_filters, n_probe_schemas)
        # n_rows=16, n_cols=256, n_filters=2, n_probe_schemas=2
        expected_shape = (16, 256, 2, 2)
        assert shape == expected_shape

    def test_derive_iris_template_shape_from_config_with_real_pipeline_config(self):
        """Test that derive_iris_template_shape_from_config works with the actual pipeline configuration from YAML."""
        # Load the real pipeline configuration
        real_config = load_yaml_config(MultiframeIrisPipeline.DEFAULT_PIPELINE_CFG_PATH)
        iris_pipeline_config = real_config["iris_pipeline"]

        # Test that it doesn't raise an exception and returns a valid shape
        shape = MultiframeIrisPipeline.derive_iris_template_shape_from_config(iris_pipeline_config)

        # Verify the shape is valid
        assert isinstance(shape, tuple)
        assert len(shape) == 4
        assert all(isinstance(dim, int) for dim in shape)
        assert all(dim > 0 for dim in shape)

        # Based on the actual pipeline.yaml, we expect (16, 256, 2, 2)
        expected_shape = (16, 256, 2, 2)
        assert shape == expected_shape

    def test_derive_iris_template_shape_from_config_missing_some_keys_raises_value_error(self):
        """Test that derive_iris_template_shape_from_config with missing 'pipeline' key or empty pipeline raises ValueError."""
        # Missing pipeline key
        config = {"some_other_key": []}

        with pytest.raises(ValueError, match="filter_bank node not found in iris pipeline configuration"):
            MultiframeIrisPipeline.derive_iris_template_shape_from_config(config)

        # Empty pipeline
        config = {"pipeline": []}
        with pytest.raises(ValueError, match="filter_bank node not found in iris pipeline configuration"):
            MultiframeIrisPipeline.derive_iris_template_shape_from_config(config)

        # Missing filter_bank node
        config = {
            "pipeline": [{"name": "other_node", "algorithm": {"class_name": "iris.OtherAlgorithm", "params": {}}}]
        }

        with pytest.raises(ValueError, match="filter_bank node not found in iris pipeline configuration"):
            MultiframeIrisPipeline.derive_iris_template_shape_from_config(config)

        # Missing algorithm params
        config = {
            "pipeline": [
                {
                    "name": "filter_bank",
                    "algorithm": {
                        "class_name": "iris.ConvFilterBank",
                        # Missing params
                    },
                }
            ]
        }

        with pytest.raises(ValueError, match="No probe_schemas found in filter_bank configuration"):
            MultiframeIrisPipeline.derive_iris_template_shape_from_config(config)

        # Missing probe_schemas
        config = {
            "pipeline": [
                {
                    "name": "filter_bank",
                    "algorithm": {
                        "class_name": "iris.ConvFilterBank",
                        "params": {
                            "filters": [
                                {
                                    "class_name": "iris.GaborFilter",
                                    "params": {"kernel_size": [41, 21]},
                                }
                            ],
                            # Missing probe_schemas
                        },
                    },
                }
            ]
        }

        with pytest.raises(ValueError, match="No probe_schemas found in filter_bank configuration"):
            MultiframeIrisPipeline.derive_iris_template_shape_from_config(config)

        # Empty probe_schemas
        config = {
            "pipeline": [
                {
                    "name": "filter_bank",
                    "algorithm": {
                        "class_name": "iris.ConvFilterBank",
                        "params": {
                            "filters": [
                                {
                                    "class_name": "iris.GaborFilter",
                                    "params": {"kernel_size": [41, 21]},
                                }
                            ],
                            "probe_schemas": [],  # Empty list
                        },
                    },
                }
            ]
        }

        with pytest.raises(ValueError, match="No probe_schemas found in filter_bank configuration"):
            MultiframeIrisPipeline.derive_iris_template_shape_from_config(config)

        # Missing n_rows
        config = {
            "pipeline": [
                {
                    "name": "filter_bank",
                    "algorithm": {
                        "class_name": "iris.ConvFilterBank",
                        "params": {
                            "filters": [
                                {
                                    "class_name": "iris.GaborFilter",
                                    "params": {"kernel_size": [41, 21]},
                                }
                            ],
                            "probe_schemas": [
                                {
                                    "class_name": "iris.RegularProbeSchema",
                                    "params": {
                                        # Missing n_rows
                                        "n_cols": 256,
                                    },
                                }
                            ],
                        },
                    },
                }
            ]
        }

        with pytest.raises(ValueError, match="n_rows or n_cols not found in probe schema configuration"):
            MultiframeIrisPipeline.derive_iris_template_shape_from_config(config)

        # Missing n_cols
        config = {
            "pipeline": [
                {
                    "name": "filter_bank",
                    "algorithm": {
                        "class_name": "iris.ConvFilterBank",
                        "params": {
                            "filters": [
                                {
                                    "class_name": "iris.GaborFilter",
                                    "params": {"kernel_size": [41, 21]},
                                }
                            ],
                            "probe_schemas": [
                                {
                                    "class_name": "iris.RegularProbeSchema",
                                    "params": {
                                        "n_rows": 16,
                                        # Missing n_cols
                                    },
                                }
                            ],
                        },
                    },
                }
            ]
        }

        with pytest.raises(ValueError, match="n_rows or n_cols not found in probe schema configuration"):
            MultiframeIrisPipeline.derive_iris_template_shape_from_config(config)

        # Missing filters
        config = {
            "pipeline": [
                {
                    "name": "filter_bank",
                    "algorithm": {
                        "class_name": "iris.ConvFilterBank",
                        "params": {
                            # Missing filters
                            "probe_schemas": [
                                {
                                    "class_name": "iris.RegularProbeSchema",
                                    "params": {
                                        "n_rows": 16,
                                        "n_cols": 256,
                                    },
                                }
                            ],
                        },
                    },
                }
            ]
        }

        with pytest.raises(ValueError, match="No filters found in filter_bank configuration"):
            MultiframeIrisPipeline.derive_iris_template_shape_from_config(config)

    def test_derive_iris_template_shape_from_config_different_probe_schemas_count(self):
        """Test that derive_iris_template_shape_from_config correctly counts probe_schemas."""
        config = {
            "pipeline": [
                {
                    "name": "filter_bank",
                    "algorithm": {
                        "class_name": "iris.ConvFilterBank",
                        "params": {
                            "filters": [
                                {
                                    "class_name": "iris.GaborFilter",
                                    "params": {"kernel_size": [41, 21]},
                                }
                            ],
                            "probe_schemas": [
                                {
                                    "class_name": "iris.RegularProbeSchema",
                                    "params": {
                                        "n_rows": 8,
                                        "n_cols": 128,
                                    },
                                },
                                {
                                    "class_name": "iris.RegularProbeSchema",
                                    "params": {
                                        "n_rows": 8,
                                        "n_cols": 128,
                                    },
                                },
                                {
                                    "class_name": "iris.RegularProbeSchema",
                                    "params": {
                                        "n_rows": 8,
                                        "n_cols": 128,
                                    },
                                },
                            ],
                        },
                    },
                }
            ]
        }

        # Should return (n_rows, n_cols, n_filters, n_probe_schemas)
        shape = MultiframeIrisPipeline.derive_iris_template_shape_from_config(config)
        expected_shape = (8, 128, 1, 3)  # n_rows=8, n_cols=128, n_filters=1, n_probe_schemas=3
        assert shape == expected_shape

    def test_derive_iris_template_shape_from_config_single_probe_schema(self):
        """Test derive_iris_template_shape_from_config with a single probe schema."""
        config = {
            "pipeline": [
                {
                    "name": "filter_bank",
                    "algorithm": {
                        "class_name": "iris.ConvFilterBank",
                        "params": {
                            "filters": [
                                {
                                    "class_name": "iris.GaborFilter",
                                    "params": {"kernel_size": [41, 21]},
                                }
                            ],
                            "probe_schemas": [
                                {
                                    "class_name": "iris.RegularProbeSchema",
                                    "params": {
                                        "n_rows": 4,
                                        "n_cols": 64,
                                    },
                                }
                            ],
                        },
                    },
                }
            ]
        }

        shape = MultiframeIrisPipeline.derive_iris_template_shape_from_config(config)
        expected_shape = (4, 64, 1, 1)  # n_rows=4, n_cols=64, n_filters=1, n_probe_schemas=1
        assert shape == expected_shape

    def test_derive_iris_template_shape_from_config_multiple_filters(self):
        """Test derive_iris_template_shape_from_config with multiple filters."""
        config = {
            "pipeline": [
                {
                    "name": "filter_bank",
                    "algorithm": {
                        "class_name": "iris.ConvFilterBank",
                        "params": {
                            "filters": [
                                {
                                    "class_name": "iris.GaborFilter",
                                    "params": {"kernel_size": [41, 21]},
                                },
                                {
                                    "class_name": "iris.GaborFilter",
                                    "params": {"kernel_size": [17, 21]},
                                },
                                {
                                    "class_name": "iris.GaborFilter",
                                    "params": {"kernel_size": [25, 15]},
                                },
                            ],
                            "probe_schemas": [
                                {
                                    "class_name": "iris.RegularProbeSchema",
                                    "params": {
                                        "n_rows": 32,
                                        "n_cols": 512,
                                    },
                                }
                            ],
                        },
                    },
                }
            ]
        }

        shape = MultiframeIrisPipeline.derive_iris_template_shape_from_config(config)
        expected_shape = (32, 512, 3, 1)  # n_rows=32, n_cols=512, n_filters=3, n_probe_schemas=1
        assert shape == expected_shape

    def test_derive_iris_template_shape_from_config_return_type_is_tuple(self, valid_iris_pipeline_config):
        """Test that derive_iris_template_shape_from_config returns a tuple of integers."""
        shape = MultiframeIrisPipeline.derive_iris_template_shape_from_config(valid_iris_pipeline_config)

        assert isinstance(shape, tuple)
        assert len(shape) == 4
        assert all(isinstance(dim, int) for dim in shape)
        assert all(dim > 0 for dim in shape)  # All dimensions should be positive

    def test_load_config_with_dict_input(self, valid_multiframe_config):
        """Test load_config method with dictionary input."""
        iris_config, templates_config = MultiframeIrisPipeline.load_config(valid_multiframe_config)

        assert isinstance(iris_config, dict)
        assert isinstance(templates_config, dict)
        assert iris_config == valid_multiframe_config["iris_pipeline"]
        assert templates_config == valid_multiframe_config["templates_aggregation_pipeline"]

    def test_load_config_with_none_input(self):
        """Test load_config method with None input (should use default path)."""
        with patch("iris.pipelines.multiframe_iris_pipeline.load_yaml_config") as mock_load:
            mock_load.return_value = {
                "iris_pipeline": {"test": "iris"},
                "templates_aggregation_pipeline": {"test": "templates"},
            }

            iris_config, templates_config = MultiframeIrisPipeline.load_config(None)

            mock_load.assert_called_once_with(MultiframeIrisPipeline.DEFAULT_PIPELINE_CFG_PATH)
            assert iris_config == {"test": "iris"}
            assert templates_config == {"test": "templates"}

    def test_load_config_with_empty_string_input(self):
        """Test load_config method with empty string input (should use default path)."""
        with patch("iris.pipelines.multiframe_iris_pipeline.load_yaml_config") as mock_load:
            mock_load.return_value = {
                "iris_pipeline": {"test": "iris"},
                "templates_aggregation_pipeline": {"test": "templates"},
            }

            iris_config, templates_config = MultiframeIrisPipeline.load_config("")

            mock_load.assert_called_once_with(MultiframeIrisPipeline.DEFAULT_PIPELINE_CFG_PATH)
            assert iris_config == {"test": "iris"}
            assert templates_config == {"test": "templates"}

    def test_load_config_missing_iris_pipeline_raises_value_error(self):
        """Test load_config method with missing iris_pipeline key raises ValueError."""
        config = {
            "templates_aggregation_pipeline": {"test": "templates"}
            # Missing iris_pipeline
        }

        with pytest.raises(ValueError, match="MultiframeIrisPipeline requires 'iris_pipeline' in the configuration."):
            MultiframeIrisPipeline.load_config(config)

    def test_load_config_missing_templates_aggregation_pipeline_raises_value_error(self):
        """Test load_config method with missing templates_aggregation_pipeline key raises ValueError."""
        config = {
            "iris_pipeline": {"test": "iris"}
            # Missing templates_aggregation_pipeline
        }

        with pytest.raises(
            ValueError, match="MultiframeIrisPipeline requires 'templates_aggregation_pipeline' in the configuration."
        ):
            MultiframeIrisPipeline.load_config(config)

    def test_load_from_config_success(self, valid_multiframe_config):
        """Test load_from_config method with valid base64 config."""
        import yaml

        from iris.utils.base64_encoding import base64_encode_str

        config_str = base64_encode_str(yaml.dump(valid_multiframe_config))
        result = MultiframeIrisPipeline.load_from_config(config_str)

        assert "agent" in result
        assert "error" in result
        # The error might not be None due to validation issues, but the structure should be correct
        assert result["agent"] is None or isinstance(result["agent"], MultiframeIrisPipeline)

    def test_load_from_config_invalid_base64_raises_error(self):
        """Test load_from_config method with invalid base64 raises error."""
        result = MultiframeIrisPipeline.load_from_config("invalid_base64")

        assert "agent" in result
        assert "error" in result
        assert result["agent"] is None
        assert result["error"] is not None
        assert "error_type" in result["error"]
        assert "message" in result["error"]
        assert "traceback" in result["error"]

    def test_initialization_with_valid_config(self, valid_multiframe_config):
        """Test MultiframeIrisPipeline initialization with valid config."""
        with patch("iris.pipelines.multiframe_iris_pipeline.IRISPipeline") as _, patch(
            "iris.pipelines.multiframe_iris_pipeline.TemplatesAggregationPipeline"
        ) as __:
            # Mock the derive_iris_template_shape_from_config method to be called without arguments
            with patch.object(
                MultiframeIrisPipeline, "derive_iris_template_shape_from_config", return_value=(16, 256, 1, 1)
            ) as mock_derive:
                pipeline = MultiframeIrisPipeline(config=valid_multiframe_config)

                # Verify the method was called (it will be called without arguments in the constructor)
                mock_derive.assert_called_once()

                assert pipeline.iris_pipeline_config == valid_multiframe_config["iris_pipeline"]
                assert (
                    pipeline.templates_aggregation_pipeline_config
                    == valid_multiframe_config["templates_aggregation_pipeline"]
                )
                assert pipeline.iris_template_shape == (16, 256, 1, 1)  # Based on the mocked return value

    def test_initialization_with_partial_config(self, valid_multiframe_config):
        """Test MultiframeIrisPipeline initialization with invalid config."""
        with pytest.raises(ValueError, match="MultiframeIrisPipeline requires 'iris_pipeline' in the configuration."):
            MultiframeIrisPipeline(config={"invalid": "config"})

        valid_cfg = valid_multiframe_config.copy()
        valid_cfg.pop("iris_pipeline")
        with pytest.raises(ValueError, match="MultiframeIrisPipeline requires 'iris_pipeline' in the configuration."):
            MultiframeIrisPipeline(config=valid_cfg)

        valid_cfg = valid_multiframe_config.copy()
        valid_cfg.pop("templates_aggregation_pipeline")
        with pytest.raises(
            ValueError, match="MultiframeIrisPipeline requires 'templates_aggregation_pipeline' in the configuration."
        ):
            MultiframeIrisPipeline(config=valid_cfg)

    def test_initialization_with_default_environment(self, valid_multiframe_config):
        """Test MultiframeIrisPipeline initialization with default environment."""
        with patch("iris.pipelines.multiframe_iris_pipeline.IRISPipeline") as _, patch(
            "iris.pipelines.multiframe_iris_pipeline.TemplatesAggregationPipeline"
        ) as __:
            # Mock the derive_iris_template_shape_from_config method
            with patch.object(
                MultiframeIrisPipeline, "derive_iris_template_shape_from_config", return_value=(16, 256, 1, 1)
            ):
                pipeline = MultiframeIrisPipeline(config=valid_multiframe_config)

                # Check that the default environment is used
                assert pipeline.env.pipeline_output_builder == build_simple_multiframe_iris_pipeline_output
                assert pipeline.env.error_manager == store_error_manager
                assert pipeline.env.call_trace_initialiser == PipelineCallTraceStorage.initialise

    def test_initialization_with_custom_environment(self, valid_multiframe_config):
        """Test MultiframeIrisPipeline initialization with custom environment."""
        custom_env = Environment(
            pipeline_output_builder=build_simple_multiframe_iris_pipeline_output,
            error_manager=store_error_manager,
            call_trace_initialiser=PipelineCallTraceStorage.initialise,
        )

        with patch("iris.pipelines.multiframe_iris_pipeline.IRISPipeline") as _, patch(
            "iris.pipelines.multiframe_iris_pipeline.TemplatesAggregationPipeline"
        ) as __:
            # Mock the derive_iris_template_shape_from_config method
            with patch.object(
                MultiframeIrisPipeline, "derive_iris_template_shape_from_config", return_value=(16, 256, 1, 1)
            ):
                pipeline = MultiframeIrisPipeline(config=valid_multiframe_config, env=custom_env)

                assert pipeline.env == custom_env

    def test_handle_input_method(self, valid_multiframe_config):
        """Test _handle_input method."""
        with patch("iris.pipelines.multiframe_iris_pipeline.IRISPipeline") as _, patch(
            "iris.pipelines.multiframe_iris_pipeline.TemplatesAggregationPipeline"
        ) as __:
            # Mock the derive_iris_template_shape_from_config method
            with patch.object(
                MultiframeIrisPipeline, "derive_iris_template_shape_from_config", return_value=(16, 256, 1, 1)
            ):
                pipeline = MultiframeIrisPipeline(config=valid_multiframe_config)

                # Mock the call_trace
                mock_call_trace = Mock()
                pipeline.call_trace = mock_call_trace

                test_input = [IRImage(img_data=np.random.rand(1, 1), image_id="image_id", eye_side="right")]
                pipeline._handle_input(test_input)

                mock_call_trace.write_input.assert_called_once_with(test_input)

    def test_handle_output_method(self, valid_multiframe_config):
        """Test _handle_output method."""
        with patch("iris.pipelines.multiframe_iris_pipeline.IRISPipeline") as _, patch(
            "iris.pipelines.multiframe_iris_pipeline.TemplatesAggregationPipeline"
        ) as __:
            # Mock the derive_iris_template_shape_from_config method
            with patch.object(
                MultiframeIrisPipeline, "derive_iris_template_shape_from_config", return_value=(16, 256, 1, 1)
            ):
                # Create a custom environment with mocked output builder
                mock_output_builder = Mock(return_value="test_output")
                custom_env = Environment(
                    pipeline_output_builder=mock_output_builder,
                    error_manager=store_error_manager,
                    call_trace_initialiser=PipelineCallTraceStorage.initialise,
                )

                pipeline = MultiframeIrisPipeline(config=valid_multiframe_config, env=custom_env)

                result = pipeline._handle_output()

                mock_output_builder.assert_called_once_with(pipeline.call_trace)
                assert result == "test_output"

    def test_handle_pipeline_error_method(self, valid_multiframe_config):
        """Test _handle_pipeline_error method."""
        with patch("iris.pipelines.multiframe_iris_pipeline.IRISPipeline") as _, patch(
            "iris.pipelines.multiframe_iris_pipeline.TemplatesAggregationPipeline"
        ) as __:
            # Mock the derive_iris_template_shape_from_config method
            with patch.object(
                MultiframeIrisPipeline, "derive_iris_template_shape_from_config", return_value=(16, 256, 1, 1)
            ):
                # Create a custom environment with mocked error manager
                mock_error_manager = Mock()
                custom_env = Environment(
                    pipeline_output_builder=build_simple_multiframe_iris_pipeline_output,
                    error_manager=mock_error_manager,
                    call_trace_initialiser=PipelineCallTraceStorage.initialise,
                )

                pipeline = MultiframeIrisPipeline(config=valid_multiframe_config, env=custom_env)

                test_error = ValueError("Test error")

                result = pipeline._handle_pipeline_error(test_error)

                mock_error_manager.assert_called_once_with(pipeline.call_trace, test_error)
                assert result is False  # Default behavior is to not skip errors

    def test_orb_environment_constant(self):
        """Test that ORB_ENVIRONMENT is properly configured."""
        env = MultiframeIrisPipeline.ORB_ENVIRONMENT
        assert env.pipeline_output_builder == build_multiframe_iris_pipeline_orb_output
        assert env.error_manager == store_error_manager
        assert env.call_trace_initialiser == PipelineCallTraceStorage.initialise

    def test_default_pipeline_cfg_path_constant(self):
        """Test that DEFAULT_PIPELINE_CFG_PATH points to a valid file."""
        import os

        assert os.path.exists(MultiframeIrisPipeline.DEFAULT_PIPELINE_CFG_PATH)

    @pytest.fixture
    def mock_iris_template(self):
        """Create a mock IrisTemplate for testing."""
        template = Mock(spec=IrisTemplate)
        template.shape = (16, 256, 2, 2)
        return template

    @pytest.fixture
    def mock_iris_pipeline_output(self, mock_iris_template):
        """Create a mock iris pipeline output."""
        return {"iris_template": mock_iris_template, "other_data": "test_data"}

    @pytest.fixture
    def mock_aggregation_pipeline_output(self):
        """Create a mock aggregation pipeline output."""
        return {"aggregated_template": Mock(spec=IrisTemplate), "aggregation_metadata": {"method": "test"}}

    def test_run_iris_pipeline_success(self, valid_multiframe_config, mock_iris_template):
        """Test successful execution of run_iris_pipeline."""
        pipeline = MultiframeIrisPipeline(config=valid_multiframe_config)

        # Mock the iris pipeline
        mock_iris_pipeline = Mock()
        mock_iris_pipeline.run.return_value = {"iris_template": mock_iris_template, "error": None}
        pipeline.iris_pipeline = mock_iris_pipeline

        # Mock call_trace
        mock_call_trace = Mock()
        pipeline.call_trace = mock_call_trace

        ir_images = [
            IRImage(img_data=np.random.rand(10, 10), image_id="img1", eye_side="left"),
            IRImage(img_data=np.random.rand(15, 15), image_id="img2", eye_side="left"),
        ]

        iris_templates, individual_templates_output = pipeline._run_iris_pipeline(ir_images)

        # Verify iris pipeline was called for each image
        assert mock_iris_pipeline.run.call_count == 2
        mock_iris_pipeline.run.assert_any_call(ir_images[0])
        mock_iris_pipeline.run.assert_any_call(ir_images[1])

        # Verify outputs
        assert len(iris_templates) == 2
        assert len(individual_templates_output) == 2
        assert all(template == mock_iris_template for template in iris_templates)
        assert all(output["iris_template"] == mock_iris_template for output in individual_templates_output)

        # Verify call_trace was written
        mock_call_trace.write.assert_called_once_with("individual_frames", individual_templates_output)

    def test_run_with_invalid_input(
        self,
        valid_multiframe_config,
    ):
        """Test run_iris_pipeline when iris_template is None."""
        pipeline = MultiframeIrisPipeline(config=valid_multiframe_config)
        eye_side = "left"
        with pytest.raises(ValueError, match="pipeline_input must be a list of IRImage."):
            pipeline.run(None)

        with pytest.raises(ValueError, match="pipeline_input must be a list of IRImage."):
            pipeline.run([1, 2, 3])

        with pytest.raises(ValueError, match="All IRImage objects must have the same eye_side."):
            pipeline.run([
                IRImage(img_data=np.random.rand(10, 10), image_id="img1", eye_side="left"),
                IRImage(img_data=np.random.rand(10, 10), image_id="img2", eye_side="right")
            ])

        # valid run but error inside the call_trace
        ir_images = [IRImage(img_data=np.random.rand(10, 10), image_id="img1", eye_side=eye_side)]
        output = pipeline.run(ir_images)
        assert output["error"] is not None
        assert output["error"]["error_type"] == "IRISPipelineError"
        assert output["iris_template"] is None
        assert output["individual_frames"][0]["error"] is not None
        assert output["individual_frames"][0]["error"]["error_type"] == "VectorizationError"

    def test_run_same_image(self, valid_multiframe_config, ir_image):
        """Test successful execution of the complete run method."""
        pipeline = MultiframeIrisPipeline(config=valid_multiframe_config)
        for i in range(1, 3):
            ir_images = [IRImage(img_data=ir_image, image_id="img1", eye_side="left")] * i
            output = pipeline.run(ir_images)

            assert output["error"] is None
            assert output["iris_template"] is not None
            assert output["individual_frames"][0]["error"] is None
            assert len(output["individual_frames"]) == i
            assert output["templates_aggregation_metadata"] is not None
            assert output["templates_aggregation_metadata"]["error"] is None

    def test_run_one_corrupted_image(self, valid_multiframe_config, ir_image):
        """Test successful execution of the complete run method."""
        pipeline = MultiframeIrisPipeline(config=valid_multiframe_config)
        ir_images = [
            IRImage(img_data=ir_image, image_id="img1", eye_side="left"),
            IRImage(img_data=np.random.rand(10, 10), image_id="img2", eye_side="left"),
        ]
        output = pipeline.run(ir_images)
        assert output["error"] is not None
        assert output["error"]["error_type"] == "IRISPipelineError"
        assert len(output["individual_frames"]) == 2
        assert output["iris_template"] is None

        output = pipeline.run([IRImage(img_data=np.random.rand(10, 10), image_id="img2", eye_side="left")])
        assert output["error"] is not None
        assert output["error"]["error_type"] == "IRISPipelineError"
        assert output["iris_template"] is None
        assert len(output["individual_frames"]) == 1

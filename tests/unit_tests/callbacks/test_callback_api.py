from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm


class MockCallback(Callback):
    ON_EXECUTE_START_MSG = "on_execute_start"
    ON_EXECUTE_END_MSG = "on_execute_end"
    MSG_SEP = " "

    def __init__(self):
        self.buffer = ""

    def on_execute_start(self) -> None:
        self._clean_buffer()
        self.buffer = MockCallback.ON_EXECUTE_START_MSG

    def on_execute_end(self, result: str) -> None:
        self.buffer = (
            f"{self.buffer}{MockCallback.MSG_SEP}{result}{MockCallback.MSG_SEP}{MockCallback.ON_EXECUTE_END_MSG}"
        )

    def _clean_buffer(self) -> None:
        self.buffer = ""


class MockParameterizedModel(Algorithm):
    EXECUTE_OUTPUT = "WorldcoinAI"

    def run(self) -> str:
        return "WorldcoinAI"


def test_callback_api() -> None:
    mock_parametrized_model = MockParameterizedModel()
    mock_callback = MockCallback()

    mock_parametrized_model._callbacks = [mock_callback]

    expected_cb_buffer = f"{MockCallback.ON_EXECUTE_START_MSG}{MockCallback.MSG_SEP}{MockParameterizedModel.EXECUTE_OUTPUT}{MockCallback.MSG_SEP}{MockCallback.ON_EXECUTE_END_MSG}"

    _ = mock_parametrized_model()

    assert mock_callback.buffer == expected_cb_buffer

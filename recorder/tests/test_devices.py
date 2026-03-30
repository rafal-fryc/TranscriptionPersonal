from unittest.mock import MagicMock, patch
import pytest

from recorder.devices import get_loopback_device, get_microphone, list_devices
from recorder.exceptions import DeviceNotFoundError


@pytest.fixture
def mock_speaker() -> MagicMock:
    speaker = MagicMock()
    speaker.name = "Speakers (Realtek Audio)"
    return speaker


@pytest.fixture
def mock_loopback() -> MagicMock:
    loopback = MagicMock()
    loopback.name = "Speakers (Realtek Audio)"
    loopback.isloopback = True
    return loopback


@pytest.fixture
def mock_mic() -> MagicMock:
    mic = MagicMock()
    mic.name = "Microphone (Realtek Audio)"
    mic.isloopback = False
    return mic


class TestGetLoopbackDevice:
    @patch("recorder.devices.sc")
    def test_should_return_loopback_for_default_speaker(
        self, mock_sc: MagicMock, mock_speaker: MagicMock, mock_loopback: MagicMock
    ) -> None:
        mock_sc.default_speaker.return_value = mock_speaker
        mock_sc.get_microphone.return_value = mock_loopback
        result = get_loopback_device()
        assert result == mock_loopback
        mock_sc.get_microphone.assert_called_once_with(
            id=str(mock_speaker.name), include_loopback=True
        )

    @patch("recorder.devices.sc")
    def test_should_raise_when_no_loopback_found(
        self, mock_sc: MagicMock, mock_speaker: MagicMock
    ) -> None:
        mock_sc.default_speaker.return_value = mock_speaker
        mock_sc.get_microphone.side_effect = RuntimeError("not found")
        with pytest.raises(DeviceNotFoundError):
            get_loopback_device()


class TestGetMicrophone:
    @patch("recorder.devices.sc")
    def test_should_return_default_mic(
        self, mock_sc: MagicMock, mock_mic: MagicMock
    ) -> None:
        mock_sc.default_microphone.return_value = mock_mic
        result = get_microphone()
        assert result == mock_mic

    @patch("recorder.devices.sc")
    def test_should_raise_when_no_mic(self, mock_sc: MagicMock) -> None:
        mock_sc.default_microphone.return_value = None
        with pytest.raises(DeviceNotFoundError):
            get_microphone()


class TestListDevices:
    @patch("recorder.devices.sc")
    def test_should_list_all_device_categories(
        self,
        mock_sc: MagicMock,
        mock_speaker: MagicMock,
        mock_mic: MagicMock,
        mock_loopback: MagicMock,
    ) -> None:
        mock_sc.all_speakers.return_value = [mock_speaker]
        mock_sc.all_microphones.side_effect = [
            [mock_mic],
            [mock_mic, mock_loopback],
        ]
        result = list_devices()
        assert "speakers" in result
        assert "microphones" in result
        assert "loopback_devices" in result
        assert len(result["speakers"]) == 1
        assert len(result["microphones"]) == 1
        assert len(result["loopback_devices"]) == 1

import ntcore

from config.config import ConfigStore


class CalibrationCommandSource:
    def get_calibrating(self, config_store: ConfigStore) -> bool:
        return False

    def get_capture_flag(self, config_store: ConfigStore) -> bool:
        return False


class NTCalibrationCommandSource(CalibrationCommandSource):
    def __init__(self) -> None:
        self._init_complete = False
        self._active_entry: ntcore.BooleanEntry | None = None
        self._capture_flag_entry: ntcore.BooleanEntry | None = None

    def _init(self, config_store: ConfigStore) -> None:
        if not self._init_complete:
            nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
                "/" + config_store.local_config.device_id + "/calibration"
            )
            self._active_entry = nt_table.getBooleanTopic("active").getEntry(False)
            self._capture_flag_entry = nt_table.getBooleanTopic(
                "capture_flag"
            ).getEntry(False)
            self._active_entry.set(False)
            self._capture_flag_entry.set(False)
            self._init_complete = True

    def get_calibrating(self, config_store: ConfigStore) -> bool:
        self._init(config_store)
        assert self._active_entry is not None
        assert self._capture_flag_entry is not None
        calibrating = self._active_entry.get()
        if not calibrating:
            self._capture_flag_entry.set(False)
        return calibrating

    def get_capture_flag(self, config_store: ConfigStore) -> bool:
        self._init(config_store)
        assert self._capture_flag_entry is not None
        if self._capture_flag_entry.get():
            self._capture_flag_entry.set(False)
            return True
        return False

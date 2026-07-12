from PyQt6.QtCore import QObject, pyqtSignal


class LogPacketBus(QObject):
    """App-wide singleton — every decoded LOG_INFO/LOG_DATA packet (ptype 0x12/0x13)
    is emitted here, regardless of which transport (USB serial or WiFi) it arrived
    on. LogTransferManager subscribes to this instead of each PacketDecoder
    instance individually, since there are three of them (teensy, esp32, wifi)."""
    packet = pyqtSignal(dict)

    _instance: "LogPacketBus | None" = None

    @classmethod
    def instance(cls) -> "LogPacketBus":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

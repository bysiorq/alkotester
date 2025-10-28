# sensors.py
import spidev
import numpy as np

class MCP3008:
    def __init__(self, bus=0, device=0, max_speed_hz=1000000):
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = max_speed_hz
        self.spi.mode = 0

    def read_channel(self, ch: int) -> int:
        if ch < 0 or ch > 7:
            raise ValueError("MCP3008 channel 0..7 only")
        r = self.spi.xfer2([1, (8 | ch) << 4, 0])
        return ((r[1] & 3) << 8) | r[2]  # 0..1023

    def close(self):
        try:
            self.spi.close()
        except Exception:
            pass


class MQ3Sensor:
    def __init__(self, adc: MCP3008, channel: int, baseline_samples: int, promille_scale: float):
        self.adc = adc
        self.channel = channel
        self.baseline_samples = baseline_samples
        self.promille_scale = promille_scale
        self.baseline = None

    def calibrate_baseline(self):
        samples = [self.adc.read_channel(self.channel) for _ in range(self.baseline_samples)]
        self.baseline = float(np.median(samples))
        return self.baseline

    def read_raw(self):
        return self.adc.read_channel(self.channel)

    def promille_from_samples(self, samples):
        import numpy as _np
        v = float(_np.mean(samples)) if samples else float(self.read_raw())
        if self.baseline is None:
            self.baseline = v
        delta = max(0.0, v - self.baseline)
        return delta / max(1e-6, self.promille_scale)

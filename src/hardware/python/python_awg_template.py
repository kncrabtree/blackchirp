"""
Blackchirp Python AWG Driver Template

This script is loaded by the PythonAwg C++ trampoline class. It provides a
complete virtual AWG implementation that you can customize for your hardware.

The AWG (Arbitrary Waveform Generator) in Blackchirp generates chirped-pulse
waveforms for CP-FTMW spectroscopy. The C++ base class handles waveform
configuration and experiment integration; your Python script handles
communication with the actual hardware.

Class name must match the pythonClass setting (default: "AwgDriver").

Available proxies (injected automatically):
    self.comm     -- communicate with hardware via the configured protocol
    self.settings -- read/write persistent settings (stored in Blackchirp)
    self.log      -- send log messages to the Blackchirp log panel
"""

import math


class AwgDriver:
    """Python AWG hardware driver.

    The AWG class in Blackchirp is a minimal interface: the base class has no
    type-specific polling methods. The primary interaction is through
    prepare_for_experiment(), where the C++ side sends waveform configuration
    and the driver programs the hardware.

    All methods are optional. Unimplemented methods return safe defaults.

    Two static helper methods are provided for computing the time-domain
    waveform and digital marker arrays from the chirp segment parameters:

        times_us, amplitudes = AwgDriver._compute_waveform(config['chirp'])
        prot, gate = AwgDriver._compute_markers(config['chirp'])

    These are equivalent to ChirpConfig::getChirpMicroseconds() and
    ChirpConfig::getMarkerData() in C++. Both require numpy and return None
    if it is unavailable or the sample rate is not configured. DDS-style AWGs
    (e.g. AD9914) do not need these; they work directly from the segment
    parameters in config['chirp']['segments'].
    """

    def initialize(self):
        """Called once when the hardware object is first created.

        Use this to set up any internal state. The comm proxy is available
        but the connection has not been tested yet.
        """
        self.log.log("AWG driver initialized")

    def test_connection(self):
        """Verify communication with the AWG hardware.

        Called when Blackchirp tests the hardware connection (e.g., on
        startup or when the user clicks "Test Connection").

        Returns:
            bool: True if communication is working, False otherwise.

        Examples:
            # Query device identity
            response = self.comm.query("*IDN?\\n")
            return len(response.strip()) > 0

            # For virtual/testing, just return True
            return True
        """
        self.log.log("Testing AWG connection")
        return True

    def read_aux_data(self):
        """Return auxiliary data for rolling data plots.

        Called periodically by Blackchirp's polling timer. Return a dict
        mapping string keys to float values. These appear in the rolling
        data plots and are saved to disk during experiments.

        Returns:
            dict[str, float]: Key-value pairs of auxiliary data.
                Return an empty dict if no auxiliary data is available.
        """
        return {}

    def prepare_for_experiment(self, config):
        """Configure the AWG for an upcoming experiment.

        Called before each experiment starts. Use this to program waveforms,
        set sample rates, configure triggers, etc.

        Args:
            config (dict): Experiment configuration containing:

                number (int): Experiment number.
                ftmw_enabled (bool): Whether an FTMW experiment is configured.
                    If False, 'chirp' and 'rf_config' are absent.

                chirp (dict): Chirp waveform parameters (when ftmw_enabled):
                    segments (list[list[dict]]): Nested list of chirp segments.
                        Outer index = chirp repetition (0..num_chirps-1).
                        Inner index = segment within that chirp.
                        Each segment dict:
                            start_freq_mhz (float): Start frequency in MHz.
                            end_freq_mhz (float): End frequency in MHz.
                            duration_us (float): Segment duration in µs.
                            alpha_us (float): Chirp rate in MHz/µs
                                              = (end - start) / duration.
                                              Zero for empty segments.
                            empty (bool): True for guard intervals (zero output).
                    num_chirps (int): Number of chirp repetitions in waveform.
                    chirp_interval_us (float): Period between chirp starts (µs).
                    pre_chirp_protection_us (float): Protection pulse pre-delay (µs).
                    post_chirp_protection_us (float): Protection pulse post-delay (µs).
                    pre_chirp_gate_us (float): Amp-enable gate pre-delay (µs).
                    post_chirp_gate_us (float): Amp-enable gate post-delay (µs).
                    sample_rate_hz (float): AWG sample clock in Hz.

                rf_config (dict): RF chain parameters (when ftmw_enabled):
                    awg_mult (float): AWG output frequency multiplier.
                    chirp_mult (float): Post-upconversion multiplier.
                    up_mix_sideband (int): Upconversion sideband (0=Upper, 1=Lower).
                    down_mix_sideband (int): Downconversion sideband (0=Upper, 1=Lower).
                    clocks (dict): Clock assignments keyed by RfConfig.ClockType int:
                        "0"=UpLO, "1"=DownLO, "2"=AwgRef, "3"=DRClock,
                        "4"=DigRef, "5"=ComRef.
                        Each value: {freq_mhz, hw_key, output}.

        Returns:
            bool: True if preparation succeeded, False to abort experiment.

        Notes:
            For memory-based AWGs (upload a full waveform to device memory):
                times_us, amplitudes = AwgDriver._compute_waveform(config['chirp'])
                prot, gate = AwgDriver._compute_markers(config['chirp'])
                # amplitudes is float64 in [-1, 1]; scale to your DAC range
                # prot/gate are bool arrays aligned sample-for-sample

            For DDS-style AWGs (program sweep parameters directly):
                seg = config['chirp']['segments'][0][0]
                start_hz = seg['start_freq_mhz'] * 1e6
                end_hz   = seg['end_freq_mhz']   * 1e6
                dur_us   = seg['duration_us']
                clk_hz   = config['rf_config']['clocks']['2']['freq_mhz'] * 1e6
        """
        self.log.log(f"Preparing AWG for experiment {config.get('number', '?')}")

        if not config.get('ftmw_enabled', False):
            return True

        # Example: memory-based AWG
        # result = AwgDriver._compute_waveform(config['chirp'])
        # if result is None:
        #     self.log.error("Could not compute waveform (numpy missing or sample rate 0)")
        #     return False
        # times_us, amplitudes = result
        # waveform_bytes = (amplitudes * 32767).astype('int16').tobytes()
        # self.comm.write_binary(waveform_bytes)

        return True

    def begin_acquisition(self):
        """Called when experiment data acquisition starts.

        The experiment has been fully configured and is now running.
        Use this to enable outputs, start triggering, etc.
        """
        self.log.debug("AWG acquisition started")

    def end_acquisition(self):
        """Called when experiment data acquisition ends.

        Use this to disable outputs, stop triggering, return to idle, etc.
        """
        self.log.debug("AWG acquisition ended")

    def sleep(self, sleeping):
        """Called when hardware enters or exits standby mode.

        Args:
            sleeping (bool): True = entering sleep, False = waking up.
        """
        if sleeping:
            self.log.debug("AWG entering sleep mode")
        else:
            self.log.debug("AWG waking from sleep mode")

    # -------------------------------------------------------------------------
    # Waveform computation helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_waveform(chirp_config):
        """Compute the time-domain chirp waveform from segment parameters.

        Equivalent to C++ ChirpConfig::getChirpMicroseconds(). The algorithm
        generates a phase-continuous multi-segment chirp: each segment is a
        linear frequency sweep described by

            y(t) = sin(2π (f0 + ½ α t) t + φ₀)

        where t is time from the segment start (µs), f0 is start_freq_mhz,
        α = alpha_us (MHz/µs), and φ₀ is the accumulated phase from previous
        segments. Empty segments output zero. The chirp is surrounded by
        zero-amplitude pre/post protection and gate delays and repeated
        num_chirps times at chirp_interval_us spacing.

        Requires numpy. Returns None if numpy is unavailable or sample_rate_hz
        is zero.

        Args:
            chirp_config (dict): The 'chirp' sub-dict from prepare_for_experiment.

        Returns:
            tuple[np.ndarray, np.ndarray] | None:
                (times_us, amplitudes) — both 1-D float64 arrays of length N.
                times_us:   sample times in microseconds.
                amplitudes: waveform values in [-1.0, 1.0].
        """
        try:
            import numpy as np
        except ImportError:
            return None

        sample_rate_hz = chirp_config.get('sample_rate_hz', 0.0)
        if sample_rate_hz <= 0.0:
            return None

        sps = sample_rate_hz / 1e6   # samples per µs
        dt  = 1.0 / sps              # µs per sample

        segments_list  = chirp_config.get('segments', [])
        num_chirps     = chirp_config.get('num_chirps', 1)
        interval_us    = chirp_config.get('chirp_interval_us', 20.0)
        pre_prot       = chirp_config.get('pre_chirp_protection_us', 0.5)
        pre_gate       = chirp_config.get('pre_chirp_gate_us', 0.5)
        post_prot      = chirp_config.get('post_chirp_protection_us', 0.5)

        if not segments_list:
            return None

        def _chirp_dur(idx):
            segs = segments_list[idx] if idx < len(segments_list) else segments_list[-1]
            return sum(s['duration_us'] for s in segs)

        total_us  = (pre_prot + pre_gate + post_prot
                     + (num_chirps - 1) * interval_us
                     + _chirp_dur(num_chirps - 1))
        n_samples = round(total_us * sps) - 1

        times = np.arange(n_samples) * dt
        out   = np.zeros(n_samples)

        for i in range(num_chirps):
            chirp_start_us = i * interval_us + pre_prot + pre_gate
            cs = round(chirp_start_us * sps)

            segs  = segments_list[i] if i < len(segments_list) else segments_list[-1]
            phase = 0.0   # accumulated phase in radians (maintained for continuity)
            seg_s = cs

            for seg in segs:
                seg_n = round(seg['duration_us'] * sps)
                seg_e = seg_s + seg_n

                if seg.get('empty', False):
                    seg_s = seg_e
                    continue

                f0    = seg['start_freq_mhz']
                alpha = seg['alpha_us']        # MHz/µs

                t = np.arange(seg_n) * dt
                out[seg_s:seg_e] = np.sin(2.0 * np.pi * (f0 + 0.5 * alpha * t) * t + phase)

                # Carry phase to next segment for continuity
                t_end = seg_n * dt
                phase = 2.0 * np.pi * (f0 * t_end + 0.5 * alpha * t_end ** 2) + phase

                seg_s = seg_e

        return times, out

    @staticmethod
    def _compute_markers(chirp_config):
        """Compute the protection/gate digital marker arrays for the waveform.

        Equivalent to C++ ChirpConfig::getMarkerData(). Returns two boolean
        arrays aligned sample-for-sample with _compute_waveform():

            protection (prot): asserted for the full protection window around
                each chirp (pre_prot + chirp + post_prot). Drives a protection
                switch to prevent the receiver from seeing the transmit pulse.

            gate (amp_enable): asserted for the gate window around each chirp
                (pre_gate + chirp + post_gate). Drives an amplifier gate (e.g.
                TWT gate pulse) to enable the transmit amplifier.

        The gate window sits inside the protection window:

            |<--- pre_prot --->|<-- pre_gate -->|<-- chirp -->|<-- post_gate -->|<-- post_prot -->|
            |   prot=T gate=F  |   prot=T gate=T              |  prot=T gate=F  |   prot=F gate=F |

        Requires numpy. Returns None if numpy is unavailable or sample_rate_hz
        is zero.

        Args:
            chirp_config (dict): The 'chirp' sub-dict from prepare_for_experiment.

        Returns:
            tuple[np.ndarray, np.ndarray] | None:
                (protection, gate) — both 1-D bool arrays of length N,
                aligned with the output of _compute_waveform().
        """
        try:
            import numpy as np
        except ImportError:
            return None

        sample_rate_hz = chirp_config.get('sample_rate_hz', 0.0)
        if sample_rate_hz <= 0.0:
            return None

        sps = sample_rate_hz / 1e6
        dt  = 1.0 / sps

        segments_list = chirp_config.get('segments', [])
        num_chirps    = chirp_config.get('num_chirps', 1)
        interval_us   = chirp_config.get('chirp_interval_us', 20.0)
        pre_prot      = chirp_config.get('pre_chirp_protection_us', 0.5)
        pre_gate      = chirp_config.get('pre_chirp_gate_us', 0.5)
        post_prot     = chirp_config.get('post_chirp_protection_us', 0.5)
        post_gate     = chirp_config.get('post_chirp_gate_us', 0.5)

        if not segments_list:
            return None

        def _chirp_dur(idx):
            segs = segments_list[idx] if idx < len(segments_list) else segments_list[-1]
            return sum(s['duration_us'] for s in segs)

        total_us  = (pre_prot + pre_gate + post_prot
                     + (num_chirps - 1) * interval_us
                     + _chirp_dur(num_chirps - 1))
        n_samples = round(total_us * sps) - 1

        prot = np.zeros(n_samples, dtype=bool)
        gate = np.zeros(n_samples, dtype=bool)

        for i in range(num_chirps):
            t0          = i * interval_us
            gate_start  = t0 + pre_prot
            chirp_end   = t0 + pre_prot + pre_gate + _chirp_dur(i)
            gate_end    = chirp_end + post_gate
            prot_end    = chirp_end + post_prot

            is_ = round(t0         * sps)
            gs  = round(gate_start * sps)
            ge  = round(gate_end   * sps) - 1
            pe  = round(prot_end   * sps) - 1

            prot[is_:pe] = True
            gate[gs:ge]  = True

        return prot, gate

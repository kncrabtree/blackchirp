"""
Blackchirp Python AWG Driver Template

This script is loaded by the PythonAwg C++ trampoline class. It provides a
complete virtual AWG implementation that you can customize for your hardware.

The AWG (Arbitrary Waveform Generator) in Blackchirp generates chirped-pulse
waveforms for CP-FTMW spectroscopy. The C++ base class handles waveform
configuration and experiment integration; your Python script handles
communication with the actual hardware.

Class name must match the Python Class setting in the Hardware Configuration
dialog (default: "AwgDriver").

Available proxies (injected automatically):
    self.comm     -- communicate with hardware via the configured protocol
    self.settings -- read/write persistent settings (stored in Blackchirp)
    self.log      -- send log messages to the Blackchirp log panel
"""

class AwgDriver:
    """Python AWG hardware driver.

    The AWG class in Blackchirp is a minimal interface: the base class has no
    type-specific polling methods. The primary interaction is through
    prepare_for_experiment(), where the C++ side sends waveform configuration
    and the driver programs the hardware.

    All methods are optional. Unimplemented methods return safe defaults.

    Three static helper methods are provided for computing the time-domain
    waveform and digital marker arrays from the chirp segment parameters:

        times_us, amplitudes = AwgDriver._compute_waveform(config['chirp'])
        indices, markers = AwgDriver._compute_markers(config['chirp'])
        packed = AwgDriver._compute_markers_packed(config['chirp'])

    These are equivalent to ChirpConfig::getChirpMicroseconds(),
    ChirpConfig::getMarkerData(), and ChirpConfig::getPackedMarkerData() in
    C++. All require numpy and return None if it is unavailable or the sample
    rate is not configured. DDS-style AWGs (e.g. AD9914) do not need these;
    they work directly from the segment parameters in
    config['chirp']['segments'].
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
                    markers (list[dict]): Marker channel definitions. Each entry:
                        name (str): User-assigned label (e.g. "Protection").
                        role (int): 0=Protection, 1=Gate, 2=Trigger, 3=Custom.
                        start_us (float): Marker start relative to chirp start
                            (µs; negative = before chirp start).
                        end_us (float): Marker end relative to chirp end
                            (µs; positive = after chirp end).
                        enabled (bool): Whether this channel is active.
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
                indices, markers = AwgDriver._compute_markers(config['chirp'])
                # amplitudes is float64 in [-1, 1]; scale to your DAC range
                # indices[k] is the channel index in config['chirp']['markers']
                # markers[k] is the corresponding bool array (same length as amplitudes)

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
        segments. Empty segments output zero. The waveform lead and tail times
        are derived from the enabled marker channels in chirp_config['markers']
        (see _compute_markers). The chirp is repeated num_chirps times at
        chirp_interval_us spacing.

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
        markers        = chirp_config.get('markers', [])

        if not segments_list:
            return None

        def _chirp_dur(idx):
            segs = segments_list[idx] if idx < len(segments_list) else segments_list[-1]
            return sum(s['duration_us'] for s in segs)

        enabled_m = [m for m in markers if m.get('enabled', True)]
        lead_us   = max((-(m['start_us']) for m in enabled_m), default=0.0)
        lead_us   = max(0.0, lead_us)
        tail_us   = max((m['end_us'] for m in enabled_m), default=0.0)
        tail_us   = max(0.0, tail_us)

        total_us  = (lead_us + tail_us
                     + (num_chirps - 1) * interval_us
                     + _chirp_dur(num_chirps - 1))
        n_samples = round(total_us * sps) - 1

        times = np.arange(n_samples) * dt
        out   = np.zeros(n_samples)

        for i in range(num_chirps):
            chirp_start_us = i * interval_us + lead_us
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
        """Compute digital marker arrays for all enabled marker channels.

        Equivalent to C++ ChirpConfig::getMarkerData(). Iterates over
        chirp_config['markers'] and returns one boolean array per enabled
        channel, aligned sample-for-sample with _compute_waveform().

        Each marker channel defines a window relative to each chirp:
            start_us: time relative to chirp start (negative = before chirp).
            end_us:   time relative to chirp end   (positive = after chirp).

        The lead time before the first sample of each chirp is derived as
        max(0, max(-m['start_us'])) over all enabled markers, matching the
        offset used by _compute_waveform().

        Requires numpy. Returns None if numpy is unavailable or sample_rate_hz
        is zero.

        Args:
            chirp_config (dict): The 'chirp' sub-dict from prepare_for_experiment.

        Returns:
            tuple[list[int], list[np.ndarray]] | None:
                (indices, arrays) where indices[k] is the zero-based position of
                the k-th enabled channel in config['chirp']['markers'] and
                arrays[k] is the corresponding 1-D bool array of length N,
                aligned with the output of _compute_waveform(). Both lists are
                empty if no marker channels are enabled.
        """
        try:
            import numpy as np
        except ImportError:
            return None

        sample_rate_hz = chirp_config.get('sample_rate_hz', 0.0)
        if sample_rate_hz <= 0.0:
            return None

        sps = sample_rate_hz / 1e6

        segments_list = chirp_config.get('segments', [])
        num_chirps    = chirp_config.get('num_chirps', 1)
        interval_us   = chirp_config.get('chirp_interval_us', 20.0)
        markers       = chirp_config.get('markers', [])

        if not segments_list:
            return None

        enabled_idx = [i for i, m in enumerate(markers) if m.get('enabled', True)]
        enabled     = [markers[i] for i in enabled_idx]

        def _chirp_dur(idx):
            segs = segments_list[idx] if idx < len(segments_list) else segments_list[-1]
            return sum(s['duration_us'] for s in segs)

        lead_us = max((-(m['start_us']) for m in enabled), default=0.0)
        lead_us = max(0.0, lead_us)
        tail_us = max((m['end_us'] for m in enabled), default=0.0)
        tail_us = max(0.0, tail_us)

        total_us  = (lead_us + tail_us
                     + (num_chirps - 1) * interval_us
                     + _chirp_dur(num_chirps - 1))
        n_samples = round(total_us * sps) - 1

        arrays = [np.zeros(n_samples, dtype=bool) for _ in enabled]

        for i in range(num_chirps):
            chirp_start = i * interval_us + lead_us
            chirp_dur   = _chirp_dur(i)
            for j, m in enumerate(enabled):
                m_start = chirp_start + m['start_us']
                m_end   = chirp_start + chirp_dur + m['end_us']
                ms = max(0, round(m_start * sps))
                me = min(n_samples, round(m_end * sps))
                if ms < me:
                    arrays[j][ms:me] = True

        return enabled_idx, arrays

    @staticmethod
    def _compute_markers_packed(chirp_config):
        """Compute a packed quint32 marker array for the waveform.

        Equivalent to C++ ChirpConfig::getPackedMarkerData(). Calls
        _compute_markers() and packs the per-channel bool arrays into a single
        uint32 array using LSB = channel 0: bit k is set when marker channel k
        is asserted. Disabled channels contribute no bits (their bit positions
        remain 0).

        This logical bit ordering matches the C++ output directly. Hardware
        implementations that use a different encoding (e.g. Tektronix MSB-first:
        bit 7 = channel 0, bit 6 = channel 1) must remap after calling this
        function.

        Requires numpy. Returns None if _compute_markers() returns None.

        Args:
            chirp_config (dict): The 'chirp' sub-dict from prepare_for_experiment.

        Returns:
            np.ndarray | None:
                1-D uint32 array of length N, aligned with the output of
                _compute_waveform(). Each element is a bitmask of active marker
                channels for that sample.
        """
        try:
            import numpy as np
        except ImportError:
            return None

        result = AwgDriver._compute_markers(chirp_config)
        if result is None:
            return None

        indices, arrays = result
        if not arrays:
            # Determine length from waveform helper so the empty array is still
            # correctly sized and aligned.
            wf = AwgDriver._compute_waveform(chirp_config)
            if wf is None:
                return None
            return np.zeros(len(wf[0]), dtype=np.uint32)

        n_samples = len(arrays[0])
        packed = np.zeros(n_samples, dtype=np.uint32)
        for ch_idx, arr in zip(indices, arrays):
            packed |= arr.astype(np.uint32) << ch_idx

        return packed

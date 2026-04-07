# PythonProcess Push Model Refactor

**Status: COMPLETED** (cmakemigration branch, 2026-04-06)

This document describes the architectural change from a C++ timer-polling model
to a Python push model for FTMW waveform acquisition, and the companion
selective proxy injection system that establishes the pattern for future
push-style hardware types. It is retained as a reference for documentation
and future development.

---

## Motivation

PythonFtmwScope needs to receive waveform data from Python. The initial
implementation used a C++ timer that synchronously polls Python via
`sendRequest({"method": "read_waveform"})` each tick. This is unnatural —
real instruments produce data at their own rate, and the Python script
should control acquisition timing (hardware triggers, vendor SDK callbacks,
etc.).

A better approach: inject a `self.scope` proxy (like `self.comm`,
`self.settings`, `self.log`) that lets Python **push** waveform data to
C++ when it's ready. This is architecturally consistent with the existing
proxy pattern — `self.log.log(msg)` already pushes unsolicited messages.

---

## Implemented Architecture

### New message type: waveform push

Python sends unsolicited waveform messages (like log messages):
```json
{"waveform": "<base64 raw bytes>", "shots": 1}
```

### C++ side: event-driven read loop (PythonProcess)

`QProcess::readyReadStandardOutput` is connected to `onReadyRead()`, which
reads all available data into `d_readBuf`, splits on `\n`, and routes each
complete JSON line:

| Message field | Action |
|---|---|
| `"log"` | emit `logMessage` signal |
| `"waveform"` | base64-decode, emit `waveformReceived(QByteArray, quint64)` |
| `"relay"` | handle relay, write response immediately |
| `"id"` | store in `d_pendingResponse`, emit `responseReady()` |

`sendRequest()` uses a nested `QEventLoop` instead of a blocking poll:
```cpp
QEventLoop loop;
connect(this, &PythonProcess::responseReady, &loop, &QEventLoop::quit);
connect(p_process, &QProcess::finished, &loop, [this, &loop]() {
    onReadyRead();   // drain remaining stdout
    loop.quit();
});
QTimer::singleShot(d_timeoutMs, &loop, &QEventLoop::quit);
loop.exec();
```
This allows relay requests and waveform pushes to be processed while
`sendRequest()` is waiting for a response.

**New signals:**
```cpp
void waveformReceived(const QByteArray &data, quint64 shotCount);
void responseReady();   // internal: wakes sendRequest() event loop
```

**State members added:**
```cpp
QByteArray d_readBuf;           // accumulates partial stdout data
bool d_waitingForResponse;
int d_expectedId;
QJsonObject d_pendingResponse;
```

`readLineJson()` and `readResponseForId()` were removed entirely.

### Selective proxy injection

`PythonProcess` exposes `setEnabledProxies(const QStringList &proxies)`,
which must be called after `initPythonProcess()` and before `start()` (i.e.
before `testConnection()`). The proxy list is forwarded to Python in the
`_init` message:
```json
{"method": "_init", "key": "...", "model": "...", "proxies": ["scope"]}
```

On the Python side, `python_hw_host.py` maintains a factory map and only
instantiates proxies that appear in the list:
```python
_OPTIONAL_PROXY_FACTORIES = {
    "scope": ScopeProxy,
}
for name in request.get("proxies", []):
    factory = _OPTIONAL_PROXY_FACTORIES.get(name)
    if factory:
        setattr(user_obj, name, factory())
```

The three standard proxies (`self.comm`, `self.settings`, `self.log`) are
always injected. Optional proxies are hardware-type-specific and must be
explicitly requested by the trampoline class.

**Adding a new push-style proxy** (future reference):
1. Implement the proxy class in `python_hw_host.py`
2. Add an entry to `_OPTIONAL_PROXY_FACTORIES`
3. Call `pu_process->setEnabledProxies({"your_proxy"})` in the trampoline's
   `initialize()` after `initPythonProcess()`
4. Connect `PythonProcess`'s corresponding signal in the trampoline

### ScopeProxy (python_hw_host.py)

```python
class ScopeProxy:
    def emit_shot(self, raw_bytes, shots=1):
        """Push raw waveform data to the C++ WaveformBuffer.

        Args:
            raw_bytes (bytes): Raw waveform data matching the configured
                format (record_length × bytes_per_point × num_records).
            shots (int): Number of shots represented (1 for single-shot,
                N for pre-accumulated data).
        """
        b64 = base64.b64encode(bytes(raw_bytes)).decode('ascii')
        _send_json({"waveform": b64, "shots": shots})
```

Thread-safe: `_send_json` holds `_stdout_lock`.

### PythonFtmwScope changes

- `QTimer *p_acqTimer` and all timer logic removed
- `readWaveform()` is a no-op (satisfies pure virtual in `FtmwScope`)
- `initialize()` calls `setEnabledProxies({"scope"})` then connects
  `waveformReceived` → `onWaveformReceived()` which calls `emitShot(data)`
- `beginAcquisition()`: sends `{"method": "begin_acquisition"}` IPC only
- `endAcquisition()`: sends `{"method": "end_acquisition"}` IPC only
- `BC::Key::PythonFtmwScope::interval` setting removed

### Template script (python_ftmwscope_template.py)

`read_waveform()` removed. Acquisition is now push-driven:

```python
def begin_acquisition(self):
    self._acquiring = True
    self._acq_thread = threading.Thread(target=self._acquisition_loop, daemon=True)
    self._acq_thread.start()

def end_acquisition(self):
    self._acquiring = False
    if self._acq_thread is not None:
        self._acq_thread.join(timeout=5.0)
        self._acq_thread = None

def _acquisition_loop(self):
    while self._acquiring:
        raw = self._generate_virtual_waveform()
        self.scope.emit_shot(raw)
        time.sleep(0.2)
```

The acquisition loop runs in a daemon thread. The main thread stays free
to receive `end_acquisition` and other IPC calls from C++.

---

## Key Design Notes

**Nested event loop reentrancy**: `sendRequest()` uses `QEventLoop::exec()`
which processes events. If a waveform arrives during `sendRequest()`, the
`waveformReceived` signal fires from within the nested loop. This is safe
because `onWaveformReceived()` only calls `emitShot()`, which writes to the
WaveformBuffer and never re-enters `sendRequest()`.

**Partial line buffering**: `readyRead` may deliver partial lines.
`d_readBuf` accumulates data; only `\n`-terminated lines are parsed.

**Thread safety on Python side**: The acquisition thread writes to stdout
via `_send_json` (protected by `_stdout_lock`). The main thread reads stdin
and also writes to stdout (relay responses, method responses). Concurrent
stdout writes are safe; stdin reads are main-thread only.

**Process death during acquisition**: `sendRequest()` connects
`QProcess::finished` to drain remaining stdout and quit the event loop
immediately rather than waiting for the full timeout.

---

## Files Modified

- `src/hardware/python/pythonprocess.h` — added `waveformReceived` signal,
  `responseReady` signal, `onReadyRead` slot, `setEnabledProxies()`,
  `d_readBuf`/`d_waitingForResponse`/`d_expectedId`/`d_pendingResponse`/
  `d_enabledProxies` members; removed `readLineJson()`, `readResponseForId()`
- `src/hardware/python/pythonprocess.cpp` — event-driven read loop, QEventLoop
  in `sendRequest()`, process-death detection, proxies list in `_init` message
- `src/hardware/python/python_hw_host.py` — `ScopeProxy` class,
  `_OPTIONAL_PROXY_FACTORIES` dispatch, conditional proxy injection in `_init`
- `src/hardware/python/pythonftmwscope.h` — removed timer and interval key,
  `readWaveform()` is inline no-op, added `onWaveformReceived` slot
- `src/hardware/python/pythonftmwscope.cpp` — removed timer logic,
  `setEnabledProxies({"scope"})`, `waveformReceived` signal connection
- `src/hardware/python/python_ftmwscope_template.py` — push-model acquisition
  thread replacing `read_waveform()`

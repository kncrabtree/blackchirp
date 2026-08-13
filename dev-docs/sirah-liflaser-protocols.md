# Sirah LIF-laser serial protocols — reference, Cobra audit, and FCU plan

**Status:** working document / plan. Not published product docs — will be
purged before release per `dev-docs/` policy.

**Scope of this document**

1. How the two Sirah drivers plug into Blackchirp's LIF-laser hardware
   contract.
2. A low-level serial-command reference for the **Sirah Cobra** dye-laser
   grating controller, transcribed from the vendor *Programmer's Guide to
   Sirah Dye Lasers* (Chapter 9, "Low Level Control").
3. An audit of the current `SirahCobra` / `sirahprotocol` implementation
   against that reference.
4. A low-level serial-command reference for the **Sirah Autotracker /
   Frequency Conversion Unit (FCU)**, transcribed from the *Autotracker
   Programmer's Guide (Version 2)*, Appendix A, cross-checked against the
   `AT-Driver` LabVIEW library.
5. An audit of the current placeholder `SirahFcu` communication code.
6. A concrete plan for replacing the FCU's borrowed-from-Cobra comm layer
   with the real Autotracker protocol, keeping the goal narrow: a
   lookup-table calibration driving the doubling-crystal motor to a target
   position by hand (no autotracking, no energy servo).

**Primary sources**

- `Programmer's Guide to Sirah Dye Lasers.pdf` (Sirah Control 2.6 →
  Manuals/Programming). Text-native PDF. Chapter 9 = low-level serial
  reference; Chapter 6 = INI/config reference; Chapter 8 = position-mode
  codes.
- `SirahLaserObject.h` (Sirah Control 2.6 → LabVIEW/Laser/DLL Support). The
  vendor C DLL header — authoritative for error codes, motor IDs, and
  position-mode codes.
- `Autotracker Programmer's Guide (Version 2).pdf`. **Photo scan**, no
  reliable embedded text; Appendix A (PDF pages 23–37) is the low-level
  serial reference. Transcribed by careful image reading — see §4.
- `AT-Driver` LabVIEW library (Sirah Control 2.6 →
  LabVIEW/Autotracker/AT-Driver). The VIs are binary; `strings` yields
  control/indicator labels and help text but not the numeric command
  constants. Used to corroborate the *shape* of the protocol (per-motor
  position query, Flow/Wait handshake, ZPRAM register access).

---

## 1. Blackchirp LIF-laser architecture (how the drivers fit)

Two sibling hardware types, both under `src/hardware/core/liflaser/`:

- **`LifLaser`** (base) → **`SirahCobra`**: the tunable dye-laser
  fundamental. Contract works in vacuum wavenumber (cm⁻¹). `setPos(cm⁻¹)`
  / `readPos()` are the driver hooks; the grating sine-bar geometry is
  evaluated in nm, so `posToWavelength()`/`wavelengthToPos()` are the
  nm↔cm⁻¹ boundary.
- **`LifFreqConversionStage`** (base) → **`SirahFcu`**: a doubling-crystal
  node in the conversion topology. `setPosition(localCm1)` (public slot,
  base class) computes/dispatches to the driver hook `setPos(localCm1)`,
  then verifies via `readPos()`. The base owns the verify flag
  (`BC::Key::LifConvStage::verify`) and tolerance
  (`verifyToleranceCm1`). Device identity is pinned: `conversionOp()`
  returns `Op::NHG`, harmonic default overridden to N=2.

Both speak their own RS-232 port. The wire format is factored into
`sirahprotocol.{h,cpp}` (`BC::Sirah::buildCommand()` / `parseStatus()`),
currently **shared by both drivers on the assumption that the FCU speaks
the Cobra's protocol** — an assumption the header itself flags as
unverified. §4–§6 show that assumption is wrong at the command level.

**Calibration on the FCU side is already solved.** `FcuCalibration`
(`data/lif/fcucalibration.{h,cpp}`) is a hardware-free value type with
three schemes:

- `Physical` — Type-I SHG phase-match + sine-bar mechanics (BBO/KDP
  Sellmeier).
- `Polynomial` — imported forward/inverse coefficient lists.
- **`Spline` — imported `(wavelength_nm, position_steps)` point table**
  (GSL monotone Steffen splines, invertible both directions).

The `Spline` scheme **is** the "lookup-table calibration" goal. So the FCU
work is *not* about the tuning math — that already exists and is used by
`SirahFcu::hwReadSettings()`. It is entirely about the **communication
layer**: making `prompt()`, `moveAbsolute()`, `moveRelative()`, and
`readPos()` speak the real Autotracker serial protocol instead of the
Cobra's.

---

## 2. Sirah Cobra low-level serial reference

Source: *Programmer's Guide to Sirah Dye Lasers*, Chapter 9.

### 2.1 Serial parameters

| Param     | Value      |
|-----------|------------|
| Baud rate | 19200      |
| Data bits | 8          |
| Stop bits | 1          |
| Parity    | none       |
| Handshake | none       |

### 2.2 Command frame (host → controller): 13 bytes, fixed

| byte | 0 | 1 | 2 … 10 | 11 | 12 |
|------|---|---|--------|----|----|
| value | `0x3C` (`<`) | cmd code | up to 9 parameter bytes | checksum | `0x3E` (`>`) |

Checksum = 8-bit sum of bytes **0..10** inclusive (start marker + command +
all parameter bytes). The manual's reference snippet computes
`for (i=0;i<11;i++) CheckSum += Command[i];`. Unused parameter bytes are
zero and contribute nothing.

### 2.3 Status ("prompt") message (controller → host): 14 bytes, fixed

| byte | 0 | 1 | 2 | 3 | 4–7 | 8 | 9–12 | 13 |
|------|---|---|---|---|-----|---|------|----|
| field | `0x5B` (`[`) | error | controller status | motor #1 status | motor #1 abs pos (LSB→MSB) | motor #2 status | motor #2 abs pos (LSB→MSB) | `0x5D` (`]`) |

- No checksum on the status frame.
- Positions are 4-byte little-endian signed integers.
- **Error byte** (byte 1) values: 0 none, 1 tx frame error, 2 bad
  direction, 3 bad device #, 4 bad command code, 5 frequency out of range,
  6 ramp length out of range, 7 motor running, 8 motor not running, 9
  acceleration undefined, 10 motor current off, 11 internal, 12 input
  buffer overflow, 13 limit switch during origin search, 14 limit switch
  during normal move, 15 tx checksum mismatch, 16 bad trigger param, 17 bad
  port status param.
- **Controller status byte** (byte 2) bits: bit0 Valid (accel defined),
  bit1 Trigger (input LOW), bit2 Manual (front-panel control), bits4–7 Rev
  (firmware revision code).
- **Motor status byte** (bytes 3, 8) bits: **bit0 Running (1=running)**,
  bit1 Hold current on, bit2 Origin end-stop pressed, bit3 Lower limit
  pressed, bit4 Upper limit pressed.

### 2.4 Command codes (Chapter 9 reference section)

| cmd | name | params | notes |
|-----|------|--------|-------|
| `0x01` | Initialize Device | — | Resets all params + positions to zero. Only after RAM loss. |
| `0x02` | Set Acceleration | start(3B), high(3B), ramp(3B) | pulse Hz / Hz / steps |
| `0x03` | Emergency Stop | — | immediate; may lose steps |
| `0x04` | Decelerating Stop | — | smooth stop, no step loss |
| `0x05` | Multiple Move | Abs1(4B), Abs2(4B) | both motors, absolute |
| `0x06` | **Move Relative** | Dev(1B), Dir(1B), Rel(4B) | Dev 0=both/1=#1/2=#2; Dir 1=CW/2=CCW |
| `0x07` | **Move Absolute** | Dev(1B), Abs(4B) | Dev 1=#1/2=#2 |
| `0x08` | Origin Search | Dev(1B), Dir1(1B), Dir2(1B) | move to calibrated end-stop |
| `0x09` | Motor Free | Dev(1B) | hold current OFF |
| `0x0A` | Motor Hold | Dev(1B) | hold current ON |
| `0x0B` | Trigger Set | — | trigger out HIGH |
| `0x0C` | Trigger Reset | — | trigger out LOW |
| `0x0D` | Prompt on Trigger Enable | — | |
| `0x0E` | Prompt on Trigger Disable | — | |
| `0x0F` | Auto-Prompt Enable | — | status once/second |
| `0x10` | **Auto-Prompt Disable** | — | stop unsolicited status |
| `0x11` | Trigger Count | Count(4B) | edges per trigger event |
| `0x12` | Constant Move Absolute | Dev(1B), Abs(4B) | const speed = start freq |
| `0x13` | Constant Multiple Move | Abs1(4B), Abs2(4B) | |
| `0x14` | Special Wait | — | wait-move → trigger dance |
| `0x15` | Set DAC Port | Dev(1B), Value(4B) | Dev 1=A/2=B; 0x1000=+10V |
| `0x16` | Instant Speed Change | Freq(3B) | change const-speed rate |
| `0x17` | **Request Prompt** | — | one status message |
| `0x18` | Const-speed absolute, variable freq | Dev(1B), Trg(1B), Abs(4B), Freq(3B) | |
| `0x19` | Set Output Ports | P1.5,P1.6,P4.5,P4.6,P4.7 (1B each) | 0 no change/1 reset/2 set |

### 2.5 Config parameters relevant to tuning (Chapter 6, Resonator §)

Sine-bar grating geometry (all factory-calibrated): `GrazingAngle`,
`GrazingGrooves`, `LeverLength`, `LinearOffset`, `AngleOffset`,
`ScrewPitch`, `MotorResolution`, `BacklashSteps` (typical 24000),
`MaximumPosition`. `WavelengthTuning` is an additive fine-tune folded into
`AngleOffset`. These are exactly the fields `SirahCobra` registers in its
`stages` array.

---

## 3. Audit — `SirahCobra` / `sirahprotocol` vs. §2

**Verdict: the Cobra implementation is correct against the manual.** No
protocol bugs found. Details:

### 3.1 Frame construction (`sirahprotocol.cpp`) — correct

- `buildCommand()`: emits `0x3C`, cmd, up to 9 arg bytes at [2..10],
  checksum at [11], `0x3E` at [12]. Checksum accumulates byte0 + byte1 +
  each arg — i.e. the sum of bytes 0..10 since the gap bytes are zero.
  **Matches §2.2 exactly.** Signedness of the `char` accumulator is
  irrelevant: only the low 8 bits of the sum are transmitted.
- `parseStatus()`: requires 14 bytes bracketed by `0x5B`/`0x5D`; unpacks
  error@1, cStatus@2, m1Status@3, m1Pos@4-7 LE, m2Status@8, m2Pos@9-12 LE.
  **Matches §2.3 byte-for-byte**, including little-endian order and the
  absence of a status-frame checksum. Leaves `out` untouched on a
  malformed frame (good).

### 3.2 Command usage (`sirahcobra.cpp`) — correct

- `prompt()` → `0x17` Request Prompt, then `readBytes(14,true)`. ✔
- `testConnection()` → prompt, then `0x10` Auto-Prompt Disable. ✔ Correct
  and important: the driver is request/response, so silencing the 1 Hz
  auto-prompt keeps the read stream from being corrupted by unsolicited
  frames.
- `moveRelative()` → `0x06` with Dev=1 (motor #1), Dir=1 (CW) for
  positive / 2 (CCW) for negative, `|steps|` as 4-byte LE. ✔
- `moveAbsolute()` → `0x07` with Dev=1, target as 4-byte LE (two's
  complement handles the `targetPos - backlash` negative intermediate). ✔
- stop-on-timeout → `0x04` Decelerating Stop (not `0x03` Emergency). ✔
  Correct — avoids step loss.
- Motor-running poll uses `m1Status % 2` = bit0 Running. ✔

### 3.3 Tuning math (`posToWavelength`/`wavelengthToPos`) — consistent

Standard grazing-incidence sine-bar law:
`x = linOff − (pitch/mRes)·pos`; `φ = angOff − asin(x/lLen)`;
`λ_mm = (sin(grazAng) + sin(φ))/grooves`; `λ_nm = λ_mm·1e6`. Dimensionally
consistent (grooves in lines/mm → 1/grooves in mm; ×1e6 → nm) and the
inverse is the correct algebraic inversion. Matches the Chapter-6
parameter set.

### 3.4 Minor notes (not bugs — candidate follow-ups)

- **Diffraction order hardcoded to 1.** The manual exposes `GrazingOrder`
  (usually 1). Fine for typical use; a non-first-order grating would need
  an `m` factor.
- **Fixed 10-step deadband** in `setPos()` (`if(qAbs(delta) < 10) return;`)
  is arbitrary in step units. The FCU driver already improved on this by
  gating on the verify tolerance in wavenumber; the Cobra could adopt the
  same pattern, but it is cosmetic.
- **`WavelengthTuning`/`AngleOffset` fine-tune** additive term from the
  manual is not surfaced as a setting; `AngleOffset` alone is used. Only
  matters if a user wants a software wavelength trim.
- `readBytes(14,true)` assumes the whole frame arrives within the 200 ms
  comm timeout and that no stray auto-prompt frame is interleaved — safe
  given the `0x10` disable in `testConnection()`, but it does **not**
  re-disable auto-prompt after a device power-cycle mid-session. Low risk.

---

## 4. Sirah Autotracker / FCU low-level serial reference

Source: *Autotracker Programmer's Guide (Version 2)*, Appendix A (PDF
pages 23–37), corroborated by the `AT-Driver` LabVIEW library. Full
byte-level transcription (all ~38 commands + error table) lives in
`scratchpad/autotracker_lowlevel_ref.md`; the subset needed for manual
position control is reproduced here.

**The Autotracker does NOT speak the Cobra protocol.** Only the broad shape
(start byte + code/ID byte + payload + checksum at index 11) is shared.
Everything else differs — see §4.5.

### 4.1 Serial parameters

Identical to the Cobra: **19200** baud, **8** data bits, **1** stop bit,
**no** parity, **no** handshake.

### 4.2 Command frame (host → Autotracker): 12 bytes, no end marker

| byte | 0 | 1 | 2 | 3 … 10 | 11 |
|------|---|---|---|--------|----|
| value | `0x3E` (`>`) | `0x00` reserved | cmd code | up to 8 data bytes | checksum |

Checksum = 8-bit sum of bytes **0..10**, stored in byte 11. **The frame
ends at byte 11 — there is no terminator byte.** Note byte 1 is a
reserved-must-be-zero byte and the **command code is at byte 2**, not
byte 1.

### 4.3 Output/response frame (Autotracker → host): 12 bytes, no end marker

| byte | 0 | 1 | 2 | 3 … 10 | 11 |
|------|---|---|---|--------|----|
| field | `0x3C` (`<`) | Adr/Status | ID | 8 payload bytes | checksum |

- **Byte 1 (Adr/Status):** low bits = internal address; the manual calls
  the MSB an error flag. **Bench reality (§4.6): this byte reads `0x81` on
  *successful* Get Position and Goto replies.** The likely explanation:
  the `Error` command drains a **queue** that "stores errors until they are
  read," so the set MSB is probably a *stale, undrained* error from earlier
  in the session rather than a live indicator — the notebook never issued
  `Error` (`0x03`) to clear it. **Working hypothesis (verify on the bench):
  drain the error queue at connection (§6.2), after which the MSB should
  track live error state and become a valid fast-path check.** Until that
  is confirmed, treat authoritative error detection as the `Error` command,
  not this bit.
- **Byte 2 (ID):** a *self-describing* tag saying what the payload is — the
  format is generic, not a fixed "motor #1 status / position" layout. A
  plain acknowledgment ("standard status message") has ID `= 0x00` and
  don't-care payload.
- **There is no documented motor-running/busy bit anywhere.** Move
  completion is handled by the command's own `Wait` flag (see Goto), not a
  polled status bit.

### 4.4 Commands needed for manual position control

Positions are **24-bit unsigned (0..0xFFFFFF), transmitted MSB-first**
(big-endian, 3 bytes) — not 32-bit little-endian.

| cmd | name | command payload | response |
|-----|------|-----------------|----------|
| `0x01` | Idle | — | standard ack (handshake/no-op) |
| `0x02` | Identify | — | ID `0x01`: byte3 `0x04`, ROM ver@4, ROM rev@5 |
| `0x03` | Error | — | ID `0x02`: Error #0..#7 @3-10 (error stack) |
| `0x17` | **Get Position** | Motor@3 (1..3) | ID `0x0b`: Motor@3, Pos 24-bit MSB-first @4-6 |
| `0x22` | **Goto Position** | Motor@3, Wait@4, Rel@5, Pos 24-bit MSB-first @6-8 | standard ack |
| `0x23` | Origin Search | Motor@3, Wait@4 | standard ack |
| `0x1F` | Set Command Move Params | Motor@3, StartF(2B MSB@4)@4-5, HighF@6-7, Ramp@8-9 | standard ack |
| `0x15` | Get Command Move Params | Motor@3 | ID `0x08`: Motor, StartF, HighF, Ramp |
| `0x26` | Set Extended Params | ManInc@3-4, MaxPos@5-8 | standard ack |
| `0x5A` / `0x5B` | Read / Write motor-controller Register | Motor@3, Reg@4[, Data@5] | ID `0x06` (read) / ack |
| `0x5C` / `0x5D` | Write / Read ZPRAM | Adr@3, Data#0-3 / — | ack / ID `0x13` |

**Goto Position (`0x22`) flags — the crux of the move loop:**
- `Wait` (byte 4): **`0` = controller waits for the motor to finish before
  sending its ack; `1` = ack immediately.** With `Wait=0` the single
  response arrives only when the move is complete, so a synchronous driver
  needs **no busy-poll and no stop command** — it just reads the ack with a
  generous timeout.
- `Rel` (byte 5): `0` = absolute target, `1` = relative to current
  position. Backlash correction can therefore be a relative Goto.

**Motor moves are 1-indexed (1..3).** The FCU is one motor on the
Autotracker; a motor-number setting (default 1) is needed rather than the
Cobra's hardcoded `Dev=0x01`.

**Error codes (Appendix A, p-37):** 0 none, 1 start char, 2 checksum, 3
watchdog, 4 command code, 5 command break (also raised when Stop Timer
aborts tracking), 6 stack overflow, 7 stack underflow, 8 bad device, 9
device inactive, 10 bad register, 14 bad motor number, 15 motor inactive,
16 bad motor parameter (1,2,3 expected), 26 timeout occurred, 32 serial
buffer overflow (plus trace/gain/trigger codes irrelevant to position
control). Full table in the scratchpad transcription.

**Scan-quality caveat.** A few table rows in the photographed manual print
the command byte shifted one column left (a dropped reserved-`0x00` cell) —
the transcription flags `Error` (`0x03`), `Sample Dual` (`0x2c`), and
`Write Register` (`0x5b`), plus a possibly-missing address byte on `Read
ZPRAM`. **None of these affect the manual-position-control path**: `Get
Position` (`0x17`), `Goto Position` (`0x22`), `Identify` (`0x02`), and
`Set Command Move Parameters` (`0x1F`) are transcribed cleanly. Confirm any
of the flagged commands against a live unit before relying on them.

### 4.5 How the Autotracker differs from the Cobra (§2) — summary

| aspect | Cobra | Autotracker |
|---|---|---|
| command frame length | 13 bytes | **12 bytes** |
| response frame length | 14 bytes | **12 bytes** |
| end marker | `0x3E` / `0x5D` | **none** |
| command start byte | `0x3C` | **`0x3E`** |
| response start byte | `0x5B` | **`0x3C`** |
| command-code byte | byte 1 | **byte 2** (byte 1 reserved `0x00`) |
| response semantics | fixed positional (err, m1/m2 status+pos) | **generic ID-tagged payload** |
| position encoding | 32-bit little-endian, signed | **24-bit big-endian, unsigned** |
| motor count | 2 | **3** |
| move completion | poll motor-status bit0 (Running) | **`Wait=0` ack-on-complete** (no status bit) |
| position query | from the status/prompt frame | **dedicated `Get Position` (`0x17`)** |
| stop command | `0x04` decel / `0x03` emergency | **none documented** (rely on `Wait` semantics) |
| auto-prompt | `0x0F`/`0x10` enable/disable | **none** (no unsolicited stream to silence) |
| checksum span | bytes 0..10 | bytes 0..10 (same), but byte 11 is the *last* byte |

Corroboration from the `AT-Driver` LabVIEW VIs (`strings`): `AT Send
Command.vi`/`AT Communicate.vi` are the framing primitives; `AT Motor Get
Position.vi` (input "Motor Nr") is the dedicated position query; `AT Motor
Goto Position.vi`/`AT Motor Origin Search.vi` expose the `Flow`
(Wait/Continue) handshake; `AT ZPRAM Read/Write.vi` and register VIs match
`0x5A`–`0x5D`. Trace/energy/tracking commands exist but are **out of scope**
for manual position control.

### 4.6 Bench confirmation — Python communication record

Source: `Downloads/FCUCtrl.ipynb` — a pyserial scratch driver run against a
real FCU (COM3/COM4, `serial.Serial(baudrate=19200, bytesize=EIGHTBITS,
parity=PARITY_NONE, stopbits=STOPBITS_ONE)`). This confirms the transcribed
protocol against live hardware. Two captured exchanges:

**Get Position (`0x17`):**
```
Sending  : 3E 00 17 01 00 00 00 00 00 00 00 56
Received : 3C 81 0B 01 FA E9 BD 00 00 00 00 69
```
Confirms, byte-for-byte: `0x3E` command start, `0x00` reserved byte, cmd
`0x17` at byte 2, Motor=`01` at byte 3, checksum `sum(bytes[0:11]) & 0xFF`
= `0x56`. Reply: `0x3C` start, ID=`0x0B` at byte 2, Motor=`01` at byte 3,
position `FA E9 BD` at bytes 4-6 decoded **big-endian**
(`b4<<16 | b5<<8 | b6` = 16,443,837 steps). ✔ matches §4.4 exactly.

**Goto Position (`0x22`):**
```
Sending  : 3E 00 22 01 00 00 FA E9 BD 00 00 01
Received : 3C 81 00 01 FB 15 09 00 00 00 00 D7
```
Confirms: cmd `0x22`@2, Motor=`01`@3, **Wait=`00`@4, Rel=`00`@5** (absolute),
24-bit big-endian target `FA E9 BD`@6-8. Reply ID=`0x00` (standard ack). ✔

**Corrections/notes this record forces on §5–§6:**
- **`Adr/Status` = `0x81` on both successful replies** → the byte-1 MSB is
  set here. Most likely a **stale queued error** the notebook never drained
  (the `Error` command reads a queue that persists until read, and the
  notebook never called it), not proof the bit is meaningless. Plan
  response: drain the error queue at connect, then re-test whether the MSB
  clears and tracks live state (§4.3, §6.2). Until confirmed, do not rely on
  the bit alone.
- The `0x00` ack's payload carried bytes (`FB 15 09`) that the notebook's
  reused parser printed as a "position", but with ID=`0x00` the payload is
  not a defined position field and disagreed with the target — **do not
  interpret the Goto ack payload as a position; confirm with a follow-up
  `Get Position`.**
- The notebook uses `wait=0` = "wait for motor finish" and a 6 s
  wait-for-12-bytes poll on top of a 1 s serial timeout — consistent with
  the §6 recommendation to read the Goto ack with a *generous* timeout.
- **Ignore the trailing notebook cells** (`MoveRelative`,
  `AutoPromptDisable/Enable`): their outputs use Cobra codes (`0x06`,
  `0x0F`, `0x10`) and Cobra-style `>`-terminated frames — stale scratch
  copied from Cobra experimentation, not valid Autotracker commands. The
  notebook's `Error()` helper also places the code at byte 1 (the
  scanned-manual anomaly reading) and is **unverified** (no captured
  response). The working, confirmed helpers are `GetPosition`,
  `GotoPosition`, `Identify`, `Idle`, `LED`.

---

## 5. Audit — current placeholder `SirahFcu` comm code

`SirahFcu` currently **reuses the Cobra protocol wholesale**:

- `prompt()` sends `BC::Sirah::buildCommand(0x17)` (Cobra Request Prompt)
  and parses a 14-byte Cobra status frame via `BC::Sirah::parseStatus`.
- `moveRelative()`/`moveAbsolute()` build Cobra `0x06`/`0x07` frames with
  Dev=1, and stop with `0x04`.
- `testConnection()` disables auto-prompt with `0x10`.
- Motor-running poll uses `d_status.m1Status % 2`.

This is a **placeholder** (so flagged in `sirahprotocol.h` and the
`SirahFcu` class comment). Given §4's corroborated evidence that the
Autotracker uses different framing, a per-motor Get-Position command, and a
Flow/Wait ack, **most or all of these byte-level assumptions are expected
to be wrong.** They must be re-derived from §4 before the FCU can talk to
real hardware. What is *sound* and should be preserved:

- The **calibration path** (`hwReadSettings()` assembling an
  `FcuCalibration`, the Spline lookup-table scheme, the NaN-guarded
  `setPos()`/`readPos()` conversion boundary). No changes needed there.
- The **base-class contract** (`setPos(localCm1)`/`readPos()` in cm⁻¹, the
  verify-tolerance redundant-move skip, backlash handling structure).
- The **backlash approach** (absolute move to `target − backlash` then
  relative move by `backlash`), assuming the Autotracker exposes
  equivalent absolute + relative moves (the LabVIEW `Position Mode`
  absolute/relative input says it does).

---

## 6. Plan — bring `SirahFcu` onto the real Autotracker protocol

Guiding constraint (from the user): **lookup-table calibration + manual
absolute move to target. No autotracking, no energy servo, no trace
readout.** So this is a comms swap, not a feature build. §4 shows the
Autotracker protocol is genuinely different from the Cobra, which makes the
swap larger than a code-table change — but it also makes the move loop
*simpler* (the `Wait=0` ack-on-complete removes the busy-poll and the need
for a stop command).

### 6.1 New wire layer — do not extend `sirahprotocol`

The Autotracker frame differs in length, markers, code position, response
semantics, and position encoding (§4.5). Do **not** try to generalize
`sirahprotocol`'s `buildCommand`/`parseStatus` over both. Instead:

- Add `autotrackerprotocol.{h,cpp}` (namespace `BC::Autotracker`) with:
  - `QByteArray buildCommand(quint8 cmd, const QByteArray &data = {})` —
    emits `0x3E`, `0x00`, cmd@2, data@3.., checksum@11 (sum of bytes
    0..10); pads to 12 bytes; **no terminator.**
  - `struct Response { quint8 adrStatus; quint8 id; QByteArray payload; }`
    and `bool parseResponse(const QByteArray &resp, Response &out)` —
    requires 12 bytes starting `0x3C`, verifies the checksum, extracts
    `id` and the 8 payload bytes. **Do not derive an error flag from
    `adrStatus & 0x80`** — the bench shows that bit set (`0x81`) on normal
    successful replies (§4.6). Treat error detection as a separate step via
    the `Error` command (`0x03`), not a byte-1 bit test.
  - Helpers `packPos24(quint32)` / `unpackPos24(...)` for the MSB-first
    24-bit position field.
- Consider renaming `sirahprotocol` → `cobraprotocol` so its Cobra-only
  frame constants stop implying FCU applicability (the current file/class
  comments already hedge on this). Optional but honest.

### 6.2 Rework `sirahfcu.cpp` comm primitives

The current `d_status`/`BC::Sirah::Status` usage is replaced by
`BC::Autotracker::Response` parsing. A **motor-number** member (from a new
setting, default 1) replaces the hardcoded `Dev=0x01`.

1. **`readPos()` / position read.** Send `Get Position` (`0x17`, Motor@3),
   read the 12-byte reply, verify ID `= 0x0b`, decode the 24-bit MSB-first
   position from payload bytes, and feed it to
   `d_calibration.posToWavelength()` → cm⁻¹ exactly as today. For error
   detection, poll `Error` (`0x03`) and surface any nonzero codes via
   `hwError` — **not** the byte-1 MSB, which is set on success (§4.6).
   Verify the reply ID and validate the echoed motor number. Return the
   negative sentinel on any comm/parse failure (base class treats `<0` as a
   hard failure).
2. **`moveAbsolute(targetPos)`.** Send `Goto Position` (`0x22`) with
   Motor, **`Wait=0` (ack on completion)**, `Rel=0`, and the 24-bit target.
   Read the single ack with a **long comm timeout** (a full-travel move can
   take seconds — the default 200 ms is far too short; use a dedicated
   longer read or a bounded retry). No busy-poll loop, no `m1Status` bit, no
   stop command. Success = a well-formed `0x00` ack received (do not parse
   its payload as a position — §4.6 — confirm via a follow-up `Get
   Position` / the base-class verify step); optionally poll `Error` (`0x03`)
   to catch a rejected move.
   - *Fallback if `Wait=0` proves unreliable on the bench:* send `Wait=1`
     (immediate ack) then poll `Get Position` until it stops changing
     across a couple of reads (there is no running bit to test).
3. **`moveRelative(steps)`.** Send `Goto Position` (`0x22`) with `Rel=1`
   and the signed step count encoded per §4 (confirm the controller's
   relative-sign convention on the bench). Used for the backlash
   correction move.
4. **Remove the stop-on-timeout `0x04` path.** With `Wait=0` there is no
   in-flight state to stop; a failed move is a failed/absent ack, reported
   via `hardwareFailure()`. (If the bench forces the `Wait=1`+poll
   fallback, a timeout there should still just report failure — the
   Autotracker has no documented decel-stop command.)
5. **`testConnection()`.** Replace the Cobra `0x10` auto-prompt-disable
   (there is no auto-prompt on the Autotracker) with an `Identify` (`0x02`)
   or `Idle` (`0x01`) round-trip; `Identify` also gives ROM ver/rev to log.
   **Then drain the error queue:** the `Error` command (`0x03`) reads a
   queue that persists across sessions until read, so issue it at connect
   (and, if the returned stack is nonempty, repeat until it reports no
   error / stack-underflow) to clear any stale codes — otherwise the very
   first reply's `Adr/Status` MSB may reflect an old error (§4.6). Log
   any drained codes at Debug. After draining, read position once, as
   today, and (bench task) confirm whether the MSB now reads clear on a
   clean reply — if so, the driver can use it as a fast error check and
   only fall back to `Error` (`0x03`) to fetch the specific code.
6. **`initialize()` / move parameters.** The placeholder never programs
   acceleration. For real moves, push `Set Command Move Parameters`
   (`0x1F`: StartF/HighF/Ramp) from the registered `sStart`/`sHigh`/`sRamp`
   settings during init, and optionally `Set Extended Parameters` (`0x26`)
   for the max-position clamp. Confirm the controller has valid accel
   defined before the first move (error 9-equivalent on the Cobra side; the
   Autotracker will reject moves with undefined accel).

### 6.3 Backlash under unsigned 24-bit positions — revisit

The shared backlash idiom (absolute move to `target − backlash`, then a
relative move by `backlash`) was written for the Cobra's **signed** 32-bit
position. Autotracker positions are **unsigned 24-bit**, so
`target − backlash` can underflow below 0. Options:
- Clamp the approach target at 0 (and skip the pre-move if
  `target < backlash`), or
- Do the whole approach with relative Gotos (`Rel=1`) so the intermediate
  is never sent as an absolute negative.
Either way, keep the existing "same-direction, small-delta → single
relative move" optimization from `setPos()`; only the absolute-underflow
edge changes.

### 6.4 Settings

- **Primary calibration path is the `Spline` scheme** (import a
  `(wavelength_nm, position_steps)` table) — that is the user's
  lookup-table goal, and it already works end-to-end in
  `hwReadSettings()`. Verify the `REGISTER_HARDWARE_ARRAY_SCHEMA` CSV
  import round-trips a real table into a valid `FcuCalibration`.
- Add a **motor-number** setting (default 1, range 1..3).
- Comm defaults stay 19200/8/N/1 (same as the Cobra); the empty
  term-char is still correct (binary protocol, no terminator).
- The `sMax` default (3,300,000) fits the 24-bit range (max 16,777,215);
  keep a **max-position clamp** so an out-of-range calibration result
  fails cleanly instead of wrapping or driving into an end-stop.

### 6.5 Validation

- **Unit tests:** `tst_autotrackerprotocol.cpp` — round-trip
  `buildCommand`/`parseResponse`, checksum correctness, 24-bit MSB-first
  pack/unpack, error-bit detection. Mirror any existing sirah protocol
  test. `FcuCalibration` spline round-trip if not already covered.
- **Bench:** `Identify` handshake; **error-queue drain + MSB behavior**
  (after draining via `Error` `0x03`, does `Adr/Status` read clear on a
  clean reply and set only on a real error? — decides whether the MSB is a
  usable fast error check); a known wavelength → Goto → `Get Position`
  readback within `verifyToleranceCm1`; backlash direction correctness;
  `Wait=0` timing (does the ack really block until the move completes, and
  within a sane timeout?). The base-class `verifyMove` mechanism is the
  built-in acceptance check.

### 6.6 Effort estimate

New protocol layer (~120 LOC + test), `sirahfcu.cpp` comm rewrite
(~150 LOC touched), one new setting, no base-class or calibration changes.
Roughly a day of focused work plus a bench pass on real hardware to nail
down the two genuinely uncertain points: the `Wait=0` completion timing and
the relative-move sign convention.

---

## Appendix — file map

| Concern | File |
|---|---|
| Cobra driver | `src/hardware/core/liflaser/sirahcobra.{h,cpp}` |
| FCU driver (placeholder comms) | `src/hardware/core/liflaser/sirahfcu.{h,cpp}` |
| Shared (Cobra) wire layer | `src/hardware/core/liflaser/sirahprotocol.{h,cpp}` |
| FCU calibration value type | `src/data/lif/fcucalibration.{h,cpp}` |
| Laser base contract | `src/hardware/core/liflaser/liflaser.{h,cpp}` |
| Conversion-stage base contract | `src/hardware/core/liflaser/liffreqconversionstage.{h,cpp}` |
| Cobra vendor low-level ref | *Programmer's Guide to Sirah Dye Lasers*, Ch. 9 |
| FCU vendor low-level ref | *Autotracker Programmer's Guide (Version 2)*, Appendix A |
| FCU LabVIEW reference impl | `LabVIEW/Autotracker/AT-Driver/` |

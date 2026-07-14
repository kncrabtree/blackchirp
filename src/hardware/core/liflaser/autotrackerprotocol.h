#ifndef AUTOTRACKERPROTOCOL_H
#define AUTOTRACKERPROTOCOL_H

#include <QByteArray>
#include <QString>

/*!
 * \file autotrackerprotocol.h
 * \brief Pure, stateless wire-format layer for the Sirah Autotracker /
 *        Frequency Conversion Unit (FCU) binary serial protocol.
 *
 * The Autotracker is a separate Sirah product from the Cobra dye-laser
 * grating controller (BC::Sirah, sirahprotocol.h) and speaks a genuinely
 * different binary protocol on its own RS232 port -- the two share only the
 * broad shape of a framed command/response exchange with a trailing
 * checksum. Do not assume interchangeability between the two namespaces.
 *
 * <b>Command frame (host -> Autotracker): 12 bytes, no end marker</b>
 *
 * | byte | 0      | 1          | 2   | 3 .. 10        | 11       |
 * |------|--------|------------|-----|-----------------|----------|
 * | value| 0x3E   | 0x00       | cmd | up to 8 data    | checksum |
 * |      | ('>')  | (reserved) |     | bytes, zero-pad |          |
 *
 * The checksum is the 8-bit sum of bytes 0..10 inclusive, stored in byte 11.
 * <b>There is no terminator byte</b> -- the frame ends at byte 11. This
 * differs from the Cobra frame in every particular: 13 bytes vs. 12, a
 * `0x3E` start marker vs. the Cobra's `0x3C`, a `0x3E` end marker on the
 * Cobra with none here, and the command code sitting at byte 2 (byte 1 is a
 * reserved, must-be-zero byte) rather than byte 1.
 *
 * <b>Response frame (Autotracker -> host): 12 bytes, no end marker</b>
 *
 * | byte | 0     | 1          | 2  | 3 .. 10       | 11       |
 * |------|-------|------------|----|-----------------|----------|
 * | value| 0x3C  | Adr/Status | ID | 8 payload bytes | checksum |
 * |      | ('<') |            |    |                 |          |
 *
 * Byte 1 (Adr/Status) carries the device's internal bus address in its low
 * bits; the vendor manual calls the MSB an error flag, but bench testing
 * shows that bit set (0x81) on ordinary *successful* replies. Do not derive
 * a per-reply error flag from this byte. Error state is queried separately
 * with the Error command (0x03), which reads a small queue of stacked error
 * codes that persists until read -- drain it once at connection time so a
 * stale queued error from an earlier session is not mistaken for a live
 * fault later.
 *
 * Byte 2 (ID) is a self-describing tag identifying the shape of the 8-byte
 * payload; a plain acknowledgment (no interesting payload) carries ID
 * 0x00. There is no documented motor-running/busy status bit anywhere in
 * this protocol -- move completion is signalled by the Goto Position
 * command's own Wait flag (see below), not by polling a status word.
 *
 * <b>Position encoding</b>
 *
 * Motor positions are transmitted as a 24-bit *unsigned* integer, MSB
 * first (big-endian) -- not the Cobra's 32-bit little-endian signed
 * encoding. packPos24()/unpackPos24() are the corresponding pack/unpack
 * helpers.
 *
 * <b>Command codes used by this driver</b>
 *
 * | cmd  | name                       | command payload                                    | response |
 * |------|----------------------------|-----------------------------------------------------|----------|
 * | 0x02 | Identify                   | --                                                  | ID 0x01: byte3 0x04 (tag), ROM version@4, ROM revision@5 |
 * | 0x03 | Error                      | --                                                  | ID 0x02: up to 8 stacked error codes @3-10 |
 * | 0x17 | Get Position               | Motor@3 (1..3)                                      | ID 0x0b: Motor@3 (echoed), Pos24 MSB-first @4-6 |
 * | 0x1F | Set Command Move Parameters| Motor@3, StartF MSB@4/LSB@5, HighF MSB@6/LSB@7, Ramp MSB@8/LSB@9 | standard ack (ID 0x00) |
 * | 0x22 | Goto Position              | Motor@3, Wait@4, Rel@5, Pos24 MSB-first @6-8        | standard ack (ID 0x00) |
 *
 * For Goto Position, Wait=0 means the controller withholds its
 * acknowledgment until the move physically completes (so a synchronous
 * caller needs no busy-poll loop and no stop command -- just a read with a
 * generous timeout), while Wait=1 acknowledges immediately. Rel=0 addresses
 * an absolute target; Rel=1 addresses a position relative to the motor's
 * current position.
 *
 * Motor numbers are 1-indexed (1..3); the Autotracker addresses up to
 * three motors, unlike the Cobra's fixed two.
 *
 * <b>Full Autotracker command set (name -> code)</b>
 *
 * This driver issues only the five commands detailed above; the remaining
 * commands are listed here for reference so a maintainer need not re-derive
 * them from the scanned vendor manual (Autotracker Programmer's Guide
 * Version 2, Appendix A). Codes not listed (0x07-0x14, 0x18-0x1a,
 * 0x1d-0x1e, 0x21, 0x24, 0x27, 0x2b, 0x30-0x31, 0x34-0x4f, 0x52, 0x55,
 * 0x59) are reserved or undocumented in that appendix.
 *
 * | code | command                              | code | command                              |
 * |------|--------------------------------------|------|--------------------------------------|
 * | 0x01 | Idle                                 | 0x2a | Set Trigger                          |
 * | 0x02 | Identify                             | 0x2c | Sample Dual                          |
 * | 0x03 | Error                                | 0x2d | Get Peak-to-Peak                     |
 * | 0x04 | LED                                  | 0x2e | Get Gain                             |
 * | 0x05 | Set Timer                            | 0x2f | Set Gain                             |
 * | 0x06 | Stop Timer                           | 0x32 | Get External Input                   |
 * | 0x15 | Get Command Move Parameters          | 0x33 | Set External Output                  |
 * | 0x16 | Get Manual Move Parameters           | 0x50 | Get Sample Parameter                 |
 * | 0x17 | Get Position                         | 0x51 | Get Tracking Parameter               |
 * | 0x1b | Get Home                             | 0x53 | Set Sample Parameter                 |
 * | 0x1c | Get Extended Parameters              | 0x54 | Set Tracking Parameter               |
 * | 0x1f | Set Command Move Parameters          | 0x56 | Control Autotracking                 |
 * | 0x20 | Set Manual Move Parameters           | 0x57 | Get Autotracking Timeout Parameter   |
 * | 0x22 | Goto Position                        | 0x58 | Set Autotracking Timeout Parameter   |
 * | 0x23 | Origin Search                        | 0x5a | Read Register                        |
 * | 0x25 | Set Home                             | 0x5b | Write Register                       |
 * | 0x26 | Set Extended Parameters              | 0x5c | Write ZPRAM                          |
 * | 0x28 | Get Trigger                          | 0x5d | Read ZPRAM                           |
 * | 0x29 | Get Wave                             |      |                                      |
 */
namespace BC::Autotracker {

/*!
 * \brief Parsed 12-byte response frame from an Autotracker-protocol device.
 */
struct Response {
    quint8 adrStatus{0}; ///< Byte 1: bus address (low bits) -- NOT a reliable per-reply error flag; see file doc.
    quint8 id{0};        ///< Byte 2: self-describing tag for the shape of \c payload.
    QByteArray payload;  ///< Bytes 3..10 (8 bytes), interpretation depends on \c id.
};

/*!
 * \brief Build a 12-byte binary command frame.
 *
 * Byte 0 is the start marker (0x3E), byte 1 is the reserved zero byte,
 * byte 2 is \a cmd, bytes 3..10 hold up to 8 bytes of \a data (zero-padded
 * if shorter), and byte 11 is the 8-bit checksum of bytes 0..10. There is
 * no terminator byte. Data beyond the first 8 bytes of \a data is silently
 * dropped.
 */
QByteArray buildCommand(quint8 cmd, const QByteArray &data = {});

/*!
 * \brief Parse a 12-byte response frame into \a out.
 *
 * Requires \a resp to be exactly 12 bytes, start with 0x3C, and carry a
 * valid checksum in byte 11 (the 8-bit sum of bytes 0..10). Does *not*
 * inspect the Adr/Status error-flag bit -- see the file-level doc comment
 * for why that bit is not a usable per-reply error indicator.
 *
 * \return \c true and fills \a out on a well-formed frame; \c false and
 * leaves \a out unmodified otherwise.
 */
bool parseResponse(const QByteArray &resp, Response &out);

/*!
 * \brief Pack \a pos as a 3-byte, big-endian (MSB-first) 24-bit field.
 *
 * The low 24 bits of \a pos are transmitted; any bits above bit 23 are
 * discarded.
 */
QByteArray packPos24(quint32 pos);

/*!
 * \brief Unpack a 3-byte, big-endian (MSB-first) 24-bit field starting at
 *        \a offset within \a b.
 *
 * \return The decoded value, or 0 if \a b does not hold at least 3 bytes
 * starting at \a offset.
 */
quint32 unpackPos24(const QByteArray &b, int offset = 0);

/*!
 * \brief Human-readable message for an Autotracker error \a code, per the
 *        Autotracker Programmer's Guide Appendix A error table.
 *
 * \return A message including the numeric code for a \a code outside the
 * documented 0..32 range.
 */
QString errorString(quint8 code);

}

#endif // AUTOTRACKERPROTOCOL_H

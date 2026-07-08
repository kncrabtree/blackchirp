#ifndef SIRAHPROTOCOL_H
#define SIRAHPROTOCOL_H

#include <QByteArray>

/*!
 * \file sirahprotocol.h
 * \brief Pure, stateless wire-format layer shared by the Sirah grating
 *        (SirahCobra) and Sirah frequency-conversion (SirahFcu) drivers.
 *
 * Both units are believed (pending hardware confirmation) to speak an
 * identical binary command/status protocol over their own RS232 port. Only
 * that wire format is shared here: frame construction (buildCommand()) and
 * status-frame parsing (parseStatus()). The comm-driving loops (prompt(),
 * the relative/absolute move state machines) and the sine-bar tuning math
 * (position <-> wavelength) are deliberately left duplicated in each
 * driver, because the two units' command sets are not yet hardware-verified
 * to be identical end to end. If bench testing on the FCU confirms the move
 * loops and tuning geometry are truly interchangeable with the Cobra's,
 * consolidating them here is a natural follow-up.
 */
namespace BC::Sirah {

//! Parsed 14-byte status response from a Sirah binary-protocol device.
struct Status {
    quint8 err;
    quint8 cStatus;
    quint8 m1Status;
    qint32 m1Pos;
    quint8 m2Status;
    qint32 m2Pos;
    int lastMoveDir{0};
};

/*!
 * \brief Build a 13-byte binary command frame.
 *
 * Byte 0 is the start marker (0x3c), byte 1 is \a cmd, bytes 2..10 hold up
 * to 9 bytes of \a args, byte 11 is a running checksum accumulated over the
 * start marker, command, and args bytes, and byte 12 is the end marker
 * (0x3e).
 */
QByteArray buildCommand(char cmd, const QByteArray &args = {});

/*!
 * \brief Parse a 14-byte status response into \a out.
 *
 * \return false if \a resp is not a well-formed 14-byte frame bracketed by
 * the expected start (0x5b) / end (0x5d) markers; \a out is left
 * unmodified in that case.
 */
bool parseStatus(const QByteArray &resp, Status &out);

}

#endif // SIRAHPROTOCOL_H

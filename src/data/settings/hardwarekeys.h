#ifndef HARDWAREKEYS_H
#define HARDWAREKEYS_H

#include <QString>

//! \file hardwarekeys.h
//! \brief Hardware settings keys that are used across multiple modules
//!
//! This header contains only the hardware keys that are:
//! 1. Defined in base hardware classes (not implementation-specific)
//! 2. Used by UI, experiment logic, or other non-hardware modules
//!
//! Implementation-specific keys should remain in their respective hardware headers.

namespace BC::Key {

// Hardware base class keys (from HardwareObject)
namespace HW {
    static const QString key{"key"}; /*!< Hardware type key */
    static const QString subKey{"subKey"}; /*!< Hardware implementation key */
    static const QString name{"prettyName"}; /*!< Display name */
    static const QString connected{"connected"}; /*!< Whether last communication was successful */
    static const QString critical{"critical"}; /*!< Whether communication failure should abort an experiment */
    static const QString threaded{"threaded"}; /*!< Whether object is in its own thread */
    static const QString commType{"commType"}; /*!< CommunicationProtocol type */
    static const QString supportedProtocols{"supportedProtocols"}; /*!< List of supported communication protocols */
    static const QString rInterval{"rollingDataIntervalSec"}; /*!< Timer interval for rolling data (seconds) */
}

} // namespace BC::Key

#endif // HARDWAREKEYS_H
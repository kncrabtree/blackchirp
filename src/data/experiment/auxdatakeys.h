#ifndef AUXDATAKEYS_H
#define AUXDATAKEYS_H

#include <QString>
#include <QLatin1StringView>
#include <QStringView>

//! \file auxdatakeys.h
//! \brief Auxiliary data keys used for experiment validation and abort conditions
//!
//! This header contains auxiliary data keys that are used to:
//! 1. Monitor hardware conditions during experiments
//! 2. Validate experimental parameters and abort if conditions are not met
//! 3. Provide real-time feedback about hardware status
//!
//! These keys are semantically different from settings keys and are used
//! by the experiment validation system.

namespace BC::Aux {

// Flow controller auxiliary data keys
namespace Flow {
    inline constexpr QLatin1StringView pressure{"Pressure"}; /*!< Pressure auxiliary data key */
    inline const QString flow{"Flow%1"}; /*!< Flow channel auxiliary data key template */
}

} // namespace BC::Aux

#endif // AUXDATAKEYS_H
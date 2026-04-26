#ifndef BC_GUIKEYS_H
#define BC_GUIKEYS_H

#include <QLatin1StringView>

// Settings keys for GUI widgets whose persistent state must be accessed
// from the data layer (e.g., to clear stale values on loadout removal).
// Keep this header free of any Qt widget or GUI includes.

namespace BC::Key::FtmwConfigWidget {
inline constexpr QLatin1StringView key{"FtmwConfigWidget"};
}

#endif // BC_GUIKEYS_H

#ifndef CRASHHANDLER_P_H
#define CRASHHANDLER_P_H

// Private interface shared between crashhandler.cpp (cross-platform glue)
// and the platform-specific implementations (crashhandler_unix.cpp,
// crashhandler_win.cpp).
//
// None of these symbols are part of the public CrashHandler API and
// they are not exported from the data library.

#include <QByteArray>

namespace BC::CrashInternal {

// Implemented in the per-platform .cpp. install() calls this once after
// formatting the build-identity header.
void installPlatformHandlers();

// Implemented in the per-platform .cpp. Called whenever the user's
// savePath changes. The crashes-directory path is already stored in
// crashesDirUtf8() at this point.
void reopenPlatform();

// Implemented in the per-platform .cpp. Called from CrashHandler::shutdown()
// before normal program exit.
void shutdownPlatform();

// Implemented in crashhandler.cpp; consumed by the per-platform handler.
const char *buildHeaderUtf8();      // null-terminated; static-lifetime
const char *crashesDirUtf8();       // null-terminated; updated by reopen()
const char *currentBasenameUtf8();  // null-terminated; per-run filename stem
int activeExperimentNumber();       // atomic load
const char *buildVersionUtf8();     // BC_BUILD_VERSION (git SHA), null-terminated

}

#endif

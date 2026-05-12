#ifndef CRASHHANDLER_H
#define CRASHHANDLER_H

#include <QString>
#include <QStringList>

/*!
 * \brief In-process diagnostic crash handler.
 *
 * Installs platform-native handlers (POSIX signal handlers on Unix,
 * SetUnhandledExceptionFilter on Windows) and writes a crash artifact
 * under \c \<savePath\>/log/crashes/ when the process faults.
 *
 * The artifact is a text log on POSIX (build identity, signal info, and
 * a std::stacktrace resolved to module+offset pairs) and a minidump
 * plus a small text sidecar on Windows.
 *
 * The handler is the documented exception to the bcLog/bcDebug logging
 * convention: signal-handler context cannot allocate, lock, or touch
 * the LogHandler. The implementation writes via raw file descriptors
 * opened from non-handler context.
 *
 * Lifecycle:
 *   - install() once at startup, immediately after QApplication
 *     construction. Sets up signal/exception handlers.
 *   - reopen(savePath) once savePath is known and again whenever the
 *     user changes savePath. Builds and opens the per-run crash log
 *     file inside \c \<savePath\>/log/crashes/.
 *   - setActiveExperiment(num) from the acquisition thread when an
 *     experiment starts (and 0 when it ends). Atomic; safe to call
 *     from any thread.
 *   - shutdown() before normal program exit. Closes the open log fd
 *     and unlinks the file if no crash was written to it (so a clean
 *     exit leaves no stray empty crash log).
 */
namespace CrashHandler {

void install();

void reopen(const QString &savePath);

void setActiveExperiment(int num);

void shutdown();

/*!
 * \brief Return paths to crash artifacts left over from prior process
 *        runs that have not yet been acknowledged by the user.
 *
 * Enumerates the crash directory configured by the most recent
 * reopen() call and returns every artifact (.log on POSIX, .dmp/.log
 * sidecars on Windows) that does not belong to the currently-running
 * process. Empty zero-byte artifacts left behind by an externally-
 * killed prior run (SIGKILL, power loss) are unlinked as a side
 * effect rather than being reported.
 *
 * Safe to call after reopen(); returns an empty list if no savePath
 * has been configured.
 */
QStringList collectPriorArtifacts();

/*!
 * \brief Return the directory under which crash artifacts are
 *        written, or an empty string if no savePath is configured.
 */
QString crashesDirectory();

/*!
 * \brief Extract the UTC timestamp portion (\c yyyyMMdd-HHmmss) from a
 *        crash artifact path's filename.
 *
 * Returns an empty string if the path's basename does not match the
 * expected \c crash-yyyyMMdd-HHmmss-\<sha\>.{log,dmp} pattern. The
 * returned string sorts lexicographically.
 */
QString artifactTimestamp(const QString &path);

}

#endif

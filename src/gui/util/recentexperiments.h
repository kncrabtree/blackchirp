#ifndef RECENTEXPERIMENTS_H
#define RECENTEXPERIMENTS_H

#include <vector>

#include <QString>

#include <data/storage/settingsstorage.h>

/*!
 * \brief Storage-free logic shared by the two recent-experiment histories.
 *
 * The acquisition app's View Experiment dialog and the standalone
 * viewer's Open Experiment dialog each keep a recent-experiment list.
 * Both lists live in the same Blackchirp settings file but under
 * different SettingsStorage scopes and array keys (the acquisition app
 * uses the root \c [Blackchirp] group with \c viewExperimentRecent; the
 * viewer uses \c [BlackchirpViewer] with \c recentExperiments). Because
 * SettingsStorage::set and setArray are protected, the actual array
 * read/write stays with each host, which has write access to its own
 * scope. This module owns only the pieces that do not touch storage:
 * the per-entry encoding, the human-readable label, and the
 * dedupe/prepend/trim list update.
 */
namespace BC::RecentExperiments {

//! SettingsMap key for the experiment number within one stored entry.
inline constexpr QLatin1StringView entryNum{"num"};
//! SettingsMap key for the experiment path within one stored entry.
inline constexpr QLatin1StringView entryPath{"path"};

//! Default cap on retained history length.
inline constexpr int maxEntries = 10;

/*!
 * \brief Label shown in menus and combo boxes.
 *
 * "Experiment N" for an experiment opened by number, or the absolute
 * directory for one opened by custom path.
 */
QString displayText(int num, const QString &path);

//! Decode one stored entry. Missing fields yield num == 0 / empty path.
void decode(const SettingsStorage::SettingsMap &entry, int &num, QString &path);

//! Encode one entry into the stored SettingsMap shape.
SettingsStorage::SettingsMap encode(int num, const QString &path);

/*!
 * \brief Apply a new entry to a recent list.
 *
 * Returns \a entries with any entry sharing (\a num, \a path)'s display
 * text removed, the new entry prepended, and the result trimmed to
 * \a max. The caller writes the result back through its own scoped
 * SettingsStorage::setArray.
 */
std::vector<SettingsStorage::SettingsMap>
prepend(std::vector<SettingsStorage::SettingsMap> entries,
        int num, const QString &path, int max = maxEntries);

}

#endif // RECENTEXPERIMENTS_H

#include "recentexperiments.h"

#include <algorithm>

#include <QDir>

namespace BC::RecentExperiments {

QString displayText(int num, const QString &path)
{
    if(path.isEmpty())
        return QString("Experiment %1").arg(num);
    return QDir(path).absolutePath();
}

void decode(const SettingsStorage::SettingsMap &entry, int &num, QString &path)
{
    num = 0;
    path.clear();
    auto nit = entry.find(entryNum);
    if(nit != entry.end())
        num = nit->second.toInt();
    auto pit = entry.find(entryPath);
    if(pit != entry.end())
        path = pit->second.toString();
}

SettingsStorage::SettingsMap encode(int num, const QString &path)
{
    SettingsStorage::SettingsMap entry;
    entry[entryNum] = num;
    entry[entryPath] = path;
    return entry;
}

std::vector<SettingsStorage::SettingsMap>
prepend(std::vector<SettingsStorage::SettingsMap> entries,
        int num, const QString &path, int max)
{
    const QString display = displayText(num, path);
    entries.erase(std::remove_if(entries.begin(), entries.end(),
        [&](const SettingsStorage::SettingsMap &m) {
            int n;
            QString p;
            decode(m, n, p);
            return displayText(n, p) == display;
        }), entries.end());

    entries.insert(entries.begin(), encode(num, path));

    if(static_cast<int>(entries.size()) > max)
        entries.resize(max);

    return entries;
}

}

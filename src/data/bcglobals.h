#ifndef BCGLOBALS_H
#define BCGLOBALS_H

#include <QString>
#include <QLatin1StringView>
#include <QStringView>

namespace BC::Key {

inline constexpr QLatin1StringView hwIndexSep{"."};

// New label-based API
QString hwKey(const QString& type, const QString& label);
QString widgetKey(const QString& widgetKey, const QString& hwKey);

// Parse functions (supports both label and index formats)
std::pair<QString, QString> parseKey(const QString& key);  // Returns {type, label}
std::pair<QString, int> parseIndexKey(const QString& key); // Legacy: returns {type, index}

// Migration utilities
QString migrateIndexKey(const QString& oldKey, const QString& type, int index);
bool isIndexKey(const QString& key);
QString generateDefaultLabel(const QString& type, const QStringList& existingLabels);

// Legacy index-based API (deprecated)
QString hwKey(const QString& k, const int index);

}

namespace BC::Unit{
inline constexpr QStringView us{u"μs"};
inline constexpr QLatin1StringView MHz{"MHz"};
inline constexpr QLatin1StringView V{"V"};
inline constexpr QLatin1StringView s{"s"};
inline constexpr QLatin1StringView Hz{"Hz"};
}

#endif // BCGLOBALS_H

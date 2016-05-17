#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <QtGlobal>
#include <QList>
#include <QString>
#include <QStringList>

namespace Analysis {

quint32 nextPowerOf2(quint32 n);
qint64 intRoundClosest(const qint64 n, const qint64 d);
QList<int> parseIntRanges(const QString str, int max);

}

#endif // ANALYSIS_H


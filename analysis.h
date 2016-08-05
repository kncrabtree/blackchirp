#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <QtGlobal>
#include <QList>
#include <QString>
#include <QStringList>
#include <QPair>

namespace Analysis {

quint32 nextPowerOf2(quint32 n);
qint64 intRoundClosest(const qint64 n, const qint64 d);
QList<int> parseIntRanges(const QString str, int max);
void split(double dat[], int n, double val, int &i, int &j);
QPair<double, double> medianFilterMeanStDev(double dat[], int n);

}

#endif // ANALYSIS_H


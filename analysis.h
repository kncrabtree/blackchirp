#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <QtGlobal>
#include <QList>
#include <QString>
#include <QStringList>
#include <QPair>
#include <QVector>

#include <eigen3/Eigen/SVD>

namespace Analysis {

quint32 nextPowerOf2(quint32 n);
qint64 intRoundClosest(const qint64 n, const qint64 d);
QList<int> parseIntRanges(const QString str, int max);
void split(double dat[], int n, double val, int &i, int &j);
QPair<double, double> medianFilterMeanStDev(double dat[], int n);
int factorial(int x);
Eigen::MatrixXd calcSavGolCoefs(int winSize, int polyOrder);
QVector<double> savGolSmooth(const Eigen::MatrixXd coefs, int derivativeOrder, QVector<double> d, double dx = 1.0);
double savGolSmoothPoint(int i, const Eigen::MatrixXd coefs, int derivativeOrder, QVector<double> d, double dx = 1.0);

}

#endif // ANALYSIS_H


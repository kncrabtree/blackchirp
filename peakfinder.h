#ifndef PEAKFINDER_H
#define PEAKFINDER_H

#include <QObject>

#include "ft.h"

#include <eigen3/Eigen/Core>

class PeakFinder : public QObject
{
    Q_OBJECT
public:
    explicit PeakFinder(QObject *parent = 0);

signals:
    void peakList(const QList<QPointF>);

public slots:
    QList<QPointF> findPeaks(const Ft ft, double minF, double maxF, double minSNR);
    void calcCoefs(int winSize, int polyOrder);

private:
    int d_window;
    int d_polyOrder;

    Eigen::MatrixXd d_coefs;

};

#endif // PEAKFINDER_H

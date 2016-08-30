#ifndef MOTORSCAN_H
#define MOTORSCAN_H

#include <QSharedDataPointer>
#include <QVector>
#include <QList>
#include <QPointF>

#include <qwt6/qwt_matrix_raster_data.h>

class MotorScanData;
class QwtMatrixRasterData;

class MotorScan
{
public:

    enum MotorDataAxis {
        MotorX,
        MotorY,
        MotorZ,
        MotorT
    };

    MotorScan();
    MotorScan(const MotorScan &rhs);
    MotorScan &operator=(const MotorScan &rhs);

    static MotorScan fromSettings();
    void saveToSettings() const;

    int xPoints() const;
    int yPoints() const;
    int zPoints() const;
    int tPoints() const;
    int numPoints(MotorDataAxis axis) const;

    double xVal(int i) const;
    double yVal(int i) const;
    double zVal(int i) const;
    double tVal(int i) const;

    int shotsPerPoint() const;
    bool isComplete() const;

    QVector<QPointF> tSlice(int x, int y, int z) const;
    QVector<double> xySlice(int z, int t) const;
    QVector<double> yzSlice(int x, int t) const;

    void setXPoints(int x);
    void setYPoints(int y);
    void setZPoints(int z);
    void setTPoints(int t);
    void setIntervals(double x0, double y0, double z0, double dx, double dy, double dz);

    void setShotsPerPoint(const int s);
    void initialize();





private:
    QSharedDataPointer<MotorScanData> data;
};

class MotorScanData : public QSharedData
{
public:
    MotorScanData() : xPoints(0), yPoints(0), zPoints(0), tPoints(0),
    x0(0.0), y0(0.0), z0(0.0), t0(0.0), dx(1.0), dy(1.0), dz(1.0), dt(1.0) {}

    int xPoints;
    int yPoints;
    int zPoints;
    int tPoints;


    double x0;
    double y0;
    double z0;
    double t0;

    double dx;
    double dy;
    double dz;
    double dt;

    int shotsPerPoint;
    int currentPointShots;

    int totalPoints;
    int currentPoint;

    QList<QList<QList<QVector<double>>>> zyxtData;
};

#endif // MOTORSCAN_H

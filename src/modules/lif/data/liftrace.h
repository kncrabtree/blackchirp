#ifndef LIFTRACE_H
#define LIFTRACE_H

#include <QSharedDataPointer>

#include <modules/lif/hardware/lifdigitizer/lifdigitizerconfig.h>

#include <QVector>
#include <QPointF>

class LifTraceData;

class LifTrace
{
public:
    LifTrace();
    explicit LifTrace(const LifDigitizerConfig &c, const QByteArray b, int dIndex, int lIndex);
    LifTrace(const LifTrace &other);
    LifTrace &operator=(const LifTrace &other);
    ~LifTrace() = default;

    double integrate(int gl1, int gl2, int gr1 = -1, int gr2 = -1) const;
    int delayIndex() const;
    int laserIndex() const;
    QVector<QPointF> lifToXY() const;
    QVector<QPointF> refToXY() const;
    double maxTime() const;
    QVector<qint64> lifRaw() const;
    QVector<qint64> refRaw() const;
    int shots() const;
    int size() const;
    bool hasRefData() const;

    void add(const LifTrace &other);
    void rollAvg(const LifTrace &other, int numShots);

private:
    QSharedDataPointer<LifTraceData> p_data;


};

class LifTraceData : public QSharedData {
public:
    double lifYMult{1.0}, refYMult{1.0};
    double xSpacing{1.0};
    int delayIndex{-1}, laserIndex{-1};
    QVector<qint64> lifData, refData;
    int shots{0};
};

#endif // LIFTRACE_H

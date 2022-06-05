#ifndef LIFTRACE_H
#define LIFTRACE_H

#include <memory>
#include <modules/lif/hardware/lifdigitizer/lifdigitizerconfig.h>

#include <QVector>
#include <QPointF>
#include <QReadWriteLock>

class LifTrace
{
public:
    LifTrace();
    explicit LifTrace(const LifDigitizerConfig &c, const QByteArray b);
    explicit LifTrace(double lm, double rm, double sp, int count, const QVector<qint64> l, const QVector<qint64> r);
    ~LifTrace() = default;

    double d_lifYMult{1.0}, d_refYMult{1.0};
    double d_xSpacing{1.0};

    double integrate(int gl1, int gl2, int gr1 = -1, int gr2 = -1) const;
    QVector<QPointF> lifToXY() const;
    QVector<QPointF> refToXY() const;
    double maxTime() const;
    QVector<qint64> lifRaw() const;
    QVector<qint64> refRaw() const;
    int count() const;
    int size() const;
    bool hasRefData() const;

    void add(const LifTrace &other);
    void rollAvg(const LifTrace &other, int numShots);

private:
    std::unique_ptr<QReadWriteLock> pu_lock{std::make_unique<QReadWriteLock>()};
    QVector<qint64> d_lifData, d_refData;
    int d_count{1};

};

#endif // LIFTRACE_H

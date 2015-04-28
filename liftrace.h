#ifndef LIFTRACE_H
#define LIFTRACE_H

#include <QSharedDataPointer>

#include <QVector>
#include <QPointF>
#include <QPair>
#include <QByteArray>
#include <QDataStream>

class LifTraceData;

class LifTrace
{
public:
    explicit LifTrace(const QByteArray b, QDataStream::ByteOrder bo, int bytesPerPoint, int recLen, double xIncr, double lym, bool ref = false, double rym = -1.0);
    LifTrace(const LifTrace &);
    LifTrace &operator=(const LifTrace &);
    ~LifTrace();

    double integrate(int gl1, int gl2, int gr1 = -1, int gr2 = -1) const;
    QVector<QPointF> lifToXY() const;
    QVector<QPointF> refToXY() const;
    double maxTime() const;
    qint64 lifAtRaw(int i) const;
    qint64 refAtRaw(int i) const;
    int count() const;

    void add(const LifTrace &other);
    void rollAvg(const LifTrace &other, int numShots);

private:
    QSharedDataPointer<LifTraceData> data;
};


class LifTraceData : public QSharedData
{
public:
    LifTraceData() : lifYMult(-1.0), refYMult(-1.0), xSpacing(-1.0), count(0) {}

    QVector<qint64> lifData, refData;
    double lifYMult, refYMult;
    double xSpacing;
    int count;
};

#endif // LIFTRACE_H

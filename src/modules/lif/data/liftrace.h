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
    explicit LifTrace(int di, int li, QVector<qint64> ld, QVector<qint64> rd, int shots, double xsp, double lym, double rym);
    LifTrace(const LifTrace &other);
    LifTrace &operator=(const LifTrace &other);
    ~LifTrace() = default;

    struct LifProcSettings {
        int lifGateStart{-1};
        int lifGateEnd{-1};
        int refGateStart{-1};
        int refGateEnd{-1};
        double lowPassAlpha{};
        bool savGolEnabled{false};
        int savGolWin{11};
        int savGolPoly{3};
    };

    double integrate(const LifProcSettings &s) const;
    int delayIndex() const;
    int laserIndex() const;
    QVector<QPointF> lifToXY(const LifProcSettings &s) const;
    QVector<QPointF> refToXY(const LifProcSettings &s) const;
    double maxTime() const;
    QVector<qint64> lifRaw() const;
    QVector<qint64> refRaw() const;
    int shots() const;
    int size() const;
    bool hasRefData() const;
    double xSpacing() const;
    double xSpacingns() const;
    double lifYMult() const;
    double refYMult() const;


    void add(const LifTrace &other);
    void rollAvg(const LifTrace &other, int numShots);

private:
    QSharedDataPointer<LifTraceData> p_data;
    QVector<QPointF> processXY(const QVector<double> d, const LifProcSettings &s) const;


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

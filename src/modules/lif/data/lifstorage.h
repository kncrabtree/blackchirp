#ifndef LIFSTORAGE_H
#define LIFSTORAGE_H

#include <memory>
#include <QDateTime>
#include <QMutex>

#include <data/storage/datastoragebase.h>
#include <modules/lif/data/liftrace.h>

class BlackchirpCSV;

namespace BC::Key::LifStorage {
static const QString lifGateStart("LifGateStartPoint");
static const QString lifGateEnd("LifGateEndPoint");
static const QString refGateStart("RefGateStartPoint");
static const QString refGateEnd("RefGateEndPoint");
static const QString lowPassAlpha("LowPassAlpha");
static const QString savGol{"SavGolEnabled"};
static const QString sgWin{"SavGolWindow"};
static const QString sgPoly{"SavGolPoly"};
}

class LifStorage : public DataStorageBase
{
public:
    LifStorage(int dp, int lp, int num, QString path="");
    ~LifStorage();

    const int d_delayPoints, d_laserPoints;

    void advance() override;
    void save() override;
    void start() override;
    void finish() override;

    int currentTraceShots() const;
    int completedShots() const;

    LifTrace getLifTrace(int di, int li);
    LifTrace currentLifTrace() const { return d_currentTrace; };
    LifTrace loadLifTrace(int di, int li);
    void writeLifTrace(const LifTrace t);

    void addTrace(const LifTrace t);

    void writeProcessingSettings(const LifTrace::LifProcSettings &c);
    bool readProcessingSettings(LifTrace::LifProcSettings &out);


private:
    bool d_acquiring{false}, d_nextNew{true};
    std::map<int,LifTrace> d_data;
    LifTrace d_currentTrace;


    int index(int dp, int lp) const;

};

#endif // LIFSTORAGE_H

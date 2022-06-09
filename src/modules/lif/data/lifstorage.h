#ifndef LIFSTORAGE_H
#define LIFSTORAGE_H

#include <memory>
#include <QDateTime>
#include <QMutex>

#include <modules/lif/data/liftrace.h>

class BlackchirpCSV;

class LifStorage
{
public:
    LifStorage(int dp, int lp, int num, QString path="");

    const int d_delayPoints, d_laserPoints, d_number;
    const QString d_path;

    void advance();
    void save();
    void start();
    void finish();

    int currentTraceShots() const;
    int completedShots() const;

    LifTrace getLifTrace(int di, int li);
    LifTrace currentLifTrace() const { return d_currentTrace; };
    LifTrace loadLifTrace(int di, int li) const;
    void writeLifTrace(const LifTrace t);

    void addTrace(const LifTrace t);


private:
    std::unique_ptr<QMutex> pu_mutex{std::make_unique<QMutex>()};
    bool d_acquiring{false};
    int d_lastIndex{-1};
    std::map<int,LifTrace> d_data;
    LifTrace d_currentTrace;


    int index(int dp, int lp) const;

};

#endif // LIFSTORAGE_H

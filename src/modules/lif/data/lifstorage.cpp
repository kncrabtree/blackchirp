#include "lifstorage.h"

#include <QMutexLocker>

LifStorage::LifStorage(int dp, int lp, int num, QString path)
    : d_delayPoints{dp}, d_laserPoints{lp}, d_number{num}, d_path{path}
{

}

void LifStorage::advance()
{
    save();

    QMutexLocker l(pu_mutex.get());
    d_lastIndex = index(d_currentTrace.delayIndex(),d_currentTrace.laserIndex());
    d_data.emplace(d_lastIndex,d_currentTrace);
    d_currentTrace = LifTrace();

}

void LifStorage::save()
{

}

void LifStorage::start()
{
    QMutexLocker l(pu_mutex.get());
    d_acquiring = true;
}

void LifStorage::finish()
{
    QMutexLocker l(pu_mutex.get());
    d_acquiring = false;
}

int LifStorage::currentTraceShots() const
{
    QMutexLocker l(pu_mutex.get());
    return d_currentTrace.shots();
}

int LifStorage::completedShots() const
{
    QMutexLocker l(pu_mutex.get());
    int out = 0;
    for(auto it = d_data.cbegin(); it != d_data.cend(); ++it)
        out += it->second.shots();

    if(!d_acquiring)
        return out;

    return out + d_currentTrace.shots();
}

LifTrace LifStorage::getLifTrace(int di, int li)
{
    auto i = index(di,li);

    QMutexLocker l(pu_mutex.get());
    auto it = d_data.find(i);
    if(it != d_data.end())
        return it->second;

    if(i == index(d_currentTrace.delayIndex(),d_currentTrace.laserIndex()))
        return d_currentTrace;

    ///TODO: if not acquiring, try to load from disk here

    return LifTrace();
}

void LifStorage::addTrace(const LifTrace t)
{
    QMutexLocker l(pu_mutex.get());
    if(d_currentTrace.shots() == 0)
        d_currentTrace = t;
    else
        d_currentTrace.add(t);
}

int LifStorage::index(int dp, int lp) const
{
    return dp*d_delayPoints + lp;
}

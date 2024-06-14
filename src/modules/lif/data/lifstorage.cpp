#include "lifstorage.h"

#include <data/storage/blackchirpcsv.h>

#include <QMutexLocker>
#include <QSaveFile>

LifStorage::LifStorage(int dp, int lp, int num, QString path)
    : DataStorageBase(num,path), d_delayPoints{dp}, d_laserPoints{lp}
{
}

LifStorage::~LifStorage()
{
}

void LifStorage::advance()
{

    QMutexLocker l(pu_mutex.get());
    d_nextNew = true;
    l.unlock();

    save();
}

void LifStorage::save()
{
    //first, write current trace file
    QMutexLocker l(pu_mutex.get());
    auto i = index(d_currentTrace.delayIndex(),d_currentTrace.laserIndex());
    auto it = d_data.emplace(i,d_currentTrace);
    if(!it.second) //insertion not successful if key already exists
        d_data[i] = d_currentTrace;

    QDir d{BlackchirpCSV::exptDir(d_number,d_path)};
    if(!d.cd(BC::CSV::lifDir))
    {
        if(!d.mkdir(BC::CSV::lifDir))
            return;
        if(!d.cd(BC::CSV::lifDir))
            return;
    }

    QSaveFile dat(d.absoluteFilePath("%1.csv").arg(i));
    if(!dat.open(QIODevice::WriteOnly|QIODevice::Text))
        return;

    QTextStream t(&dat);

    if(d_currentTrace.hasRefData())
    {
        t << "lif" << BC::CSV::del << "ref" << BC::CSV::nl;

        auto lr = d_currentTrace.lifRaw();
        auto rr = d_currentTrace.refRaw();

        for(int i=0; i<d_currentTrace.size(); i++)
        {
            t << BlackchirpCSV::formatInt64(lr.at(i))
              << BC::CSV::del
              << BlackchirpCSV::formatInt64(rr.at(i))
              << BC::CSV::nl;
        }
    }
    else
    {
        t << "lif" << BC::CSV::nl;

        auto lr = d_currentTrace.lifRaw();
        for(int i=0; i<d_currentTrace.size(); i++)
        {
            t << BlackchirpCSV::formatInt64(lr.at(i))
              << BC::CSV::nl;
        }
    }

    l.unlock();

    if(!dat.commit())
        return;

    QSaveFile hdr(d.absoluteFilePath(BC::CSV::lifparams));
    if(!hdr.open(QIODevice::WriteOnly|QIODevice::Text))
        return;

    QTextStream txt(&hdr);

    BlackchirpCSV::writeLine(txt,{"lIndex","dIndex","shots","lifsize","refsize","spacing","lifymult","refymult"});

    l.relock();
    for(auto it = d_data.cbegin(); it != d_data.cend(); ++it)
    {
        auto &t = it->second;
        if(t.hasRefData())
            BlackchirpCSV::writeLine(txt,{t.laserIndex(),t.delayIndex(),t.shots(),t.size(),t.size(),t.xSpacing(),t.lifYMult(),t.refYMult()});
        else
            BlackchirpCSV::writeLine(txt,{t.laserIndex(),t.delayIndex(),t.shots(),t.size(),0,t.xSpacing(),t.lifYMult(),0});
    }
    l.unlock();

    hdr.commit();

}

void LifStorage::start()
{
    QMutexLocker l(pu_mutex.get());
    d_acquiring = true;
    d_nextNew = true;
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

    if(d_nextNew)
        return out;

    return out + d_currentTrace.shots();
}

LifTrace LifStorage::getLifTrace(int di, int li)
{
    auto i = index(di,li);

    QMutexLocker l(pu_mutex.get());

    if(i == index(d_currentTrace.delayIndex(),d_currentTrace.laserIndex()))
        return d_currentTrace;

    auto it = d_data.find(i);
    if(it != d_data.end())
        return it->second;

    if(!d_acquiring)
    {
        l.unlock();
        return loadLifTrace(di,li);
    }

    return LifTrace();
}

LifTrace LifStorage::loadLifTrace(int di, int li)
{
    LifTrace out;
    QDir d{BlackchirpCSV::exptDir(d_number,d_path)};
    d.cd(BC::CSV::lifDir);

    QFile hdr(d.absoluteFilePath(BC::CSV::lifparams));
    if(!hdr.open(QIODevice::ReadOnly|QIODevice::Text))
        return out;

    bool found = false;
    int lsize{0}, rsize{0}, shots{0};
    double xsp{1.0}, lym{0.0}, rym{0.0};

    while(!hdr.atEnd())
    {
        auto l = pu_csv->readLine(hdr);
        if(l.size() < 8)
            continue;

        bool ok = false;
        int lli = l.constFirst().toInt(&ok);
        if(ok)
        {
            int ddi = l.at(1).toInt(&ok);
            if(ok)
            {
                if(lli == li && ddi == di)
                {
                    found = true;
                    shots = l.at(2).toInt();
                    lsize = l.at(3).toInt();
                    rsize = l.at(4).toInt();
                    xsp = l.at(5).toDouble();
                    lym = l.at(6).toDouble();
                    rym = l.at(7).toDouble();
                    break;
                }
            }
        }
    }

    if(!found)
        return out;

    auto idx = index(di,li);
    QFile dat(d.absoluteFilePath("%1.csv").arg(idx));
    if(!dat.open(QIODevice::ReadOnly|QIODevice::Text))
        return out;

    QVector<qint64> lifData(lsize), refData(rsize);
    auto l = pu_csv->readLine(dat); //read first line which contains titles
    for(int i=0; i<lsize; i++)
    {
        l = pu_csv->readLine(dat);
        lifData[i] = l.constFirst().toString().toLongLong(nullptr,36);
        if(i<rsize && l.size() == 2)
            refData[i] = l.at(1).toString().toLongLong(nullptr,36);
    }

    out = LifTrace(di,li,lifData,refData,shots,xsp,lym,rym);
    QMutexLocker lock(pu_mutex.get());
    d_data.emplace(idx,out);
    return out;
}

void LifStorage::addTrace(const LifTrace t)
{
    QMutexLocker l(pu_mutex.get());
    if(d_nextNew)
    {
        auto idx = index(t.delayIndex(),t.laserIndex());
        auto it = d_data.find(idx);
        if(it != d_data.end())
        {
            d_currentTrace = it->second;
            d_currentTrace.add(t);
        }
        else
            d_currentTrace = t;

        d_nextNew = false;
    }
    else
        d_currentTrace.add(t);
}

void LifStorage::writeProcessingSettings(const LifTrace::LifProcSettings &c)
{
    using namespace BC::Key::LifStorage;
    std::map<QString,QVariant> m;
    m.emplace(lifGateStart,c.lifGateStart);
    m.emplace(lifGateEnd,c.lifGateEnd);
    m.emplace(refGateStart,c.refGateStart);
    m.emplace(refGateEnd,c.refGateEnd);
    m.emplace(lowPassAlpha,c.lowPassAlpha);
    m.emplace(savGol,c.savGolEnabled);
    m.emplace(sgWin,c.savGolWin);
    m.emplace(sgPoly,c.savGolPoly);

    writeMetadata(m,BC::CSV::lifDir);
}

bool LifStorage::readProcessingSettings(LifTrace::LifProcSettings &out)
{
    using namespace BC::Key::LifStorage;
    std::map<QString,QVariant> m;
    readMetadata(m,BC::CSV::lifDir);

    if(m.empty())
        return false;

    auto it = m.find(lifGateStart);
    if(it != m.end())
        out.lifGateStart = it->second.toInt();
    it = m.find(lifGateEnd);
    if(it != m.end())
        out.lifGateEnd = it->second.toInt();
    it = m.find(refGateStart);
    if(it != m.end())
        out.refGateStart = it->second.toInt();
    it = m.find(refGateEnd);
    if(it != m.end())
        out.refGateEnd = it->second.toInt();
    it = m.find(lowPassAlpha);
    if(it != m.end())
        out.lowPassAlpha = it->second.toDouble();
    it = m.find(savGol);
    if(it != m.end())
        out.savGolEnabled = it->second.toBool();
    it = m.find(sgWin);
    if(it != m.end())
        out.savGolWin = it->second.toInt();
    it = m.find(sgPoly);
    if(it != m.end())
        out.savGolPoly = it->second.toInt();

    return true;
}

int LifStorage::index(int dp, int lp) const
{
    return dp*d_laserPoints + lp;
}

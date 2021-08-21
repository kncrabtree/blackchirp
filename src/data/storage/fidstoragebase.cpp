#include "fidstoragebase.h"

#include <QSaveFile>
#include <QDir>
#include <data/storage/blackchirpcsv.h>

FidStorageBase::FidStorageBase(int numRecords, int number, QString path) :
    d_number(number), d_numRecords(numRecords), d_path(path)
{
    pu_csv = std::make_unique<BlackchirpCSV>(number,path);
    pu_mutex = std::make_unique<QMutex>();
}

FidStorageBase::~FidStorageBase()
{
}

void FidStorageBase::advance()
{
    save();
    _advance();
}

void FidStorageBase::save()
{
    //if path isn't set, then data can't be saved
    //Don't throw an error; this is probably intentional (peak up mode)
    if(d_number < 1)
        return;

    auto l = getCurrentFidList();
    auto i = getCurrentIndex();

    saveFidList(l,i);
}

void FidStorageBase::start()
{
    d_acquiring = true;
}

void FidStorageBase::finish()
{
    d_acquiring = false;
}

void FidStorageBase::saveFidList(const FidList l, int i)
{
    if(d_number < 1)
        return;

    auto f = l.constFirst();
    f.setData({});

    while(i > d_templateList.size())
        d_templateList.push_back(Fid());
    if(i == d_templateList.size())
        d_templateList.push_back(f);
    else
        d_templateList[i] = f;

    //Note: this copy is made so that it can be captuted in the lambda function below for saving.
    auto tl = d_templateList;

    QDir d{BlackchirpCSV::exptDir(d_number,d_path)};
    if(!d.cd(BC::CSV::fidDir))
    {
        if(!d.mkdir(BC::CSV::fidDir))
            return;
        if(!d.cd(BC::CSV::fidDir))
            return;
    }


    QSaveFile hdr(d.absoluteFilePath(BC::CSV::fidparams));
    if(!hdr.open(QIODevice::WriteOnly|QIODevice::Text))
        return;

    QTextStream txt(&hdr);
    BlackchirpCSV::writeLine(txt,{"index","spacing","probefreq","vmult","shots","sideband","size"});
    for(int idx=0; idx<tl.size(); ++idx)
    {
        auto &f = tl.at(idx);
        BlackchirpCSV::writeLine(txt,{idx,f.spacing(),
                                      f.probeFreq(),f.vMult(),f.shots(),f.sideband(),l.constFirst().size()});
    }
    pu_mutex->lock();
    bool success = hdr.commit();
    pu_mutex->unlock();
    if(!success)
        return;


    QSaveFile dat(d.absoluteFilePath("%1.csv").arg(i));
    if(!dat.open(QIODevice::WriteOnly|QIODevice::Text))
        return;

    BlackchirpCSV::writeFidList(dat,l);

    if(!dat.commit())
        return;

}

FidList FidStorageBase::loadFidList(int i)
{
    QMutexLocker lock(pu_mutex.get());
    if(i == d_currentSegment)
    {
        auto fl = getCurrentFidList();
        if(!fl.isEmpty())
            return fl;
    }

    FidList out;
    Fid fidTemplate;

    auto it = d_cache.find(i);
    if(it != d_cache.end())
    {
        out = it->second;
        //if we are no longer acquiring, we don't need to check
        //to make sure the cache is updated
        if(!d_acquiring)
            return out;
    }

    QDir d{BlackchirpCSV::exptDir(d_number,d_path)};
    d.cd(BC::CSV::fidDir);
    auto csv = pu_csv.get();

    bool found = false;
    int size = 0;
    QFile hdr(d.absoluteFilePath(BC::CSV::fidparams));
    if(!hdr.open(QIODevice::ReadOnly|QIODevice::Text))
        return out;

    while(!hdr.atEnd())
    {
        auto l = csv->readLine(hdr);
        if(l.size() != 7)
            continue;

        bool ok = false;
        int idx = l.constFirst().toInt(&ok);
        if(ok)
        {
            if(idx != i)
                continue;

            found = true;
            fidTemplate.setSpacing(l.at(1).toDouble());
            fidTemplate.setProbeFreq(l.at(2).toDouble());
            fidTemplate.setVMult(l.at(3).toDouble());
            fidTemplate.setShots(l.at(4).toULongLong());
            fidTemplate.setSideband(l.at(5).value<RfConfig::Sideband>());
            size = l.at(6).toInt();
        }
    }
    hdr.close();

    if(!found)
        return out;

    //at this point, if out is not empty, then the FidList was found in the cache
    //If the number of shots in the fidTemplate matches the number of shots in
    //the fidlist, then the cache is up to date and we can return it without
    //re-reading the fid data from disk
    if(!out.isEmpty() && out.constFirst().shots() == fidTemplate.shots())
        return out;

    QFile fid(d.absoluteFilePath(QString("%1.csv").arg(i)));
    if(!fid.open(QIODevice::ReadOnly|QIODevice::Text))
        return out;

    if(fid.atEnd())
        return out;

    //the first line contains titles, but can be parsed to figure out how many
    //FIDs are in the file
    auto l = csv->readLine(fid);
    if(l.isEmpty())
        return out;

    QVector<QVector<qint64>> data;
    for(int j=0; j<l.size(); ++j)
    {
        out << fidTemplate;
        QVector<qint64> _d;
        _d.reserve(size);
        data << _d;
    }

    while(!fid.atEnd())
    {
        auto sl = csv->readFidLine(fid);
        if(sl.size() != data.size())
            continue;

        for(int j=0; j<sl.size(); ++j)
            data[j].append(sl.at(j));
    }

    for(int j=0; j<data.size(); ++j)
        out[j].setData(data.at(j));

    //at this point, we either need to update the cache or add this item to the cache
    if(!out.isEmpty())
    {
        if(it != d_cache.end())
            it->second = out;
        else
        {
            std::size_t recSize = out.constFirst().size()*out.size();
            if(!d_cache.empty() && (recSize * (d_cache.size() + 1) > d_maxCacheSize))
            {
                //remove an item from the cache
                d_cache.erase(d_cacheKeys.front());
                d_cacheKeys.pop();
            }

            d_cacheKeys.push(i);
            d_cache.emplace(i,out);
        }
    }

    return out;

}


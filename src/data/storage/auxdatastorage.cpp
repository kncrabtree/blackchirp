#include "auxdatastorage.h"

#include <data/storage/blackchirpcsv.h>

AuxDataStorage::AuxDataStorage(int number, const QString path) : d_number(number), d_path(path)
{

}

void AuxDataStorage::registerKey(const QString objKey, const QString key)
{
    auto k = BC::Aux::keyTemplate.arg(objKey).arg(key);
    d_allowedKeys.insert(makeKey(objKey,key));
}

void AuxDataStorage::registerKey(const QString hwKey, const QString hwSubKey, const QString key)
{
    auto k = BC::Aux::hwKeyTemplate.arg(hwKey).arg(hwSubKey).arg(key);
    d_allowedKeys.insert(makeKey(hwKey,hwSubKey,key));
}

void AuxDataStorage::addDataPoints(AuxDataStorage::AuxDataMap &m)
{
    for(auto it = m.begin(); it != m.end(); ++it)
    {
        auto it2 = d_allowedKeys.find(it->first);
        if(it2 != d_allowedKeys.end())
            d_currentPoint.map.insert({it->first,it->second});
    }
}

void AuxDataStorage::startNewPoint()
{
    if(d_allowedKeys.empty())
        return;

    QDir d = BlackchirpCSV::exptDir(d_number);
    if(!d.exists())
        return;

    QFile f(d.absoluteFilePath(BC::CSV::auxFile));
    if(!f.open(QIODevice::Append|QIODevice::Text))
        return;

    QTextStream t(&f);

    if(!d_allowedKeys.empty() && d_currentPoint.map.empty())
    {
        d_startTime = QDateTime::currentDateTime();

        //write column headers
        QVariantList l {"timestamp","epochtime","elapsedsecs"};
        l.reserve(3+d_allowedKeys.size());
        for(auto it = d_allowedKeys.cbegin(); it != d_allowedKeys.cend(); ++it)
            l.append(*it);
        BlackchirpCSV::writeLine(t,l);

    }
    else
    {
        QVariantList l {d_currentPoint.dateTime.toString(),
                    d_currentPoint.dateTime.toSecsSinceEpoch(),
                    d_startTime.secsTo(d_currentPoint.dateTime)};
        l.reserve(3+d_allowedKeys.size());
        for(auto it = d_allowedKeys.cbegin(); it != d_allowedKeys.cend(); ++it)
        {
            auto it2 = d_currentPoint.map.find(*it);
            if(it2 == d_currentPoint.map.end())
                l.append("");
            else
                l.append(it2->second);
        }
        BlackchirpCSV::writeLine(t,l);
    }

    d_currentPoint.dateTime = QDateTime::currentDateTime();
    d_currentPoint.map.clear();
}

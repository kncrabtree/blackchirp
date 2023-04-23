#include "datastoragebase.h"

#include <QSaveFile>
#include <data/storage/blackchirpcsv.h>

DataStorageBase::DataStorageBase(int number, QString path) : d_number(number), d_path(path)
{
    pu_csv = std::make_unique<BlackchirpCSV>(number,path);
    pu_mutex = std::make_unique<QMutex>();
}

DataStorageBase::~DataStorageBase()
{
}

void DataStorageBase::writeMetadata(const std::map<QString, QVariant> &dat, QString dir)
{
    QDir d(BlackchirpCSV::exptDir(d_number,d_path));
    if(!dir.isEmpty())
    {
        if(!d.cd(dir))
        {
            if(!d.mkdir(dir))
                return;
            if(!d.cd(dir))
                return;
        }
    }

    QSaveFile f(d.absoluteFilePath(BC::Key::DS::proc));
    if(!f.open(QIODevice::WriteOnly|QIODevice::Text))
        return;
    QTextStream t(&f);

    BlackchirpCSV::writeLine(t,{BC::CSV::ok,BC::CSV::vv});
    for(auto it = dat.cbegin(); it != dat.cend(); ++it)
        BlackchirpCSV::writeLine(t,{it->first,it->second});

    f.commit();
}

void DataStorageBase::readMetadata(std::map<QString, QVariant> &out, QString dir)
{
    QDir d(BlackchirpCSV::exptDir(d_number,d_path));
    if(!dir.isEmpty())
    {
        if(!d.cd(dir))
            return;
    }

    QFile f(d.absoluteFilePath(BC::Key::DS::proc));
    if(!f.open(QIODevice::ReadOnly|QIODevice::Text))
        return;

    auto l = pu_csv->readLine(f);
    if(l.size() < 2)
        return;

    while(!f.atEnd())
    {
        l = pu_csv->readLine(f);
        if(l.size() < 2)
            continue;
        out.emplace(l.at(0).toString(),l.at(1));
    }
}

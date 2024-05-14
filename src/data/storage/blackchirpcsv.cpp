#include "blackchirpcsv.h"

//#include <gui/plot/blackchirpplotcurve.h>
#include <data/storage/settingsstorage.h>

BlackchirpCSV::BlackchirpCSV() : d_delimiter(BC::CSV::del)
{

}

BlackchirpCSV::BlackchirpCSV(const int num, const QString path)
{
    QDir d(BlackchirpCSV::exptDir(num,path));
    if(!d.exists())
        return;

    QFile ver(d.absoluteFilePath(BC::CSV::versionFile));
    d_delimiter = BC::CSV::del;

    if(ver.open(QIODevice::ReadOnly|QIODevice::Text))
    {
        //first line contains delimiter
        auto l = ver.readLine().trimmed();
        if(!l.isEmpty())
            d_delimiter = QString(l.at(0));

        while(!ver.atEnd())
        {
            auto line = QString(ver.readLine().trimmed());
            if(line.startsWith("key"))
                continue;

            auto list = line.split(d_delimiter);
            if(list.size() == 2)
                d_configMap.insert_or_assign(list.constFirst(),list.constLast());
        }
    }
}

bool BlackchirpCSV::writeXY(QIODevice &device, const QVector<QPointF> d, const QString prefix)
{
    using namespace BC::CSV;

    QTextStream t(&device);

    if(prefix.isEmpty())
        t << x << del << y;
    else
        t << prefix << sep << x << del << prefix << sep << y;

    for(auto it = d.constBegin(); it != d.constEnd(); it++)
        t << nl << QVariant{it->x()}.toString() << del << QVariant{it->y()}.toString();

    return true;
}

bool BlackchirpCSV::writeMultiple(QIODevice &device, const std::vector<QVector<QPointF> > &l, const std::vector<QString> &n)
{
    using namespace BC::CSV;

    QTextStream t(&device);

    //create a title for every element of l, and determine max size
    int max = 0;
    for(std::size_t i=0; i<l.size(); ++i)
    {
        max = qMax(max,l.at(i).size());
        if(i > 0)
            t << del;
        if(i < n.size())
            t << n.at(i) << sep << x << del << n.at(i) << sep << y;
        else
            t << x << i << del << y << i;
    }

    for(int i = 0; i<max; ++i)
    {
        t << nl;

        for(std::size_t j = 0; j < l.size(); ++j)
        {
            if(j > 0)
                t << del;
            if(i <l.at(j).size())
                t << QVariant{l.at(j).at(i).x()}.toString() << del << QVariant{l.at(j).at(i).y()}.toString();
            else
                t << del;
        }

    }
    return true;
}

bool BlackchirpCSV::writeHeader(QIODevice &device, const std::multimap<QString, std::tuple<QString, QString, QString, QString, QString> > header)
{
    using namespace BC::CSV;

    QTextStream t(&device);

    t << ok << del << ak << del << ai << del << vk << del << vv << del << vu;

    for(auto it = header.cbegin(); it != header.cend(); ++it)
        t << nl << it->first << del
          << std::get<0>(it->second) << del
          << std::get<1>(it->second) << del
          << std::get<2>(it->second) << del
          << std::get<3>(it->second) << del
          << std::get<4>(it->second);

    return true;
}

void BlackchirpCSV::writeLine(QTextStream &t, const QVariantList l)
{
    using namespace BC::CSV;

    if(l.isEmpty())
        return;

    t << l.constFirst().toString();
    int num = l.size();
    for(int i=1; i<num; i++)
        t << del << l.at(i).toString();
    t << nl;
}

QString BlackchirpCSV::formatInt64(qint64 n)
{
    if(n>=0)
        return QString::number(n,36);
    else
        return QString("-")+QString::number(qAbs(n),36);
}

void BlackchirpCSV::writeFidList(QIODevice &device, const FidList l)
{
    using namespace BC::CSV;

    QTextStream t(&device);

    t << "fid0";
    auto maxSize = l.constFirst().size();
    for(int i=1; i<l.size(); ++i)
    {
        t << del << "fid" << QString::number(i);
        maxSize = qMax(maxSize,l.at(i).size());
    }

    for(int i=0; i<maxSize; ++i)
    {
        t << nl << formatInt64(l.constFirst().valueRaw(i));

        for(int j=1; j<l.size(); ++j)
            t << del << formatInt64(l.at(j).valueRaw(i));
    }
}

bool BlackchirpCSV::writeVersionFile(int num)
{
    QDir d(BlackchirpCSV::exptDir(num));
    QFile ver(d.absoluteFilePath(BC::CSV::versionFile));
    if(!ver.open(QIODevice::WriteOnly|QIODevice::Text))
        return false;

    using namespace BC::CSV;

    QTextStream t(&ver);
    //the first line should contain just the delimiter
    BlackchirpCSV::writeLine(t,{"",""});
    BlackchirpCSV::writeLine(t,{"key","value"});
    BlackchirpCSV::writeLine(t,{majver,BC_MAJOR_VERSION});
    BlackchirpCSV::writeLine(t,{minver,BC_MINOR_VERSION});
    BlackchirpCSV::writeLine(t,{patchver,BC_PATCH_VERSION});
    BlackchirpCSV::writeLine(t,{relver,STRINGIFY(BC_RELEASE_VERSION)});
    BlackchirpCSV::writeLine(t,{buildver,STRINGIFY(BC_BUILD_VERSION)});

    return true;
}

QVariantList BlackchirpCSV::readLine(QIODevice &device)
{
    QVariantList out;
    auto l = QString(device.readLine()).trimmed();
    if(l.isEmpty())
        return out;
    auto list = l.split(d_delimiter);
    for(auto &str : list)
        out << str;

    return out;

}

QVector<qint64> BlackchirpCSV::readFidLine(QIODevice &device)
{
    QVector<qint64> out;
    auto l = QString(device.readLine()).trimmed();
    if(l.isEmpty())
        return out;
    auto list = l.split(d_delimiter);
    for(auto &str : list)
        out << str.toLongLong(nullptr,36);

    return out;

}

int BlackchirpCSV::majorVersion() const
{
    auto it = d_configMap.find(BC::CSV::majver);
    if(it != d_configMap.end())
        return it->second.toInt();
    return -1;
}

int BlackchirpCSV::minorVersion() const
{
    auto it = d_configMap.find(BC::CSV::minver);
    if(it != d_configMap.end())
        return it->second.toInt();
    return -1;
}

int BlackchirpCSV::patchVersion() const
{
    auto it = d_configMap.find(BC::CSV::patchver);
    if(it != d_configMap.end())
        return it->second.toInt();
    return -1;
}

QString BlackchirpCSV::releaseVersion() const
{
    auto it = d_configMap.find(BC::CSV::relver);
    if(it != d_configMap.end())
        return it->second.toString();
    return "";
}

QString BlackchirpCSV::buildVersion() const
{
    auto it = d_configMap.find(BC::CSV::buildver);
    if(it != d_configMap.end())
        return it->second.toString();
    return "";
}

bool BlackchirpCSV::exptDirExists(int num)
{
    int mil = num/1000000;
    int th = num/1000;
    SettingsStorage s;
    QDir out(s.get(BC::Key::savePath,QString("")));
    if(!out.cd(BC::Key::exptDir))
        return false;
    if(!out.cd(QString::number(mil)))
        return false;
    if(!out.cd(QString::number(th)))
        return false;
    if(!out.cd(QString::number(num)))
        return false;

    return true;
}

bool BlackchirpCSV::createExptDir(int num)
{
    QString mil = QString::number(num/1000000);
    QString th = QString::number(num/1000);
    QString n = QString::number(num);

    SettingsStorage s;
    QDir out(s.get(BC::Key::savePath,QString("")));
    if(!out.cd(BC::Key::exptDir))
        return false;

    if(!out.cd(mil))
    {
        if(!out.mkdir(mil))
            return false;
        if(!out.cd(mil))
            return false;
    }
    if(!out.cd(th))
    {
        if(!out.mkdir(th))
            return false;
        if(!out.cd(th))
            return false;
    }
    if(!out.cd(n))
    {
        if(!out.mkdir(n))
            return false;
    }

    return true;
}

QDir BlackchirpCSV::exptDir(int num, QString path)
{
    int mil = num/1000000;
    int th = num/1000;
    SettingsStorage s;
    QDir out(path.isEmpty() ? s.get(BC::Key::savePath,QString("")) : path);
    out.cd(BC::Key::exptDir);
    out.cd(QString::number(mil));
    out.cd(QString::number(th));
    out.cd(QString::number(num));

    return out;
}

QDir BlackchirpCSV::logDir()
{
    SettingsStorage s;
    QDir out(s.get(BC::Key::savePath,QString("")));
    out.cd(BC::Key::logDir);
    return out;
}

QDir BlackchirpCSV::textExportDir()
{
    SettingsStorage s;
    QDir out(s.get(BC::Key::savePath,QString("")));
    out.cd(BC::Key::exportDir);
    return out;
}

QDir BlackchirpCSV::trackingDir()
{
    SettingsStorage s;
    QDir out(s.get(BC::Key::savePath,QString("")));
    out.cd(BC::Key::trackingDir);
    return out;
}

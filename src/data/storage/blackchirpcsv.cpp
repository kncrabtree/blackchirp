#include "blackchirpcsv.h"

//#include <gui/plot/blackchirpplotcurve.h>
#include <data/storage/settingsstorage.h>

BlackchirpCSV::BlackchirpCSV()
{

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

QVariantList BlackchirpCSV::readLine(QIODevice &device)
{
    QVariantList out;
    auto l = QString(device.readLine()).trimmed();
    if(l.isEmpty())
        return out;
    auto list = l.split(BC::CSV::del);
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
    auto list = l.split(BC::CSV::del);
    for(auto &str : list)
        out << str.toLongLong(nullptr,36);

    return out;

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

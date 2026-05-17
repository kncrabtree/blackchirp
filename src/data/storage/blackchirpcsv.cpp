#include "blackchirpcsv.h"

#include <QCoreApplication>
#include <QSettings>

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

bool BlackchirpCSV::writeXY(QIODevice &device, const QVector<QPointF> d, const QString prefix, XYFormat fmt)
{
    using namespace BC::CSV;

    QTextStream t(&device);

    const QString xh = prefix.isEmpty() ? QString(x)
                                        : prefix + QString(sep) + QString(x);
    const QString yh = prefix.isEmpty() ? QString(y)
                                        : prefix + QString(sep) + QString(y);

    if(fmt == XYFormat::Aligned)
    {
        // Two in-memory columns: render every value, find the widest in
        // the first column (header included), then left-justify it so the
        // second column lines up for a human reader. Left- (not right-)
        // justified, and the last column is unpadded, so no whitespace
        // ever precedes a value — the file still parses cleanly with a
        // whitespace separator (pandas: sep=r"\s+").
        QStringList xs, ys;
        xs.reserve(d.size());
        ys.reserve(d.size());
        int wx = xh.size();
        for(const auto &p : d)
        {
            const QString sx = QVariant{p.x()}.toString();
            wx = qMax(wx, static_cast<int>(sx.size()));
            xs << sx;
            ys << QVariant{p.y()}.toString();
        }
        t << xh.leftJustified(wx) << "  " << yh;
        for(int i = 0; i < xs.size(); ++i)
            t << nl << xs.at(i).leftJustified(wx) << "  " << ys.at(i);
        return true;
    }

    QString cd;
    switch(fmt)
    {
    case XYFormat::Comma: cd = QString(","); break;
    case XYFormat::Tab:   cd = QString("\t"); break;
    case XYFormat::Aligned: // handled above
    case XYFormat::Semicolon:
    default:              cd = QString(del); break;
    }

    t << xh << cd << yh;

    for(auto it = d.constBegin(); it != d.constEnd(); it++)
        t << nl << QVariant{it->x()}.toString() << cd << QVariant{it->y()}.toString();

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

void BlackchirpCSV::writeLine(QTextStream &t, const std::vector<QVariant> l)
{
    using namespace BC::CSV;

    auto s = l.size();
    if(s == 0)
        return;

    t << l[0].toString();
    for(uint i=1; i<s; i++)
        t << del << l[i].toString();
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
#ifndef BC_VIEWER
    BlackchirpCSV::writeLine(t,{majver,BC_MAJOR_VERSION});
    BlackchirpCSV::writeLine(t,{minver,BC_MINOR_VERSION});
    BlackchirpCSV::writeLine(t,{patchver,BC_PATCH_VERSION});
    BlackchirpCSV::writeLine(t,{relver,STRINGIFY(BC_RELEASE_VERSION)});
    BlackchirpCSV::writeLine(t,{buildver,STRINGIFY(BC_BUILD_VERSION)});
#endif

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

int BlackchirpCSV::scanMaxExptNumOnDisk(const QString &basePath)
{
    QString root = basePath;
    if(root.isEmpty())
    {
        SettingsStorage s;
        root = s.get(BC::Key::savePath, QString(""));
    }

    QDir base(root);
    if(root.isEmpty() || !base.cd(BC::Key::exptDir))
        return 0;

    auto rightmostNumericChild = [](const QDir &dir) -> int {
        int best = -1;
        const auto entries = dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        for(const auto &e : entries)
        {
            bool ok = false;
            int n = e.toInt(&ok);
            if(ok && n > best)
                best = n;
        }
        return best;
    };

    int mil = rightmostNumericChild(base);
    if(mil < 0 || !base.cd(QString::number(mil)))
        return 0;

    int th = rightmostNumericChild(base);
    if(th < 0 || !base.cd(QString::number(th)))
        return 0;

    int num = rightmostNumericChild(base);
    return num > 0 ? num : 0;
}

void BlackchirpCSV::mirrorExptNumToV1Settings(int num)
{
    // v1.x predates the per-major-version applicationName convention
    // introduced in v2.x, so its QSettings store is the unsuffixed
    // "Blackchirp" file alongside v2.x's "Blackchirp2".
    SettingsStorage s;
    const QString v2Path = QDir::cleanPath(s.get(BC::Key::savePath, QString("")));

    QSettings v1(QCoreApplication::organizationName(), QLatin1StringView("Blackchirp"));
    v1.setFallbacksEnabled(false);
    v1.beginGroup(BC::Key::BC);
    const QString v1Path = QDir::cleanPath(v1.value(BC::Key::savePath, QString("")).toString());

    // exptNum is coupled to savePath: each data tree has its own
    // counter. Only mirror when v1.x and v2.x point at the same tree,
    // otherwise we would clobber v1.x's counter with a value drawn from
    // an unrelated disk and reintroduce the duplicate-number problem.
    // If v1.x has no recorded savePath (user never installed v1.x, or
    // never configured it), there is nothing to keep in sync.
    if(v1Path.isEmpty() || v2Path.isEmpty() || v1Path != v2Path)
    {
        v1.endGroup();
        return;
    }

    v1.setValue(BC::Key::exptNum, num);
    v1.endGroup();
    v1.sync();
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
    if(path.isEmpty())
    {
        out.cd(BC::Key::exptDir);
        out.cd(QString::number(mil));
        out.cd(QString::number(th));
        out.cd(QString::number(num));
    }

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

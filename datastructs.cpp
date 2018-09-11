#include "datastructs.h"

#include <QDir>

QString BlackChirp::getExptFile(int num, BlackChirp::ExptFileType t, QString path, int snapNum)
{
    QString file = QString::number(num);
    switch(t) {
    case HeaderFile:
        file.append(QString(".hdr"));
        break;
    case ChirpFile:
        file.append(QString(".chp"));
        break;
    case FidFile:
        if(snapNum >= 0)
            file.append(QString("-snap%1").arg(snapNum));
        file.append(QString(".fid"));
        if(snapNum >= 0)
            file.append(QString("~"));
        break;
    case LifFile:
        if(snapNum >= 0)
            file.append(QString("-snap%1").arg(snapNum));
        file.append(QString(".lif"));
        if(snapNum >= 0)
            file.append(QString("~"));
        break;
    case SnapFile:
        file.append(QString(".snp"));
        break;
    case TimeFile:
        file.append(QString(".tdt"));
        break;
    case LogFile:
        file.append(QString(".log"));
        break;
    case MotorFile:
        file.append(QString(".mdt"));
        break;
    }

    return getExptDir(num,path) + file;
}


QString BlackChirp::getExptDir(int num, QString path)
{
    QString out;
    if(path.isEmpty())
    {
        QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
        QString savePath = s.value(QString("savePath"),QString(".")).toString();
        int mil = num/1000000;
        int th = num/1000;

        out = savePath + QString("/experiments/%1/%2/%3/").arg(mil).arg(th).arg(num);
    }
    else
    {
        out = path;
        if(!path.endsWith(QChar('/')))
            out.append('/');
    }

    return out;

}


QString BlackChirp::headerMapToString(QMap<QString, QPair<QVariant, QString> > map)
{
    QString out;
    QString tab("\t");
    QString nl("\n");

    if(map.isEmpty())
        return out;

    auto it = map.constBegin();
    auto v = it.value();
    out.append(it.key() + tab + v.first.toString() + tab + v.second);
    it++;
    while(it != map.constEnd())
    {
        v = it.value();
        out.append(nl + it.key() + tab + v.first.toString() + tab + v.second);
        it++;
    }

    return out;
}


QString BlackChirp::channelNameLookup(QString key)
{
    QString subKey, arrayName;

    if(key.startsWith(QString("flow")))
    {
        QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
        subKey = QString("flowController/%1").arg(s.value(QString("subKey"),QString("virtual")).toString());
        arrayName = QString("channels");
    }
    else if(key.startsWith(QString("ain")))
    {
        subKey = QString("iobconfig");
        arrayName = QString("analog");
    }
    else if(key.startsWith(QString("din")))
    {
        subKey = QString("iobconfig");
        arrayName = QString("digital");
    }

    if(subKey.isEmpty() || arrayName.isEmpty())
        return QString("");

    QStringList l = key.split(QString("."));
    if(l.size() < 1)
        return QString("");

    bool ok = false;
    int index = l.at(1).trimmed().toInt(&ok);
    if(!ok)
        return QString("");

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(subKey);
    s.beginReadArray(arrayName);
    s.setArrayIndex(index);
    QString out = s.value(QString("name"),QString("")).toString();
    s.endArray();
    s.endGroup();

    return out;
}


QString BlackChirp::getExportDir()
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    QString path = s.value(QString("exportPath"),QDir::homePath()).toString();
    return path;
}


void BlackChirp::setExportDir(const QString fileName)
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    QString newPath = QFileInfo(fileName).dir().absolutePath();
    s.setValue(QString("exportPath"),newPath);
}


QString BlackChirp::clockPrettyName(BlackChirp::ClockType t)
{
    switch (t) {
    case UpConversionLO:
        return QString("Upconversion LO");
    case DownConversionLO:
        return QString("Downconversion LO");
    case AwgClock:
        return QString("AWG Reference Clock");
    case DRClock:
        return QString("Double Resonance Clock");
    case DigitizerClock:
        return QString("Digitizer Reference Clock");
    case ReferenceClock:
        return QString("Common Reference Clock");
    default:
        return QString("Unknown Type");
    }

}


QString BlackChirp::clockKey(BlackChirp::ClockType t)
{
    switch (t) {
    case UpConversionLO:
        return QString("UpLO");
    case DownConversionLO:
        return QString("DownLO");
    case AwgClock:
        return QString("AwgRef");
    case DRClock:
        return QString("DrClock");
    case DigitizerClock:
        return QString("DigRef");
    case ReferenceClock:
        return QString("ComRef");
    default:
        return QString("Unknown");
    }
}

QList<BlackChirp::ClockType> BlackChirp::allClockTypes()
{
    QList <BlackChirp::ClockType> out;
    out << UpConversionLO << DownConversionLO << AwgClock
         << DRClock << DigitizerClock << ReferenceClock;
    return out;
}

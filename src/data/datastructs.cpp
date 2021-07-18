#include <data/datastructs.h>

#include <QDir>
#include <data/storage/settingsstorage.h>

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
    case MultiFidFile:
        if(snapNum >= 0)
            file.append(QString("-snap%1").arg(snapNum));
        file.append(QString(".mfd"));
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
    case ClockFile:
        file.append(QString(".rfc"));
        break;
    }

    return getExptDir(num,path) + file;
}


QString BlackChirp::getExptDir(int num, QString path)
{
    QString out;
    if(path.isEmpty())
    {
        SettingsStorage s;
        auto savePath = s.get<QString>("savePath",".");
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

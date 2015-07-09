#include "datastructs.h"

QString BlackChirp::getExptFile(int num, BlackChirp::ExptFileType t)
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
        file.append(QString(".fid"));
        break;
    case LifFile:
        file.append(QString(".lif"));
        break;
    }

    return getExptDir(num) + file;
}


QString BlackChirp::getExptDir(int num)
{
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    QString savePath = s.value(QString("savePath"),QString(".")).toString();
    int mil = num/1000000;
    int th = num/1000;

    return savePath + QString("/experiments/%1/%2/%3/").arg(mil).arg(th).arg(num);
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

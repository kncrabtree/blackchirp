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

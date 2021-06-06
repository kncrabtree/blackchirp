#include "virtualioboard.h"

VirtualIOBoard::VirtualIOBoard(QObject *parent) :
    IOBoard(parent)
{
    d_subKey = QString("virtual");
    d_prettyName = QString("Virtual IO Board");
    d_commType = CommunicationProtocol::Virtual;
    d_threaded = false;

    //See labjacku3.cpp for an explanation of these parameters
    d_numAnalog = 4;
    d_numDigital = 16-d_numAnalog;
    d_reservedAnalog = 0;
    d_reservedDigital = 0;
}



bool VirtualIOBoard::testConnection()
{
    return true;
}

void VirtualIOBoard::initialize()
{
}

bool VirtualIOBoard::prepareForExperiment(Experiment &exp)
{
    d_config = exp.iobConfig();
    return true;
}

QList<QPair<QString, QVariant> > VirtualIOBoard::readAuxPlotData()
{
    return auxData(true);
}

QList<QPair<QString, QVariant> > VirtualIOBoard::readAuxNoPlotData()
{
    return auxData(false);
}

QList<QPair<QString, QVariant> > VirtualIOBoard::auxData(bool plot)
{
    QList<QPair<QString,QVariant>> out;

    auto it = d_config.analogList().constBegin();
    for(;it!=d_config.analogList().constEnd();it++)
    {
        auto ch = it.value();
        if(ch.enabled)
        {
            double val = static_cast<double>((qrand() % 20000) - 10000)/1000.0;
            if(ch.plot == plot)
                out.append(qMakePair(QString("ain.%1").arg(it.key()),val));
        }
    }
    it = d_config.digitalList().constBegin();
    for(;it != d_config.digitalList().constEnd(); it++)
    {
        auto ch = it.value();
        if(ch.enabled)
        {
            int val = qrand() % 2;
            if(ch.plot == plot)
                out.append(qMakePair(QString("din.%1").arg(it.key()),val));
        }
    }

    return out;
}


void VirtualIOBoard::readIOBSettings()
{
}

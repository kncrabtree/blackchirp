#include "virtualioboard.h"

#include "virtualinstrument.h"

VirtualIOBoard::VirtualIOBoard(QObject *parent) :
    IOBoard(parent)
{
    d_subKey = QString("virtual");
    d_prettyName = QString("Virtual IO Board");

    p_comm = new VirtualInstrument(d_key,this);

    //See labjacku3.cpp for an explanation of these parameters
    d_numAnalog = 4;
    d_numDigital = 16-d_numAnalog;
    d_reservedAnalog = 0;
    d_reservedDigital = 0;
}



bool VirtualIOBoard::testConnection()
{
    readSettings();
    emit connected();
    return true;
}

void VirtualIOBoard::initialize()
{
    testConnection();
}

Experiment VirtualIOBoard::prepareForExperiment(Experiment exp)
{
    d_config = exp.iobConfig();
    return exp;
}

void VirtualIOBoard::beginAcquisition()
{
}

void VirtualIOBoard::endAcquisition()
{
}

void VirtualIOBoard::readTimeData()
{
    QList<QPair<QString,QVariant>> outPlot, outNoPlot;

    auto it = d_config.analogList().constBegin();
    for(;it!=d_config.analogList().constEnd();it++)
    {
        auto ch = it.value();
        if(ch.enabled)
        {
            double val = static_cast<double>((qrand() % 20000) - 10000)/1000.0;
            if(ch.plot)
                outPlot.append(qMakePair(QString("ain.%1").arg(it.key()),val));
            else
                outNoPlot.append(qMakePair(QString("ain.%1").arg(it.key()),val));
        }
    }
    it = d_config.digitalList().constBegin();
    for(;it != d_config.digitalList().constEnd(); it++)
    {
        auto ch = it.value();
        if(ch.enabled)
        {
            int val = qrand() % 2;
            if(ch.plot)
                outPlot.append(qMakePair(QString("din.%1").arg(it.key()),val));
            else
                outNoPlot.append(qMakePair(QString("din.%1").arg(it.key()),val));
        }
    }

    emit timeDataRead(outPlot);
    emit timeDataReadNoPlot(outNoPlot);

}

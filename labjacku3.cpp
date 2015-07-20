#include "labjacku3.h"

#include "custominstrument.h"

LabjackU3::LabjackU3(QObject *parent) :
    IOBoard(parent), d_handle(nullptr), d_serialNo(3)
{
    d_subKey = QString("labjacku3");
    d_prettyName = QString("Labjack U3 IO Board");

    p_comm = new CustomInstrument(d_key,d_subKey,this);
    connect(p_comm,&CommunicationProtocol::logMessage,this,&HardwareObject::logMessage);
    connect(p_comm,&CommunicationProtocol::hardwareFailure,this,&HardwareObject::hardwareFailure);

    //note that all "reserved" channels come first!
    //any unreserved channels may be used as arbitrary validation conditions
    //For the U3, there are 16 FIO lines (FIO0-7 and EIO0-7)
    //These are indexed 0-15.
    //if numAnalog is 4, then 0-3 will be analog, and 4-15 will be digital.
    //if d_reservedAnalog is 2, then only channels 2 and 3 wil be available for use as analog validation conditions
    //similar statements apply to digital lines
    //note that in the program, DIO0 will refer to the first digital channel, but if d_numAnalog changes,
    //DIO0 will refer to a different physical pin!
    //Be wary of chaning the number of analog and digital channels

    //These are default example settings. You can override them in the settings file
    d_numAnalog = 4; //for U3-LV, can be 0-16, but numAnalog+numDigital must be <= 16. (for U3-HV, numAnalog must be >=4
    d_numDigital = 16-d_numAnalog;
    d_reservedAnalog = 0; //if you have specific channels implemented; this should be nonzero
    d_reservedDigital = 0; //if you have counters, timers, or other dedicated digital I/O lines, this should be nonzero

}

void LabjackU3::configure()
{
    if(d_handle == nullptr)
        return;
}

void LabjackU3::closeConnection()
{
    closeUSBConnection(d_handle);
    d_handle = nullptr;
}



bool LabjackU3::testConnection()
{
    readSettings();

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(d_key);
    s.beginGroup(d_subKey);

    d_serialNo = s.value(QString("serialNo"),3).toInt();
    s.endGroup();
    s.endGroup();

    if(d_handle != nullptr)
        closeConnection();

    d_handle = openUSBConnection(d_serialNo);
    if(d_handle == nullptr)
    {
        emit connected(false,QString("Could not open USB connection."));
        return false;
    }

    //get configuration info
    if(getCalibrationInfo(d_handle,&d_calInfo)< 0)
    {
        closeConnection();
        emit connected(false,QString("Could not retrieve calibration info."));
        return false;
    }

    configure();

    emit connected();
    emit logMessage(QString("ID response: %1").arg(d_calInfo.prodID));
    return true;

}

void LabjackU3::initialize()
{
    testConnection();
}

Experiment LabjackU3::prepareForExperiment(Experiment exp)
{
    d_config = exp.iobConfig();
    return exp;
}

void LabjackU3::beginAcquisition()
{
}

void LabjackU3::endAcquisition()
{
}

void LabjackU3::readTimeData()
{
    QList<QPair<QString,QVariant>> outPlot, outNoPlot;

    auto it = d_config.analogList().constBegin();


    for(;it!=d_config.analogList().constEnd();it++)
    {
        auto ch = it.value();
        if(ch.enabled)
        {
            double val;
            long error = eAIN(d_handle,&d_calInfo,1,0,it.key()+d_config.reservedAnalogChannels(),32,&val,0,0,0,0,0,0);
            if(error)
            {
                emit logMessage(QString("eAIN function call returned error code %1").arg(error),BlackChirp::LogError);
                emit hardwareFailure();
                return;
            }
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
            long val;
            long error = eDI(d_handle,1,it.key()+d_config.reservedDigitalChannels()+d_config.numAnalogChannels(),&val);
            if(error)
            {
                emit logMessage(QString("eDI function call returned error code %1").arg(error),BlackChirp::LogError);
                emit hardwareFailure();
                return;
            }
            if(ch.plot)
                outPlot.append(qMakePair(QString("din.%1").arg(it.key()),static_cast<int>(val)));
            else
                outNoPlot.append(qMakePair(QString("din.%1").arg(it.key()),static_cast<int>(val)));
        }
    }

    emit timeDataRead(outPlot);
    emit timeDataReadNoPlot(outNoPlot);
}

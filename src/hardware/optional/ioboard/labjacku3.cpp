#include "labjacku3.h"

LabjackU3::LabjackU3(QObject *parent) :
    IOBoard(BC::Key::IOB::labjacku3,BC::Key::IOB::labjacku3Name,CommunicationProtocol::Custom,parent),
    d_handle(nullptr), d_serialNo(3)
{
    //For the U3, there are 16 FIO lines (FIO0-7 and EIO0-7)
    //These are indexed 0-15.
    //if numAnalog is 4, then 0-3 will be analog, and 4-15 will be digital.
    //note that in the program, dio1 will refer to the first digital channel, but if d_numAnalog changes,
    //DIO0 will refer to a different physical pin!
    //Be wary of changing the number of analog and digital channels
    using namespace BC::Key::Digi;

    setDefault(numAnalogChannels,8);
    setDefault(numDigitalChannels,8);
    setDefault(hasAuxTriggerChannel,false);
    setDefault(minFullScale,2.44);
    setDefault(maxFullScale,2.44);
    setDefault(minVOffset,0.0);
    setDefault(maxVOffset,0.0);
    setDefault(isTriggered,false);
    setDefault(minTrigDelay,0.0);
    setDefault(maxTrigDelay,0.0);
    setDefault(minTrigLevel,0.0);
    setDefault(maxTrigLevel,0.0);
    setDefault(maxRecordLength,1);
    setDefault(canBlockAverage,false);
    setDefault(canMultiRecord,false);
    setDefault(multiBlock,false);
    setDefault(maxBytes,2);

    if(!containsArray(sampleRates))
        setArray(sampleRates,{
                     {{srText,"N/A"},{srValue,0.0}},
                 });

    if(!containsArray(BC::Key::Custom::comm))
        setArray(BC::Key::Custom::comm, {
                    {{BC::Key::Custom::key,BC::Key::IOB::serialNo},
                     {BC::Key::Custom::type,BC::Key::Custom::intKey},
                     {BC::Key::Custom::label,"Serial Number"}}
                 });

    save();
}

bool LabjackU3::configure()
{
    if(d_handle == nullptr)
    {
        emit logMessage(QString("Handle is null."),LogHandler::Error);
        return false;
    }

    long enableTimers[2] = {0,0}, enableCounters[2] = {0,0}, timerModes[2] = {0,0};
    double timerValues[2] = {0.0,0.0};
    long error = eTCConfig(d_handle,enableTimers,enableCounters,4,LJ_tc48MHZ,0,timerModes,timerValues,0,0);
    if(error)
    {
        emit logMessage(QString("eTCConfig function call returned error code %1.").arg(error),LogHandler::Error);
        return false;
    }

    return true;

}

void LabjackU3::closeConnection()
{
    closeUSBConnection(d_handle);
    d_handle = nullptr;
}


bool LabjackU3::testConnection()
{
    if(d_handle != nullptr)
        closeConnection();

    d_serialNo = getArrayValue(BC::Key::Custom::comm,0,BC::Key::IOB::serialNo,3);
    d_handle = openUSBConnection(d_serialNo);
    if(d_handle == nullptr)
    {
        d_errorString = QString("Could not open USB connection.");
        return false;
    }

    //get configuration info
    if(getCalibrationInfo(d_handle,&d_calInfo)< 0)
    {
        closeConnection();
        d_errorString = QString("Could not retrieve calibration info.");
        return false;
    }

    if(!configure())
    {
        closeConnection();
        d_errorString = QString("Could not configure.");
        return false;
    }

    emit logMessage(QString("ID response: %1").arg(d_calInfo.prodID));
    return true;

}

void LabjackU3::initialize()
{
}


std::map<int, double> LabjackU3::readAnalogChannels()
{
    std::map<int,double> out;
    for(auto const &[k,ch] : d_analogChannels)
    {
        if(ch.enabled)
        {
            double d;
            eAIN(d_handle,&d_calInfo,1,NULL,k,31,&d,0,0,0,0,0,0);
            out.insert({k,d});
        }
    }

    return out;
}

std::map<int, bool> LabjackU3::readDigitalChannels()
{
    std::map<int,bool> out;
    for(auto const &[k,ch] : d_digitalChannels)
    {
        if(ch.enabled)
        {
            long d;
            eDI(d_handle,1,k,&d);
            out.insert({k,d});
        }
    }

    return out;
}

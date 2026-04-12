#include "labjacku3.h"
#include <hardware/core/hardwareregistration.h>
#include <hardware/library/labjacklibrary.h>

// Register hardware implementation
REGISTER_HARDWARE_META(LabjackU3, "Labjack U3 IOBoard")
REGISTER_HARDWARE_PROTOCOLS(LabjackU3, CommunicationProtocol::Custom)
REGISTER_LIBRARY(LabjackU3, LabjackLibrary)
REGISTER_HARDWARE_SETTINGS(LabjackU3,
    {BC::Key::Digi::numAnalogChannels, "Analog Channels", "Number of analog input channels",
     8, 0, 128, HwSettingPriority::Required},
    {BC::Key::Digi::numDigitalChannels, "Digital Channels", "Number of digital input channels",
     8, 0, 128, HwSettingPriority::Required},
    {BC::Key::Digi::hasAuxTriggerChannel, "Aux Trigger Channel", "Has auxiliary trigger input",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::minFullScale, "Min Full Scale (V)", "Minimum full-scale voltage range",
     2.44, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxFullScale, "Max Full Scale (V)", "Maximum full-scale voltage range",
     2.44, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::minVOffset, "Min V Offset (V)", "Minimum vertical offset",
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxVOffset, "Max V Offset (V)", "Maximum vertical offset",
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::minTrigDelay, "Min Trig Delay (us)", "Minimum trigger delay in microseconds",
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxTrigDelay, "Max Trig Delay (us)", "Maximum trigger delay in microseconds",
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::minTrigLevel, "Min Trig Level (V)", "Minimum trigger threshold voltage",
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxTrigLevel, "Max Trig Level (V)", "Maximum trigger threshold voltage",
     0.0, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxRecordLength, "Max Record Length", "Maximum record length in samples",
     1, 1, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::canBlockAverage, "Block Average", "Supports block averaging",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::canMultiRecord, "Multi Record", "Supports multi-record acquisition",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::multiBlock, "Multi Block", "Can simultaneously block average and multi-record",
     false, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::Digi::maxBytes, "Max Bytes/Point", "Maximum bytes per sample",
     2, 1, 8, HwSettingPriority::Optional}
)
REGISTER_HARDWARE_ARRAY(LabjackU3, BC::Key::Digi::sampleRates,
    "Sample Rates", "Available sample rates (for IO boards typically 'N/A')",
    HwSettingPriority::Optional)
REGISTER_HARDWARE_ARRAY_ENTRY(LabjackU3, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "50 kSa/s"}, {BC::Key::Digi::srValue, 50e3}})

LabjackU3::LabjackU3(const QString& label, QObject *parent) :
    IOBoard(QString(LabjackU3::staticMetaObject.className()), label, parent),
    d_handle(nullptr), d_serialNo(3)
{
    //For the U3, there are 16 FIO lines (FIO0-7 and EIO0-7)
    //These are indexed 0-15.
    //if numAnalog is 4, then 0-3 will be analog, and 4-15 will be digital.
    //note that in the program, dio1 will refer to the first digital channel, but if d_numAnalog changes,
    //DIO0 will refer to a different physical pin!
    //Be wary of changing the number of analog and digital channels
    using namespace BC::Key::Digi;

    if(!containsArray(BC::Key::Custom::comm))
        setArray(BC::Key::Custom::comm, {
                    {{BC::Key::Custom::key,BC::Key::IOB::serialNo},
                     {BC::Key::Custom::type,BC::Key::Custom::intKey},
                     {BC::Key::Custom::label,"Serial Number"}}
                 });

    save();
}

bool LabjackU3::configure(IOBoardConfig &config)
{
    Q_UNUSED(config)
    return configureTimers();
}

bool LabjackU3::configureTimers()
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

    if(!configureTimers())
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

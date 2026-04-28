#include "labjacku3.h"
#include <hardware/core/hardwareregistration.h>
#include <hardware/library/labjacklibrary.h>

// Register hardware implementation
REGISTER_HARDWARE_META(LabjackU3, "Labjack U3 IOBoard")
REGISTER_HARDWARE_PROTOCOLS(LabjackU3, CommunicationProtocol::Custom)
REGISTER_LIBRARY(LabjackU3, LabjackLibrary)

REGISTER_HARDWARE_ARRAY(LabjackU3, BC::Key::Digi::sampleRates,
    "Sample Rates", "Available sample rates (for IO boards typically 'N/A')",
    HwSettingPriority::Optional)
REGISTER_HARDWARE_ARRAY_ENTRY(LabjackU3, BC::Key::Digi::sampleRates,
    {{BC::Key::Digi::srText, "50 kSa/s"}, {BC::Key::Digi::srValue, 50e3}})

LabjackU3::LabjackU3(const QString& label, QObject *parent) :
    IOBoard(QString(LabjackU3::staticMetaObject.className()), label, parent),
    d_handle(nullptr, nullptr), d_serialNo(3)
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
    if(!d_handle)
    {
        hwError("Handle is null."_L1);
        return false;
    }

    bool ok = BC::Labjack::configureTimers(d_handle.get(),
                                           {0L, 0L}, {0L, 0L}, 4L,
                                           BC::Labjack::Const::tc48MHZ, 0L,
                                           {0L, 0L}, {0.0, 0.0});
    if (!ok)
    {
        hwError("eTCConfig call failed."_L1);
        return false;
    }

    return true;
}

void LabjackU3::closeConnection()
{
    d_handle.reset();
}


bool LabjackU3::testConnection()
{
    if(d_handle)
        closeConnection();

    d_serialNo = getArrayValue(BC::Key::Custom::comm,0,BC::Key::IOB::serialNo,3);
    d_handle = BC::Labjack::openU3(d_serialNo);
    if(!d_handle)
    {
        d_errorString = BC::Labjack::errorString();
        if (d_errorString.isEmpty())
            d_errorString = QString("Could not open USB connection.");
        return false;
    }

    if(!configureTimers())
    {
        closeConnection();
        d_errorString = QString("Could not configure.");
        return false;
    }

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
            double v = 0.0;
            BC::Labjack::readAnalog(d_handle.get(), k, v);
            out.insert({k, v});
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
            bool b = false;
            BC::Labjack::readDigital(d_handle.get(), k, b);
            out.insert({k, b});
        }
    }

    return out;
}

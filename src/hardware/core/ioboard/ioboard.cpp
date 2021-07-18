#include <hardware/core/ioboard/ioboard.h>

IOBoard::IOBoard(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent, bool threaded, bool critical)  :
    HardwareObject(BC::Key::IOB::ioboard, subKey, name, commType, parent, threaded, critical), IOBoardConfig()
{
    using namespace BC::Key::Digi;
    setDefault(isTriggered,false);
}

IOBoard::~IOBoard()
{

}

QStringList IOBoard::validationKeys() const
{
    using namespace BC::Key::Digi;
    int an = get(numAnalogChannels,0);
    int dn = get(numDigitalChannels,0);

    QStringList out;
    out.reserve(an+dn);
    for(int i=0; i<an; ++i)
        out.append(BC::Aux::IOB::ain.arg(i+1));
    for(int i=0; i<dn; ++i)
        out.append(BC::Aux::IOB::din.arg(i+1));

    return out;
}

AuxDataStorage::AuxDataMap IOBoard::readAuxData()
{
    AuxDataStorage::AuxDataMap out;
    auto m = readAnalogChannels();
    for(auto it = m.cbegin(); it != m.cend(); ++it)
    {
        auto name = analogName(it->first);
        if(!name.isEmpty())
            out.insert({name+"."+BC::Aux::IOB::ain.arg(it->first),it->second});
        else
            out.insert({BC::Aux::IOB::ain.arg(it->first),it->second});
    }

    return out;
}

AuxDataStorage::AuxDataMap IOBoard::readValidationData()
{
    AuxDataStorage::AuxDataMap out;
    auto m = readDigitalChannels();
    for(auto it = m.cbegin(); it != m.cend(); ++it)
    {
        auto name = digitalName(it->first);
        if(!name.isEmpty())
            out.insert({name+"."+BC::Aux::IOB::din.arg(it->first),it->second});
        else
            out.insert({BC::Aux::IOB::din.arg(it->first),it->second});
    }

    return out;
}

bool IOBoard::prepareForExperiment(Experiment &exp)
{
    static_cast<IOBoardConfig&>(*this) = exp.d_iobCfg;

    for(auto it = d_analogChannels.cbegin();it!=d_analogChannels.cend();++it)
    {
        auto name = analogName(it->first);
        if(!name.isEmpty())
            exp.auxData()->registerKey(d_key,d_subKey,name+"."+BC::Aux::IOB::ain.arg(it->first));
        else
            exp.auxData()->registerKey(d_key,d_subKey,BC::Aux::IOB::ain.arg(it->first));
    }

    //note: digital channels should not be registered because they do not need to be plotted and
    //saved to disk. their only purpose is to potentially abort the experiment if they are set
    //as a validation condition.

    return true;
}

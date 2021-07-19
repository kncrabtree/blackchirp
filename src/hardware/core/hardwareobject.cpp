#include <hardware/core/hardwareobject.h>

#ifdef BC_GPIBCONTROLLER
#include <hardware/optional/gpibcontroller/gpibcontroller.h>
#endif

HardwareObject::HardwareObject(const QString key, const QString subKey, const QString name,
                               CommunicationProtocol::CommType commType,
                               QObject *parent, bool threaded, bool critical) :
    QObject(parent), SettingsStorage({key,subKey},General), d_name(name), d_key(key),
    d_subKey(subKey), d_critical(critical), d_threaded(threaded),
    d_commType(commType), d_enabledForExperiment(true),
    d_isConnected(false)
{
    set(BC::Key::HW::key,d_key); set(BC::Key::HW::name,d_name);
    set(BC::Key::HW::critical,critical);
    set(BC::Key::HW::threaded,threaded);
    set(BC::Key::HW::commType,commType);

    //it is necessary to write the subKey one level above the SettingsStorage group, which
    //is referenced to d_key/d_subKey, so that other parts of the application can determine
    //the current subKey for looking up settings.
    QSettings s(QApplication::organizationName(),QApplication::applicationName());
    s.setFallbacksEnabled(false);
    s.setValue(d_key + "/" + BC::Key::HW::subKey,d_subKey);
    s.sync();
}

HardwareObject::~HardwareObject()
{

}

QString HardwareObject::errorString()
{
    QString out = d_errorString;
    d_errorString.clear();
    return out;
}

void HardwareObject::bcInitInstrument()
{
    if(p_comm)
    {
        if(p_comm->thread() != thread())
            p_comm->moveToThread(thread());
        p_comm->initialize();
    }

    readSettings();
    initialize();
    bcTestConnection();

    connect(this,&HardwareObject::hardwareFailure,[=](){
        d_isConnected = false;
        set(BC::Key::HW::connected,false);
    });
    auto interval = get(BC::Key::HW::rInterval,0)*1000;
    if(interval > 0)
        d_rollingDataTimerId = startTimer(interval);
}

void HardwareObject::bcTestConnection()
{
    d_isConnected = false;
    readSettings();
    if(p_comm)
    {
        if(!p_comm->bcTestConnection())
        {
            emit connected(false,p_comm->errorString(),QPrivateSignal());
            return;
        }
    }
    bool success = testConnection();
    d_isConnected = success;
    set(BC::Key::HW::connected,success);
    emit connected(success,errorString(),QPrivateSignal());
}

void HardwareObject::bcReadAuxData()
{
    if(!d_isConnected)
        return;

    auto pl = readAuxData();
    if(!pl.empty())
        emit auxDataRead(pl,QPrivateSignal());

    auto vl = readValidationData();
    if(!vl.empty())
        emit validationDataRead(pl,QPrivateSignal());
}

void HardwareObject::setRollingTimerInterval(int interval)
{
    set(BC::Key::HW::rInterval,interval);
    if(d_rollingDataTimerId >= 0)
        killTimer(d_rollingDataTimerId);

    d_rollingDataTimerId = startTimer(interval*1000);
}

void HardwareObject::readSettings()
{

}


void HardwareObject::buildCommunication(QObject *gc)
{
#ifdef BC_GPIBCONTROLLER
    GpibController *c = dynamic_cast<GpibController*>(gc);
#else
    Q_UNUSED(gc)
#endif
    switch(d_commType)
    {
    case CommunicationProtocol::Rs232:
        p_comm = new Rs232Instrument(d_key,this);
        break;
    case CommunicationProtocol::Tcp:
        p_comm = new TcpInstrument(d_key,this);
        break;
#ifdef BC_GPIBCONTROLLER
    case CommunicationProtocol::Gpib:
        p_comm = new GpibInstrument(d_key,c,this);
        setParent(c);
        break;
#endif
    case CommunicationProtocol::Custom:
        p_comm = new CustomInstrument(d_key,this);
        break;
    case CommunicationProtocol::Virtual:
        p_comm = new VirtualInstrument(d_key,this);
        break;
    case CommunicationProtocol::None:
    default:
        p_comm = nullptr;
        break;
    }

    if(p_comm)
    {
        connect(p_comm,&CommunicationProtocol::logMessage,this,&HardwareObject::logMessage);
        connect(p_comm,&CommunicationProtocol::hardwareFailure,this,&HardwareObject::hardwareFailure);
    }
}

AuxDataStorage::AuxDataMap HardwareObject::readAuxData()
{
    return {};
}

AuxDataStorage::AuxDataMap HardwareObject::readValidationData()
{
    return {};
}

void HardwareObject::sleep(bool b)
{
    Q_UNUSED(b)
}


void HardwareObject::timerEvent(QTimerEvent *event)
{
    if(d_isConnected && event->timerId() == d_rollingDataTimerId)
    {
        auto rd = readAuxData();
        emit rollingDataRead(rd,QPrivateSignal());
        event->accept();
        return;
    }

    return QObject::timerEvent(event);
}

#include <src/hardware/core/hardwareobject.h>

HardwareObject::HardwareObject(const QString key, const QString subKey, const QString name,
                               CommunicationProtocol::CommType commType,
                               QObject *parent, bool threaded, bool critical) :
    QObject(parent), SettingsStorage({key,subKey},General), d_name(name), d_key(key),
    d_subKey(subKey), d_critical(critical), d_threaded(threaded),
    d_commType(commType), d_enabledForExperiment(true),
    d_isConnected(false)
{
    set(BC::Key::hwKey,d_key);
    set(BC::Key::hwName,d_name);
    set(BC::Key::hwCritical,critical);
    set(BC::Key::hwThreaded,threaded);
    set(BC::Key::hwCommType,commType);

    //it is necessary to write the subKey one level above the SettingsStorage group, which
    //is referenced to d_key/d_subKey, so that other parts of the application can determine
    //the current subKey for looking up settings.
    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    s.setValue(d_key + "/" + BC::Key::hwSubKey,d_subKey);
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
        p_comm->initialize();

    readSettings();
    initialize();
    bcTestConnection();

    connect(this,&HardwareObject::hardwareFailure,[=](){
        d_isConnected = false;
        set(BC::Key::hwConnected,false);
    });
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
    set(BC::Key::hwConnected,success);
    emit connected(success,errorString(),QPrivateSignal());
}

void HardwareObject::bcReadTimeData()
{
    if(!d_isConnected)
        return;

    auto pl = readAuxPlotData();
    if(!pl.isEmpty())
        emit timeDataRead(pl,true,QPrivateSignal());

    auto npl = readAuxNoPlotData();
    if(!npl.isEmpty())
        emit timeDataRead(npl,false,QPrivateSignal());

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

QList<QPair<QString, QVariant> > HardwareObject::readAuxPlotData()
{
    return QList<QPair<QString,QVariant>>();
}

QList<QPair<QString, QVariant> > HardwareObject::readAuxNoPlotData()
{
    return QList<QPair<QString,QVariant>>();
}

void HardwareObject::sleep(bool b)
{
    Q_UNUSED(b)
}

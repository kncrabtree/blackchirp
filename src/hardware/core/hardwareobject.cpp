#include <hardware/core/hardwareobject.h>
#include <QMetaEnum>

#ifdef BC_GPIBCONTROLLER
#include <hardware/optional/gpibcontroller/gpibcontroller.h>
#endif

HardwareObject::HardwareObject(const QString hwType, const QString subKey, const QString name,
                               CommunicationProtocol::CommType commType,
                               QObject *parent, bool threaded, bool critical, int index) :
    QObject(parent),
    SettingsStorage({BC::Key::hwKey(hwType,index),subKey},General),
    d_name(name), d_key(BC::Key::hwKey(hwType,index)),
    d_subKey(subKey), d_index(index), d_threaded(threaded), d_commType(commType),
    d_enabledForExperiment(true), d_isConnected(false), p_comm(nullptr)
{
    set(BC::Key::HW::key,d_key);
    setDefault(BC::Key::HW::name,d_name);
    setDefault(BC::Key::HW::critical,critical);
    setDefault(BC::Key::HW::rInterval,0);
    setDefault(BC::Key::HW::commType,static_cast<int>(commType));
    save();

    //it is necessary to write the subKey one level above the SettingsStorage group, which
    //is referenced to d_key/d_subKey, so that other parts of the application can determine
    //the current subKey for looking up settings.
    QSettings s(QCoreApplication::organizationName(),QCoreApplication::applicationName());
    s.setFallbacksEnabled(false);
    s.setValue(d_key + "/" + BC::Key::HW::subKey,d_subKey);
    s.sync();

    d_critical = get(BC::Key::HW::critical,critical);
    d_name = get(BC::Key::HW::name,name);

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

QVector<CommunicationProtocol::CommType> HardwareObject::supportedProtocols() const
{
    // Default implementation returns the hardcoded protocol from constructor
    return {d_commType};
}

bool HardwareObject::setCommProtocol(CommunicationProtocol::CommType commType, QObject *gc)
{
    auto commTypeEnum = QMetaEnum::fromType<CommunicationProtocol::CommType>();
    QString protocolName = commTypeEnum.valueToKey(static_cast<int>(commType));
    
    // Validate that the requested protocol is supported
    auto supported = supportedProtocols();
    if (!supported.contains(commType)) {
        d_errorString = QString("Protocol %1 not supported by %2").arg(protocolName).arg(d_name);
        emit logMessage(d_errorString, LogHandler::Error);
        return false;
    }
    
    emit logMessage(QString("Switching %1 to %2 protocol").arg(d_name).arg(protocolName), LogHandler::Normal);
    
    // Store the new protocol type in settings
    set(BC::Key::HW::commType, static_cast<int>(commType));
    d_commType = commType; // Update member variable
    
    // Rebuild communication with new protocol
    buildCommunication(gc, commType);
    
    emit logMessage(QString("Protocol switch to %1 completed for %2").arg(protocolName).arg(d_name), LogHandler::Normal);
    
    return true;
}

void HardwareObject::bcInitInstrument()
{
    // Read settings to get the correct protocol before initialization
    readAll();
    auto settingsProtocolInt = get(BC::Key::HW::commType, static_cast<int>(d_commType));
    auto settingsProtocol = static_cast<CommunicationProtocol::CommType>(settingsProtocolInt);
    
    // Check if protocol differs from what was used during construction
    if(settingsProtocol != d_commType) {
        auto commTypeEnum = QMetaEnum::fromType<CommunicationProtocol::CommType>();
        QString oldProtocolName = commTypeEnum.valueToKey(static_cast<int>(d_commType));
        QString newProtocolName = commTypeEnum.valueToKey(static_cast<int>(settingsProtocol));
        
        emit logMessage(QString("Protocol mismatch for %1: constructed as %2, settings show %3. Rebuilding communication.")
                       .arg(d_name).arg(oldProtocolName).arg(newProtocolName), LogHandler::Warning);
        
        // LIMITATION: Thread management for GPIB switching is not handled here.
        // Switching to/from GPIB protocol would require complex thread migration
        // which is not currently supported at runtime.
        if((d_commType == CommunicationProtocol::Gpib) != (settingsProtocol == CommunicationProtocol::Gpib)) {
            emit logMessage(QString("Warning: Cannot switch to/from GPIB protocol at runtime. Thread management required."), LogHandler::Error);
        } else {
            // Safe to rebuild communication for non-GPIB protocol switches
            d_commType = settingsProtocol;
            buildCommunication(parent()); // Use current parent as GPIB controller
        }
    }

    if(p_comm)
    {
        if(p_comm->thread() != thread())
            p_comm->moveToThread(thread());
        p_comm->initialize();
    }

    initialize();
    
    // Store supported protocols for UI access (after object is fully constructed)
    QVariantList protocolList;
    auto protocols = supportedProtocols();
    for(auto protocol : protocols) {
        protocolList.append(static_cast<int>(protocol));
    }
    set(BC::Key::HW::supportedProtocols, protocolList);
    save();
    
    bcTestConnection();

    connect(this,&HardwareObject::hardwareFailure,this,[this](){
        d_isConnected = false;
        set(BC::Key::HW::connected,false);
    });
}

void HardwareObject::bcTestConnection()
{
    d_isConnected = false;
    bcReadSettings();
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

void HardwareObject::bcReadSettings()
{
    readAll();
    d_name = get(BC::Key::HW::name,QString(""));
    d_critical = get(BC::Key::HW::critical,true);
    auto interval = get(BC::Key::HW::rInterval,0);

    // Check for runtime protocol changes (from HWDialog changes)
    auto settingsProtocolInt = get(BC::Key::HW::commType, static_cast<int>(d_commType));
    auto settingsProtocol = static_cast<CommunicationProtocol::CommType>(settingsProtocolInt);
    
    if(settingsProtocol != d_commType) {
        // Protocol has changed at runtime - handle threading properly
        bool success = false;
        if(thread() != QThread::currentThread()) {
            // We're being called from a different thread - use blocking queued invocation
            QMetaObject::invokeMethod(this, [this, settingsProtocol](){
                return setCommProtocol(settingsProtocol, parent());
            }, Qt::BlockingQueuedConnection, &success);
        } else {
            // Same thread - direct call is safe
            success = setCommProtocol(settingsProtocol, parent());
        }
    }

    if(d_rollingDataTimerId >= 0)
        killTimer(d_rollingDataTimerId);

    if(interval > 0)
        d_rollingDataTimerId = startTimer(interval*1000);

    readSettings();
}


void HardwareObject::buildCommunication(QObject *gc, CommunicationProtocol::CommType commType)
{
#ifdef BC_GPIBCONTROLLER
    GpibController *c = dynamic_cast<GpibController*>(gc);
#else
    Q_UNUSED(gc)
#endif

    // Use provided commType, or fall back to member variable
    CommunicationProtocol::CommType protocolType = (commType == CommunicationProtocol::None) ? d_commType : commType;

    // Clean up existing communication object
    if(p_comm) {
        p_comm->deleteLater();
        p_comm = nullptr;
    }

    switch(protocolType)
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
        
        // Initialize the communication protocol to create underlying devices
        p_comm->initialize();
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

bool HardwareObject::hwPrepareForExperiment(Experiment &exp)
{
    if(!d_isConnected)
    {
        if(!testConnection())
        {
            if(d_critical)
            {
                exp.d_errorString = QString("%1 is not connected").arg(d_name);
                return false;
            }
            else
                return true;
        }
    }

    return prepareForExperiment(exp);
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

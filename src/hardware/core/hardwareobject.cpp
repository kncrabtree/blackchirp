#include <hardware/core/hardwareobject.h>
#include <hardware/core/hardwaremanager.h>
#include <hardware/core/hardwareregistration.h>
#include <data/settings/hardwarekeys.h>
#include <QMetaEnum>

// GPIB support included
#include <hardware/optional/gpibcontroller/gpibcontroller.h>
#include <hardware/core/communication/gpibinstrument.h>

REGISTER_HARDWARE_BASE(HardwareObject,
    {BC::Key::HW::critical, "Critical Hardware",
     "If enabled, a communication failure with this device will abort any running experiment.",
     true, QVariant{}, QVariant{}, HwSettingPriority::Optional},
    {BC::Key::HW::rInterval, "Rolling Data Interval (s)",
     "Period for periodic rolling-data readback in seconds. Set to 0 to disable.",
     0, 0, QVariant{}, HwSettingPriority::Optional}
)

HardwareObject::HardwareObject(const QString& hwType, const QString& hwImpl, const QString& label, QObject *parent) :
    QObject(parent),
    SettingsStorage(QStringList{hwType + BC::Key::hwIndexSep + label}),
    d_key(hwType + BC::Key::hwIndexSep + label),
    d_model(hwImpl),
    d_threaded(false),
    d_commType(CommunicationProtocol::Virtual),
    d_enabledForExperiment(true),
    p_comm(nullptr),
    d_isConnected(false)
{
    // Set basic identifying keys
    set(BC::Key::HW::key, d_key);
    set(BC::Key::HW::model, d_model);

    // Load or set default values from settings
    setDefault(BC::Key::HW::name, QString("%1 %2 (%3)")
               .arg(hwType,label,hwImpl)); // Seed user-editable display name
    d_critical = get(BC::Key::HW::critical, true); // Default to critical

    // Look up supported protocols from registry (no vtable dispatch issue).
    // Virtual implementations may not register protocols; default to Virtual.
    auto registeredProtocols = HardwareRegistry::instance().getSupportedProtocols(hwType, hwImpl);
    if (registeredProtocols.isEmpty())
        registeredProtocols = {CommunicationProtocol::Virtual};

    // Load persisted protocol, defaulting to the first registered protocol.
    setDefault(BC::Key::HW::commType, static_cast<int>(registeredProtocols.first()));
    d_commType = static_cast<CommunicationProtocol::CommType>(
        get(BC::Key::HW::commType, static_cast<int>(registeredProtocols.first())));

    // Validate persisted value against supported protocols; reset if invalid.
    if (!registeredProtocols.contains(d_commType)) {
        d_commType = registeredProtocols.first();
        set(BC::Key::HW::commType, static_cast<int>(d_commType));
    }

    applyRegisteredSettings(hwType);

    save();
}

HardwareObject::~HardwareObject()
{
    // p_comm will be deleted automatically as a child QObject - no manual cleanup needed
}

void HardwareObject::applyRegisteredSettings(const QString& hwType)
{
    auto& reg = HardwareRegistry::instance();

    for (const auto& s : reg.getSettingDefs(hwType, d_model))
        setDefault(s.key, s.defaultValue);

    auto arrays = reg.getArraySettingDefs(hwType, d_model);
    for (auto it = arrays.cbegin(); it != arrays.cend(); ++it) {
        if (!containsArray(it.key()) && !it.value().entries.empty())
            setArray(it.key(), it.value().entries);
    }
}

QString HardwareObject::errorString()
{
    QString out = d_errorString;
    d_errorString.clear();
    return out;
}

QVector<CommunicationProtocol::CommType> HardwareObject::supportedProtocols() const
{
    auto [hwType, label] = BC::Key::parseKey(d_key);
    auto protocols = HardwareRegistry::instance().getSupportedProtocols(hwType, d_model);
    if (protocols.isEmpty())
        return {CommunicationProtocol::Virtual};
    return protocols;
}

bool HardwareObject::setCommProtocol(CommunicationProtocol::CommType commType, QObject *gc)
{
    auto commTypeEnum = QMetaEnum::fromType<CommunicationProtocol::CommType>();
    QString protocolName = commTypeEnum.valueToKey(static_cast<int>(commType));
    
    // Validate that the requested protocol is supported
    auto supported = supportedProtocols();
    if (!supported.contains(commType)) {
        d_errorString = QString("Protocol %1 not supported by %2").arg(protocolName).arg(d_key);
        bcError(d_errorString);
        return false;
    }

    bcDebug(u"Switching %1 to %2 protocol"_s.arg(d_key, protocolName));
    
    // Store the new protocol type in settings
    set(BC::Key::HW::commType, static_cast<int>(commType));
    save(); // Persist the protocol change to settings
    d_commType = commType; // Update member variable
    
    // Rebuild communication with new protocol
    buildCommunication(gc, commType);
    
    bcDebug(u"Protocol switch to %1 completed for %2"_s.arg(protocolName, d_key));
    
    return true;
}

void HardwareObject::bcInitInstrument()
{
    readAll();

    // Build communication protocol. d_commType is already validated in constructor
    // from registry + persisted settings. Pass nullptr for GPIB controller —
    // resolveGpibControllersForInstruments() will rebuild GPIB instruments with
    // the proper controller before connection testing.
    buildCommunication(nullptr);

    if(p_comm)
    {
        if(p_comm->thread() != thread())
            p_comm->moveToThread(thread());
        p_comm->initialize();
    }

    initialize();

    connect(this,&HardwareObject::hardwareFailure,this,[this](){
        d_isConnected = false;
        set(BC::Key::HW::connected,false);
    });
}

void HardwareObject::bcTestConnection()
{
    d_isConnected = false;
    bcReadSettings();
    bcDebug(u"bcTestConnection: key=%1 d_commType=%2 p_comm=%3"_s
            .arg(d_key).arg(static_cast<int>(d_commType)).arg(p_comm ? p_comm->metaObject()->className() : "null"));
    if(p_comm)
    {
        if(!p_comm->bcTestConnection())
        {
            set(BC::Key::HW::connected, false, true);
            emit connected(false,p_comm->errorString(),QPrivateSignal());
            return;
        }
    }
    bool success = testConnection();
    d_isConnected = success;
    set(BC::Key::HW::connected, success, true);
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
        emit validationDataRead(vl,QPrivateSignal());
}

void HardwareObject::bcReadSettings()
{
    readAll();
    d_critical = get(BC::Key::HW::critical,true);
    auto interval = get(BC::Key::HW::rInterval,0);

    // Reload commType so protocol changes from CommunicationDialog are picked up.
    d_commType = static_cast<CommunicationProtocol::CommType>(
        get(BC::Key::HW::commType, static_cast<int>(d_commType)));

    if(d_rollingDataTimerId >= 0)
        killTimer(d_rollingDataTimerId);

    if(interval > 0)
        d_rollingDataTimerId = startTimer(interval*1000);

    readSettings();
}


void HardwareObject::buildCommunication(QObject *gc, CommunicationProtocol::CommType commType)
{
    // GPIB support included
    GpibController *c = dynamic_cast<GpibController*>(gc);
    bcDebug(u"buildCommunication: key=%1 gc=%2 commType=%3 c=%4"_s
            .arg(d_key, gc ? gc->metaObject()->className() : "null")
            .arg(static_cast<int>(commType)).arg(c ? "valid" : "null"));
    
    // If no GPIB controller provided and we need GPIB, connection will fail during testing
    // GPIB controller resolution is handled during the synchronization phase when all hardware exists
    if (!c && (commType == CommunicationProtocol::Gpib || 
               (commType == CommunicationProtocol::None && d_commType == CommunicationProtocol::Gpib))) {
        // Defer GPIB controller resolution until connection testing phase
        // This avoids initialization order dependencies during dynamic hardware creation
    }

    // Use provided commType, or fall back to member variable
    CommunicationProtocol::CommType protocolType = (commType == CommunicationProtocol::None) ? d_commType : commType;

    // Clean up existing communication object
    if(p_comm) {
        delete p_comm;
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
    case CommunicationProtocol::Gpib:
        p_comm = new GpibInstrument(d_key,c,this);
        break;
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
                exp.d_errorString = QString("%1 is not connected").arg(d_key);
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

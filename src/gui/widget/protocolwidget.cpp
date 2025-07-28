#include <gui/widget/protocolwidget.h>
#include <hardware/core/communication/communicationprotocol.h>
#include <hardware/core/hardwareobject.h>
#include <QCoreApplication>
#include <QSettings>

void ProtocolWidget::saveProtocolSettings(CommunicationProtocol::CommType protocolType, int timeout, const QString& termChar)
{
    // Save common settings that all protocols share
    set(BC::Key::HW::commType, static_cast<int>(protocolType));
    
    // Save read options for this protocol using native group support
    QString protocolKey;
    switch(protocolType) {
    case CommunicationProtocol::Rs232:
        protocolKey = BC::Key::Comm::rs232;
        break;
    case CommunicationProtocol::Tcp:
        protocolKey = BC::Key::Comm::tcp;
        break;
    case CommunicationProtocol::Gpib:
        protocolKey = BC::Key::Comm::gpib;
        break;
    case CommunicationProtocol::Custom:
        protocolKey = BC::Key::Comm::custom;
        break;
    case CommunicationProtocol::Virtual:
        protocolKey = BC::Key::Comm::hwVirtual;
        break;
    default:
        // For None or unknown protocols, don't save read options
        break;
    }
    
    if (!protocolKey.isEmpty()) {
        // Save read options using native SettingsStorage group support
        setGroupValue(protocolKey, BC::Key::Comm::timeout, timeout);
        setGroupValue(protocolKey, BC::Key::Comm::termChar, termChar);
    }
    
    // Call derived class to save protocol-specific settings
    saveProtocolSpecificSettings();
    
    // Final save of all SettingsStorage changes
    save();
}

std::pair<int, QString> ProtocolWidget::loadProtocolReadOptions(CommunicationProtocol::CommType protocolType, int defaultTimeout, const QString& defaultTermChar)
{
    QString protocolKey;
    switch(protocolType) {
    case CommunicationProtocol::Rs232:
        protocolKey = BC::Key::Comm::rs232;
        break;
    case CommunicationProtocol::Tcp:
        protocolKey = BC::Key::Comm::tcp;
        break;
    case CommunicationProtocol::Gpib:
        protocolKey = BC::Key::Comm::gpib;
        break;
    case CommunicationProtocol::Custom:
        protocolKey = BC::Key::Comm::custom;
        break;
    case CommunicationProtocol::Virtual:
        protocolKey = BC::Key::Comm::hwVirtual;
        break;
    default:
        // For None or unknown protocols, return defaults
        return std::make_pair(defaultTimeout, defaultTermChar);
    }
    
    // Load from group settings
    int timeout = getGroupValue<int>(protocolKey, BC::Key::Comm::timeout, defaultTimeout);
    QString termChar = getGroupValue<QString>(protocolKey, BC::Key::Comm::termChar, defaultTermChar);
    
    return std::make_pair(timeout, termChar);
}
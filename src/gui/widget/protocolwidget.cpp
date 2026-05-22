#include <gui/widget/protocolwidget.h>
#include <hardware/core/communication/communicationprotocol.h>
#include <hardware/core/hardwareobject.h>
#include <data/settings/hardwarekeys.h>
#include <QCoreApplication>
#include <QSettings>

void ProtocolWidget::saveProtocolSettings(CommunicationProtocol::CommType protocolType, int timeout, const QString& termChar)
{
    // Save common settings that all protocols share
    set(BC::Key::HW::commType, static_cast<int>(protocolType));
    
    // Save read options for this protocol using native group support.
    // None/unrecognized protocols yield an empty key and are not saved.
    const QString protocolKey = CommunicationProtocol::protocolGroupKey(protocolType);
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
    // None/unrecognized protocols have no settings group; return the defaults.
    const QString protocolKey = CommunicationProtocol::protocolGroupKey(protocolType);
    if(protocolKey.isEmpty())
        return std::make_pair(defaultTimeout, defaultTermChar);

    // Load from group settings
    int timeout = getGroupValue<int>(protocolKey, BC::Key::Comm::timeout, defaultTimeout);
    QString termChar = getGroupValue<QString>(protocolKey, BC::Key::Comm::termChar, defaultTermChar);
    
    return std::make_pair(timeout, termChar);
}
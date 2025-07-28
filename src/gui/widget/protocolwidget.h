#ifndef PROTOCOLWIDGET_H
#define PROTOCOLWIDGET_H

#include <QWidget>
#include <utility>
#include <data/storage/settingsstorage.h>
#include <hardware/core/communication/communicationprotocol.h>

/**
 * @brief Pure interface for protocol-specific configuration widgets
 * 
 * Provides minimal interface for protocol-specific settings widgets used in 
 * CommunicationDialog. Each protocol type creates its own UI in the constructor
 * and implements load/save methods for protocol-specific settings only.
 * 
 * Inherits from SettingsStorage to provide direct access to hardware settings
 * using the same pattern as HwDialog.
 * 
 * Common settings (timeout, termination chars, protocol selection) are handled
 * by the parent dialog, not by these widgets.
 */
class ProtocolWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT

public:
    explicit ProtocolWidget(const QString& hwKey, QWidget *parent = nullptr) 
        : QWidget(parent), SettingsStorage(hwKey, SettingsStorage::Hardware) {}
    virtual ~ProtocolWidget() = default;

    /**
     * @brief Get the protocol type this widget configures
     * @return Communication protocol type
     */
    virtual CommunicationProtocol::CommType getProtocolType() const = 0;

    /**
     * @brief Load protocol-specific settings from storage and populate UI
     * 
     * Settings are automatically loaded from the hardware key configured in constructor.
     * Derived classes should call get() for specific protocol settings.
     */
    virtual void loadProtocolSettings() = 0;

    /**
     * @brief Save current UI values to protocol-specific settings storage
     * 
     * Base class handles saving common settings (protocol type, read options),
     * then calls saveProtocolSpecificSettings() for protocol-specific settings,
     * then performs final save().
     * 
     * @param protocolType The communication protocol type being saved
     * @param timeout Read timeout in milliseconds
     * @param termChar Read termination character
     */
    void saveProtocolSettings(CommunicationProtocol::CommType protocolType, int timeout, const QString& termChar);

protected:
    /**
     * @brief Save protocol-specific settings to storage
     * 
     * Called by saveProtocolSettings() after common settings are saved.
     * Derived classes should call set() for their specific protocol settings.
     * Base class will call save() after this method returns.
     */
    virtual void saveProtocolSpecificSettings() = 0;

    /**
     * @brief Load read options for a specific protocol using group support with fallback
     * 
     * @param protocolType The communication protocol type to load settings for
     * @param defaultTimeout Default timeout value if not found in settings
     * @param defaultTermChar Default termination character if not found in settings
     * @return Pair of (timeout, termChar) loaded from settings or defaults
     */
    std::pair<int, QString> loadProtocolReadOptions(CommunicationProtocol::CommType protocolType, int defaultTimeout = 500, const QString& defaultTermChar = QString("\n"));

signals:
    /**
     * @brief Emitted when protocol-specific settings have changed
     */
    void settingsChanged();
};

#endif // PROTOCOLWIDGET_H
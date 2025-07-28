#ifndef RS232PROTOCOLWIDGET_H
#define RS232PROTOCOLWIDGET_H

#include <gui/widget/protocolwidget.h>

class QComboBox;
class QLineEdit;
class QFormLayout;

/**
 * @brief Protocol widget for RS232/Serial communication settings
 * 
 * Handles RS232-specific settings: baud rate, data bits, parity, stop bits, device ID
 */
class Rs232ProtocolWidget : public ProtocolWidget
{
    Q_OBJECT

public:
    explicit Rs232ProtocolWidget(const QString& hwKey, QWidget *parent = nullptr);

    CommunicationProtocol::CommType getProtocolType() const override {
        return CommunicationProtocol::Rs232;
    }

    void loadProtocolSettings() override;

protected:
    void saveProtocolSpecificSettings() override;

private:
    void setupUI();
    void connectSignals();

    // RS232-specific UI controls
    QFormLayout *p_layout;
    QLineEdit *p_deviceIdEdit;
    QComboBox *p_baudRateCombo;
    QComboBox *p_dataBitsCombo;
    QComboBox *p_parityCombo;
    QComboBox *p_stopBitsCombo;
    QComboBox *p_flowControlCombo;
};

#endif // RS232PROTOCOLWIDGET_H
#ifndef TCPPROTOCOLWIDGET_H
#define TCPPROTOCOLWIDGET_H

#include <gui/widget/protocolwidget.h>

class QFormLayout;
class QLineEdit;
class QSpinBox;

/**
 * @brief TCP protocol configuration widget
 * 
 * Provides UI for configuring TCP/IP connection settings:
 * - IP address/hostname
 * - Port number
 */
class TcpProtocolWidget : public ProtocolWidget
{
    Q_OBJECT

public:
    explicit TcpProtocolWidget(const QString& hwKey, QWidget *parent = nullptr);

    // ProtocolWidget interface
    CommunicationProtocol::CommType getProtocolType() const override;
    void loadProtocolSettings() override;

protected:
    void saveProtocolSpecificSettings() override;

private:
    void setupUI();
    void connectSignals();

    QFormLayout *p_layout;
    QLineEdit *p_ipEdit;
    QSpinBox *p_portSpinBox;
};

#endif // TCPPROTOCOLWIDGET_H
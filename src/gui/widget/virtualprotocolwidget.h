#ifndef VIRTUALPROTOCOLWIDGET_H
#define VIRTUALPROTOCOLWIDGET_H

#include <gui/widget/protocolwidget.h>

class QVBoxLayout;
class QLabel;

/**
 * @brief Virtual protocol configuration widget
 * 
 * Displays a simple message indicating that no configuration is available
 * for virtual instruments since they don't perform actual communication.
 */
class VirtualProtocolWidget : public ProtocolWidget
{
    Q_OBJECT

public:
    explicit VirtualProtocolWidget(const QString& hwKey, QWidget *parent = nullptr);

    // ProtocolWidget interface
    CommunicationProtocol::CommType getProtocolType() const override;
    void loadProtocolSettings() override;

protected:
    void saveProtocolSpecificSettings() override;

private:
    void setupUI();

    QVBoxLayout *p_layout;
    QLabel *p_messageLabel;
};

#endif // VIRTUALPROTOCOLWIDGET_H
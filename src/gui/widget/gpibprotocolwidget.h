#ifndef GPIBPROTOCOLWIDGET_H
#define GPIBPROTOCOLWIDGET_H

#include <gui/widget/protocolwidget.h>

class QComboBox;
class QSpinBox;
class QFormLayout;
class QLabel;
class GpibController;

/**
 * @brief Protocol widget for GPIB communication settings
 * 
 * Handles GPIB-specific settings: address selection and controller management.
 * Future-proofed for multiple controller support.
 */
class GpibProtocolWidget : public ProtocolWidget
{
    Q_OBJECT

public:
    explicit GpibProtocolWidget(const QString& hwKey, QWidget *parent = nullptr);

    CommunicationProtocol::CommType getProtocolType() const override {
        return CommunicationProtocol::Gpib;
    }

    void loadProtocolSettings() override;
    QString selectedController() const;

protected:
    void saveProtocolSpecificSettings() override;

private:
    void setupUI();
    void connectSignals();
    void populateControllerList();

    // GPIB-specific UI controls
    QFormLayout *p_layout;
    QComboBox *p_controllerCombo;
    QSpinBox *p_addressSpinBox;
};

#endif // GPIBPROTOCOLWIDGET_H
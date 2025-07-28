#ifndef CUSTOMPROTOCOLWIDGET_H
#define CUSTOMPROTOCOLWIDGET_H

#include <gui/widget/protocolwidget.h>

class QFormLayout;
class QLineEdit;
class QSpinBox;
class QVBoxLayout;

/**
 * @brief Protocol widget for Custom communication protocol
 * 
 * This widget dynamically generates UI based on the BC::Key::Custom::comm array
 * defined by custom hardware implementations. Each custom hardware type declares
 * its required settings through SettingsStorage arrays containing field definitions
 * for strings and integers with validation parameters.
 */
class CustomProtocolWidget : public ProtocolWidget
{
    Q_OBJECT
    
public:
    explicit CustomProtocolWidget(const QString& hardwareKey, QWidget *parent = nullptr);
    
    CommunicationProtocol::CommType getProtocolType() const override { return CommunicationProtocol::Custom; }
    void loadProtocolSettings() override;
    void saveProtocolSpecificSettings() override;
    
private:
    void setupUI();
    void generateDynamicUI();
    void clearDynamicUI();
    
    // UI components
    QVBoxLayout *p_layout;
    QFormLayout *p_formLayout;
    
    // Dynamic UI tracking
    QVector<QWidget*> d_dynamicWidgets;
    QVector<QPair<QString, QLineEdit*>> d_stringFields;    // key -> line edit
    QVector<QPair<QString, QSpinBox*>> d_intFields;        // key -> spin box
};

#endif // CUSTOMPROTOCOLWIDGET_H
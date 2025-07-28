#include <gui/widget/customprotocolwidget.h>
#include <hardware/core/communication/custominstrument.h>

#include <QVBoxLayout>
#include <QFormLayout>
#include <QLineEdit>
#include <QSpinBox>
#include <QLabel>

CustomProtocolWidget::CustomProtocolWidget(const QString& hardwareKey, QWidget *parent) :
    ProtocolWidget(hardwareKey, parent)
{
    setupUI();
    generateDynamicUI();
}

void CustomProtocolWidget::setupUI()
{
    p_layout = new QVBoxLayout(this);
    
    // Header label
    auto headerLabel = new QLabel("<b>Custom Protocol Configuration</b>", this);
    p_layout->addWidget(headerLabel);
    
    // Form layout for dynamic fields
    p_formLayout = new QFormLayout();
    p_layout->addLayout(p_formLayout);
    
    p_layout->addStretch();
}

void CustomProtocolWidget::generateDynamicUI()
{
    // Clear any existing dynamic UI
    clearDynamicUI();
    
    // Check if the hardware has custom communication settings defined
    if (!containsArray(BC::Key::Custom::comm)) {
        auto noSettingsLabel = new QLabel("No custom settings defined for this hardware.", this);
        noSettingsLabel->setStyleSheet("color: gray; font-style: italic;");
        p_formLayout->addRow(noSettingsLabel);
        d_dynamicWidgets.append(noSettingsLabel);
        return;
    }
    
    // Get the communication settings array
    auto commArraySize = getArraySize(BC::Key::Custom::comm);
    
    for (std::size_t i = 0; i < commArraySize; ++i) {
        // Get field definition
        QString key = getArrayValue<QString>(BC::Key::Custom::comm, i, BC::Key::Custom::key, QString());
        QString type = getArrayValue<QString>(BC::Key::Custom::comm, i, BC::Key::Custom::type, QString());
        QString label = getArrayValue<QString>(BC::Key::Custom::comm, i, BC::Key::Custom::label, key);
        
        if (key.isEmpty() || type.isEmpty()) {
            continue;
        }
        
        if (type == BC::Key::Custom::stringKey) {
            // Create string input field
            auto lineEdit = new QLineEdit(this);
            
            // Check for maximum length constraint
            int maxLen = getArrayValue<int>(BC::Key::Custom::comm, i, BC::Key::Custom::maxLen, 100);
            lineEdit->setMaxLength(maxLen);
            
            // Add to form
            p_formLayout->addRow(label + ":", lineEdit);
            
            // Track for later access
            d_stringFields.append(qMakePair(key, lineEdit));
            d_dynamicWidgets.append(lineEdit);
            
        } else if (type == BC::Key::Custom::intKey) {
            // Create integer input field
            auto spinBox = new QSpinBox(this);
            
            // Set range constraints
            int minVal = getArrayValue<int>(BC::Key::Custom::comm, i, BC::Key::Custom::intMin, 0);
            int maxVal = getArrayValue<int>(BC::Key::Custom::comm, i, BC::Key::Custom::intMax, 100000);
            spinBox->setRange(minVal, maxVal);
            
            // Add to form
            p_formLayout->addRow(label + ":", spinBox);
            
            // Track for later access
            d_intFields.append(qMakePair(key, spinBox));
            d_dynamicWidgets.append(spinBox);
        }
    }
}

void CustomProtocolWidget::clearDynamicUI()
{
    // Remove and delete all dynamic widgets
    for (auto widget : d_dynamicWidgets) {
        p_formLayout->removeWidget(widget);
        widget->deleteLater();
    }
    
    d_dynamicWidgets.clear();
    d_stringFields.clear();
    d_intFields.clear();
}

void CustomProtocolWidget::loadProtocolSettings()
{
    // Load values from group storage into UI controls
    for (const auto& field : d_stringFields) {
        const QString& key = field.first;
        QLineEdit* lineEdit = field.second;
        
        QString value = getGroupValue<QString>(BC::Key::Comm::custom, key, QString());
        lineEdit->setText(value);
    }
    
    for (const auto& field : d_intFields) {
        const QString& key = field.first;
        QSpinBox* spinBox = field.second;
        
        int value = getGroupValue<int>(BC::Key::Comm::custom, key, spinBox->minimum());
        spinBox->setValue(value);
    }
}

void CustomProtocolWidget::saveProtocolSpecificSettings()
{
    // Save values from UI controls to group storage
    for (const auto& field : d_stringFields) {
        const QString& key = field.first;
        QLineEdit* lineEdit = field.second;
        
        setGroupValue(BC::Key::Comm::custom, key, lineEdit->text());
    }
    
    for (const auto& field : d_intFields) {
        const QString& key = field.first;
        QSpinBox* spinBox = field.second;
        
        setGroupValue(BC::Key::Comm::custom, key, spinBox->value());
    }
}
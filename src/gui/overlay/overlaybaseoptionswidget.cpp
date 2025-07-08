#include "overlaybaseoptionswidget.h"

#include <QFormLayout>
#include <QLabel>
#include <limits>

OverlayBaseOptionsWidget::OverlayBaseOptionsWidget(const QStringList &plotNames, QWidget *parent)
    : QWidget(parent)
{
    setupUI();
    
    // Populate plot names (live plots already excluded by FtmwViewWidget)
    for (const QString &name : plotNames) {
        p_plotIdComboBox->addItem(name);
    }
    
    initializeDefaults();
}

void OverlayBaseOptionsWidget::setupUI()
{
    QFormLayout *layout = new QFormLayout(this);
    
    // Label
    p_labelLineEdit = new QLineEdit(this);
    p_labelLineEdit->setPlaceholderText("Enter overlay label");
    layout->addRow("Label:", p_labelLineEdit);
    
    // Plot ID
    p_plotIdComboBox = new QComboBox(this);
    layout->addRow("Plot ID:", p_plotIdComboBox);
    
    // Y Scale
    p_yScaleSpinBox = new QDoubleSpinBox(this);
    p_yScaleSpinBox->setRange(-1e10, 1e10);
    p_yScaleSpinBox->setDecimals(4);
    p_yScaleSpinBox->setSingleStep(1.0);
    layout->addRow("Y Scale:", p_yScaleSpinBox);
    
    // Y Offset
    p_yOffsetSpinBox = new QDoubleSpinBox(this);
    p_yOffsetSpinBox->setRange(-1e10, 1e10);
    p_yOffsetSpinBox->setDecimals(4);
    p_yOffsetSpinBox->setSingleStep(1.0);
    layout->addRow("Y Offset:", p_yOffsetSpinBox);
    
    // X Offset
    p_xOffsetSpinBox = new QDoubleSpinBox(this);
    p_xOffsetSpinBox->setRange(-1e10, 1e10);
    p_xOffsetSpinBox->setDecimals(4);
    p_xOffsetSpinBox->setSingleStep(1.0);
    layout->addRow("X Offset:", p_xOffsetSpinBox);
    
    setLayout(layout);
}

void OverlayBaseOptionsWidget::initializeDefaults()
{
    // Set default values
    p_labelLineEdit->clear(); // Empty label
    p_yScaleSpinBox->setValue(1.0);
    p_yOffsetSpinBox->setValue(0.0);
    p_xOffsetSpinBox->setValue(0.0);
    
    // Set default plot to main plot (case insensitive search for "ft" and "main")
    int mainPlotIndex = -1;
    for (int i = 0; i < p_plotIdComboBox->count(); ++i) {
        QString itemText = p_plotIdComboBox->itemText(i).toLower();
        if (itemText.contains("ft") && itemText.contains("main")) {
            mainPlotIndex = i;
            break;
        }
    }
    
    if (mainPlotIndex >= 0) {
        p_plotIdComboBox->setCurrentIndex(mainPlotIndex);
    } else if (p_plotIdComboBox->count() > 0) {
        p_plotIdComboBox->setCurrentIndex(0);
    }
}

// Getters
QString OverlayBaseOptionsWidget::getLabel() const
{
    return p_labelLineEdit->text();
}

QString OverlayBaseOptionsWidget::getPlotId() const
{
    return p_plotIdComboBox->currentText();
}

double OverlayBaseOptionsWidget::getYScale() const
{
    return p_yScaleSpinBox->value();
}

double OverlayBaseOptionsWidget::getYOffset() const
{
    return p_yOffsetSpinBox->value();
}

double OverlayBaseOptionsWidget::getXOffset() const
{
    return p_xOffsetSpinBox->value();
}

// Setters
void OverlayBaseOptionsWidget::setLabel(const QString &label)
{
    p_labelLineEdit->setText(label);
}

void OverlayBaseOptionsWidget::setPlotId(const QString &plotId)
{
    int index = p_plotIdComboBox->findText(plotId);
    if (index >= 0) {
        p_plotIdComboBox->setCurrentIndex(index);
    }
}

void OverlayBaseOptionsWidget::setYScale(double yScale)
{
    p_yScaleSpinBox->setValue(yScale);
}

void OverlayBaseOptionsWidget::setYOffset(double yOffset)
{
    p_yOffsetSpinBox->setValue(yOffset);
}

void OverlayBaseOptionsWidget::setXOffset(double xOffset)
{
    p_xOffsetSpinBox->setValue(xOffset);
}

bool OverlayBaseOptionsWidget::validateSettings(QString &errorMessage, const QVector<std::shared_ptr<OverlayBase>> &existingOverlays) const
{
    QStringList errors;
    
    // Check if plot ID is selected
    if (p_plotIdComboBox->currentText().isEmpty()) {
        errors << "Please select a valid plot ID";
    }
    
    // Check that label is not empty
    QString currentLabel = getLabel();
    if (currentLabel.isEmpty()) {
        errors << "Label cannot be empty";
    } else {
        // Check for duplicate label
        for (const auto &overlay : existingOverlays) {
            if (overlay && overlay->getLabel() == currentLabel) {
                errors << QString("An overlay with the label '%1' already exists").arg(currentLabel);
                break;
            }
        }
    }
    
    // Check for reasonable scale value (not zero, not infinite)
    double yScale = getYScale();
    if (qAbs(yScale) < 1e-15) {
        errors << "Y Scale cannot be zero";
    }
    if (!qIsFinite(yScale)) {
        errors << "Y Scale must be a finite number";
    }
    
    // Check for finite offset values
    if (!qIsFinite(getYOffset())) {
        errors << "Y Offset must be a finite number";
    }
    if (!qIsFinite(getXOffset())) {
        errors << "X Offset must be a finite number";
    }
    
    if (!errors.isEmpty()) {
        errorMessage = errors.join("\n");
        return false;
    }
    
    return true;
}

void OverlayBaseOptionsWidget::applyToOverlay(std::shared_ptr<OverlayBase> overlay) const
{
    if (!overlay) {
        return;
    }
    
    overlay->setLabel(getLabel());
    overlay->setPlotId(getPlotId());
    overlay->setYScale(getYScale());
    overlay->setYOffset(getYOffset());
    overlay->setXOffset(getXOffset());
}

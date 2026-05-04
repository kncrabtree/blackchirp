#include "scientificinputwidget.h"
#include <gui/style/themecolors.h>

#include <QValidator>
#include <QTimer>
#include <QResizeEvent>
#include <QApplication>
#include <QLineEdit>
#include <QDebug>

ScientificInputWidget::ScientificInputWidget(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
    setupConnections();
    
    // Initialize validation state
    d_lastValidValue = p_spinBox->value();
    d_lastValidText = p_spinBox->textFromValue(d_lastValidValue);
    
    // Set initial size policy
    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
}

double ScientificInputWidget::value() const
{
    if (d_isValid) {
        return p_spinBox->value();
    } else {
        return d_lastValidValue;
    }
}

void ScientificInputWidget::setValue(double value)
{
    // Update both spinbox and our cached values
    p_spinBox->setValue(value);
    d_lastValidValue = value;
    d_lastValidText = p_spinBox->textFromValue(value);
    
    // Ensure we're in valid state
    if (!d_isValid) {
        updateValidationState(true);
    }
}

double ScientificInputWidget::minimum() const
{
    return p_spinBox->minimum();
}

void ScientificInputWidget::setMinimum(double min)
{
    p_spinBox->setMinimum(min);
    
    // Revalidate current input
    QTimer::singleShot(0, this, &ScientificInputWidget::validateCurrentInput);
}

double ScientificInputWidget::maximum() const
{
    return p_spinBox->maximum();
}

void ScientificInputWidget::setMaximum(double max)
{
    p_spinBox->setMaximum(max);
    
    // Revalidate current input
    QTimer::singleShot(0, this, &ScientificInputWidget::validateCurrentInput);
}

void ScientificInputWidget::setRange(double min, double max)
{
    p_spinBox->setRange(min, max);
    
    // Revalidate current input
    QTimer::singleShot(0, this, &ScientificInputWidget::validateCurrentInput);
}

double ScientificInputWidget::singleStep() const
{
    return p_spinBox->singleStep();
}

void ScientificInputWidget::setSingleStep(double step)
{
    p_spinBox->setSingleStep(step);
}

int ScientificInputWidget::displayPrecision() const
{
    return p_spinBox->displayPrecision();
}

void ScientificInputWidget::setDisplayPrecision(int precision)
{
    p_spinBox->setDisplayPrecision(precision);
}

bool ScientificInputWidget::isValid() const
{
    return d_isValid;
}

QString ScientificInputWidget::lastValidValueText() const
{
    return d_lastValidText;
}

ScientificSpinBox* ScientificInputWidget::spinBox() const
{
    return p_spinBox;
}

void ScientificInputWidget::setKeyboardTracking(bool enabled)
{
    d_keyboardTracking = enabled;
}

bool ScientificInputWidget::keyboardTracking() const
{
    return d_keyboardTracking;
}

void ScientificInputWidget::setFocus()
{
    p_spinBox->setFocus();
}

bool ScientificInputWidget::hasFocus() const
{
    return p_spinBox->hasFocus();
}

void ScientificInputWidget::selectAll()
{
    p_spinBox->selectAll();
}

void ScientificInputWidget::clear()
{
    setValue(0.0);
}

void ScientificInputWidget::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    
    // Update warning label styling if visible
    if (p_warningLabel->isVisible()) {
        updateWarningStyle();
    }
}

void ScientificInputWidget::onSpinBoxValueChanged(double value)
{
    // Only emit if we're in a valid state and not blocking signals
    if (d_isValid && !d_signalsBlocked) {
        // Update our cached valid values
        d_lastValidValue = value;
        d_lastValidText = p_spinBox->textFromValue(value);
        
        emit valueChanged(value);
    }
}

void ScientificInputWidget::onSpinBoxEditingFinished()
{
    // Validate the final input
    validateCurrentInput();
    
    // Always emit editingFinished, regardless of validity
    emit editingFinished();
}

void ScientificInputWidget::onSpinBoxTextChanged()
{
    // Only do real-time validation if keyboard tracking is enabled
    if (d_keyboardTracking) {
        QTimer::singleShot(100, this, &ScientificInputWidget::validateCurrentInput);
    }
}

void ScientificInputWidget::validateCurrentInput()
{
    // Get current text and validate it
    QString currentText = p_spinBox->lineEdit()->text();
    int pos = 0;
    
    QValidator::State state = p_spinBox->validate(currentText, pos);
    bool isCurrentlyValid = (state == QValidator::Acceptable);
    
    if (isCurrentlyValid) {
        // Try to parse the value
        double parsedValue = p_spinBox->valueFromText(currentText);
        
        // Check if it's within range
        if (parsedValue >= minimum() && parsedValue <= maximum()) {
            // Input is valid
            if (!d_isValid) {
                updateValidationState(true);
            }
            
            // Update cached values
            d_lastValidValue = parsedValue;
            d_lastValidText = currentText;
            
            // Emit value changed if keyboard tracking is enabled
            if (d_keyboardTracking && !d_signalsBlocked) {
                emit valueChanged(parsedValue);
            }
        } else {
            // Value is out of range
            if (d_isValid) {
                updateValidationState(false);
            }
        }
    } else {
        // Input is syntactically invalid
        if (d_isValid) {
            updateValidationState(false);
        }
    }
}

void ScientificInputWidget::setupUI()
{
    // Create layout
    p_layout = new QVBoxLayout(this);
    p_layout->setContentsMargins(0, 0, 0, 0);
    p_layout->setSpacing(2);
    
    // Create spinbox
    p_spinBox = new ScientificSpinBox(this);
    p_layout->addWidget(p_spinBox);
    
    // Create warning label (initially hidden)
    p_warningLabel = new QLabel(this);
    p_warningLabel->setWordWrap(true);
    p_warningLabel->setVisible(false);
    p_warningLabel->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    p_layout->addWidget(p_warningLabel);
    
    // Set initial styling
    updateWarningStyle();
}

void ScientificInputWidget::setupConnections()
{
    // Connect spinbox signals
    connect(p_spinBox, QOverload<double>::of(&ScientificSpinBox::valueChanged),
            this, &ScientificInputWidget::onSpinBoxValueChanged);
    
    connect(p_spinBox, &ScientificSpinBox::editingFinished,
            this, &ScientificInputWidget::onSpinBoxEditingFinished);
    
    // Connect to text changes for real-time validation
    connect(p_spinBox->lineEdit(), &QLineEdit::textChanged,
            this, &ScientificInputWidget::onSpinBoxTextChanged);
}

void ScientificInputWidget::updateValidationState(bool isValid)
{
    if (d_isValid == isValid) {
        return; // No change
    }
    
    d_isValid = isValid;
    
    if (isValid) {
        hideWarning();
        blockValueSignals(false);
    } else {
        QString warningMsg = formatWarningMessage(d_lastValidValue);
        showWarning(warningMsg);
        blockValueSignals(true);
    }
    
    emit validationStateChanged(isValid);
}

void ScientificInputWidget::showWarning(const QString &message)
{
    p_warningLabel->setText(message);
    p_warningLabel->setVisible(true);
    updateWarningStyle();
}

void ScientificInputWidget::hideWarning()
{
    p_warningLabel->setVisible(false);
}

void ScientificInputWidget::blockValueSignals(bool block)
{
    d_signalsBlocked = block;
}

QString ScientificInputWidget::formatWarningMessage(double lastValidValue) const
{
    QString valueText = p_spinBox->textFromValue(lastValidValue);
    return QString("Invalid input. Last valid value: %1").arg(valueText);
}

void ScientificInputWidget::updateWarningStyle()
{
    if (!p_warningLabel) {
        return;
    }
    
    // Use theme-aware styling for the warning label
    QString warningColor = ThemeColors::getCSSColor(ThemeColors::StatusWarning, this);
    QString backgroundColor = ThemeColors::getCSSColor(ThemeColors::StatusWarning, this);
    backgroundColor.replace("rgb(", "rgba(").replace(")", ", 0.1)"); // Add transparency
    
    QString style = QString(
        "QLabel {"
        "    color: %1;"
        "    background-color: %2;"
        "    border: 1px solid %1;"
        "    border-radius: 3px;"
        "    padding: 4px;"
        "    font-size: 11px;"
        "    font-weight: bold;"
        "}"
    ).arg(warningColor, backgroundColor);
    
    p_warningLabel->setStyleSheet(style);
}
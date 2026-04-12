#include "scientificspinbox.h"

#include <QApplication>
#include <QLineEdit>
#include <QFontMetrics>
#include <QWheelEvent>
#include <QRegularExpression>
#include <QDebug>
#include <cmath>
#include <algorithm>
#include <limits>

ScientificSpinBox::ScientificSpinBox(QWidget *parent)
    : QAbstractSpinBox(parent)
{
    // Set up line edit properties
    lineEdit()->setMaxLength(MAX_INPUT_LENGTH);
    
    // Connect internal signals
    connect(lineEdit(), &QLineEdit::editingFinished, this, &ScientificSpinBox::onEditingFinished);
    connect(lineEdit(), &QLineEdit::textChanged, this, &ScientificSpinBox::onTextChanged);
    
    // Initialize display
    updateDisplayText();
    lineEdit()->setText(d_cachedDisplayText);

    // Set default size policy
    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Fixed);
}

double ScientificSpinBox::value() const
{
    return d_value;
}

void ScientificSpinBox::setValue(double value)
{
    // Clamp to range
    value = std::clamp(value, d_minimum, d_maximum);
    
    if (qFuzzyCompare(d_value, value))
        return;
        
    d_value = value;
    
    // If this setValue is called programmatically (not from user editing),
    // reset precision to auto-mode so it can be detected from new input
    if (!d_isEditing) {
        // Reset to auto-precision mode for programmatic changes
        d_displayPrecision = -1;
    }
    
    updateDisplayText();
    updateEditText();
    
    if (!d_isEditing) {
        lineEdit()->setText(d_cachedDisplayText);
    }
    
    emitValueChanged();
}

double ScientificSpinBox::minimum() const
{
    return d_minimum;
}

void ScientificSpinBox::setMinimum(double min)
{
    d_minimum = min;
    if (d_value < d_minimum) {
        setValue(d_minimum);
    }
}

double ScientificSpinBox::maximum() const
{
    return d_maximum;
}

void ScientificSpinBox::setMaximum(double max)
{
    d_maximum = max;
    if (d_value > d_maximum) {
        setValue(d_maximum);
    }
}

void ScientificSpinBox::setRange(double min, double max)
{
    setMinimum(min);
    setMaximum(max);
}

double ScientificSpinBox::singleStep() const
{
    return d_singleStep;
}

void ScientificSpinBox::setSingleStep(double step)
{
    d_singleStep = std::abs(step);
}

int ScientificSpinBox::displayPrecision() const
{
    return d_displayPrecision;
}

void ScientificSpinBox::setDisplayPrecision(int precision)
{
    d_displayPrecision = std::max(-1, precision);
    updateDisplayText();
    updateEditText();
    
    if (!d_isEditing) {
        lineEdit()->setText(d_cachedDisplayText);
    }
}

void ScientificSpinBox::stepBy(int steps)
{
    if (steps == 0)
        return;
        
    double stepSize;
    
    // Check for Ctrl modifier (use wheel event modifiers if available, otherwise keyboard modifiers)
    Qt::KeyboardModifiers modifiers = (d_currentWheelModifiers != Qt::NoModifier) 
                                     ? d_currentWheelModifiers 
                                     : QApplication::keyboardModifiers();
    
    if (modifiers & Qt::ControlModifier) {
        if (qFuzzyIsNull(d_value)) {
            stepSize = 1.0; // Special case for zero
        } else {
            stepSize = std::abs(d_value) * 0.1; // 10% of current value
        }
    } else {
        stepSize = calculateStepSize(); // 1% of current value (or 0.1 for zero)
    }
    
    double newValue = d_value + (steps * stepSize);
    setValue(newValue);
}

QValidator::State ScientificSpinBox::validate(QString &input, int &pos) const
{
    Q_UNUSED(pos)
    
    initializeValidator();
    
    // Remove superscript characters for validation
    QString cleanInput = removeSuperscript(input);
    
    return d_validator->validate(cleanInput, pos);
}

void ScientificSpinBox::fixup(QString &input) const
{
    initializeValidator();
    
    // Try to fix common issues
    QString cleanInput = removeSuperscript(input);
    
    // Convert × to e for parsing
    cleanInput.replace(QString("×"), QString("e"));
    cleanInput.replace(QString("*"), QString("e"));
    
    bool ok;
    double value = cleanInput.toDouble(&ok);
    if (ok) {
        input = formatForEdit(value);
    }
}

QSize ScientificSpinBox::sizeHint() const
{
    // Use the same approach as QDoubleSpinBox for consistent sizing
    QSize size = QAbstractSpinBox::sizeHint();
    
    const QFontMetrics fm(fontMetrics());
    
    // Calculate width based on new readable format with parentheses
    // e.g., "-1.23456 × 10^(-45)" - longer than old superscript format
    QString sampleText = QString("-1.23456 × 10^(-45)");
    int width = fm.horizontalAdvance(sampleText);
    
    // Add more padding for the longer format and space for buttons
    width += 50;
    
    // Use the larger of calculated width or base size width
    size.setWidth(std::max(width, size.width()));
    
    return size;
}

QSize ScientificSpinBox::minimumSizeHint() const
{
    // Use the same approach as QDoubleSpinBox for consistent sizing
    QSize size = QAbstractSpinBox::minimumSizeHint();
    
    const QFontMetrics fm(fontMetrics());
    
    // Minimum width for basic scientific notation in new format
    QString minText = QString("-1 × 10^(-9)");
    int width = fm.horizontalAdvance(minText);
    width += 40;
    
    // Use the larger of calculated width or base size width
    size.setWidth(std::max(width, size.width()));
    
    return size;
}

void ScientificSpinBox::selectAll()
{
    lineEdit()->selectAll();
}

void ScientificSpinBox::clear()
{
    setValue(0.0);
}

QLineEdit* ScientificSpinBox::lineEdit() const
{
    return QAbstractSpinBox::lineEdit();
}

QAbstractSpinBox::StepEnabled ScientificSpinBox::stepEnabled() const
{
    StepEnabled enabled = StepNone;
    
    if (d_value > d_minimum || qFuzzyCompare(d_value, d_minimum)) {
        enabled |= StepDownEnabled;
    }
    
    if (d_value < d_maximum || qFuzzyCompare(d_value, d_maximum)) {
        enabled |= StepUpEnabled;
    }
    
    return enabled;
}

void ScientificSpinBox::keyPressEvent(QKeyEvent *event)
{
    // Handle special keys
    if (event->key() == Qt::Key_Up) {
        stepBy(1);
        return;
    } else if (event->key() == Qt::Key_Down) {
        stepBy(-1);
        return;
    }
    
    QAbstractSpinBox::keyPressEvent(event);
}

void ScientificSpinBox::focusInEvent(QFocusEvent *event)
{
    d_isEditing = true;
    updateEditText();
    lineEdit()->setText(d_cachedEditText);
    
    QAbstractSpinBox::focusInEvent(event);
    
    // Select all text for easy editing
    QMetaObject::invokeMethod(this, "selectAll", Qt::QueuedConnection);
}

void ScientificSpinBox::focusOutEvent(QFocusEvent *event)
{
    d_isEditing = false;
    
    // Try to parse current text
    QString currentText = lineEdit()->text();
    double newValue = valueFromText(currentText);
    
    // Check if the input is valid by comparing with current value
    // valueFromText returns d_value if parsing fails
    if (!qFuzzyCompare(newValue, d_value)) {
        // Detect precision from user input and update display precision
        if (d_displayPrecision < 0) { // Only in auto-precision mode
            int detectedPrecision = detectPrecision(currentText);
            d_displayPrecision = detectedPrecision;
        }
        
        setValue(newValue);
        // Note: step size automatically recalculates based on new value
        // since calculateStepSize() uses current d_value dynamically
    } else {
        // Even if value didn't change, check if precision changed
        if (d_displayPrecision < 0) { // Only in auto-precision mode
            int detectedPrecision = detectPrecision(currentText);
            if (detectedPrecision != detectPrecision(d_cachedDisplayText)) {
                d_displayPrecision = detectedPrecision;
                updateDisplayText();
                lineEdit()->setText(d_cachedDisplayText);
            }
        }
        
        if (d_displayPrecision >= 0) {
            // Restore display text with current precision
            updateDisplayText();
            lineEdit()->setText(d_cachedDisplayText);
        }
    }
    
    QAbstractSpinBox::focusOutEvent(event);
}

void ScientificSpinBox::wheelEvent(QWheelEvent *event)
{
    if (!hasFocus()) {
        event->ignore();
        return;
    }
    
    const int delta = event->angleDelta().y();
    const int steps = delta / 120; // Standard wheel step
    
    // Store the event modifiers for stepBy to use
    d_currentWheelModifiers = event->modifiers();
    
    stepBy(steps);
    
    // Clear the stored modifiers
    d_currentWheelModifiers = Qt::NoModifier;
    
    // Force immediate display update during wheel scrolling
    // Temporarily switch to display mode to show proper formatting
    bool wasEditing = d_isEditing;
    d_isEditing = false;
    updateDisplayText();
    lineEdit()->setText(d_cachedDisplayText);
    d_isEditing = wasEditing;
    
    event->accept();
}

QString ScientificSpinBox::textFromValue(double value) const
{
    if (d_isEditing) {
        return formatForEdit(value);
    } else {
        return formatForDisplay(value);
    }
}

double ScientificSpinBox::valueFromText(const QString &text) const
{
    QString cleanText = removeSuperscript(text);
    
    // Handle various input formats
    cleanText.replace(QString("×"), QString("e"));
    cleanText.replace(QString("*"), QString("e"));
    
    bool ok;
    double value = cleanText.toDouble(&ok);
    
    if (!ok) {
        return d_value; // Return current value if parsing fails
    }
    
    return std::clamp(value, d_minimum, d_maximum);
}

void ScientificSpinBox::onEditingFinished()
{
    QString currentText = lineEdit()->text();
    double newValue = valueFromText(currentText);
    
    // Detect and update precision before setting value
    if (d_displayPrecision < 0) { // Only in auto-precision mode
        int detectedPrecision = detectPrecision(currentText);
        d_displayPrecision = detectedPrecision;
    }
    
    if (!qFuzzyCompare(newValue, d_value)) {
        setValue(newValue);
    } else {
        // Even if value didn't change, update display with detected precision
        updateDisplayText();
        if (!d_isEditing) {
            lineEdit()->setText(d_cachedDisplayText);
        }
    }
    
    emit editingFinished();
}

void ScientificSpinBox::onTextChanged()
{
    // Real-time validation feedback could be added here
    // For now, we rely on focusOut validation
}

void ScientificSpinBox::initializeValidator() const
{
    if (!d_validator) {
        d_validator = std::make_unique<QDoubleValidator>(d_minimum, d_maximum, 16);
        d_validator->setNotation(QDoubleValidator::ScientificNotation);
    } else {
        d_validator->setRange(d_minimum, d_maximum, 16);
    }
}

void ScientificSpinBox::updateDisplayText()
{
    d_cachedDisplayText = formatForDisplay(d_value);
}

void ScientificSpinBox::updateEditText()
{
    d_cachedEditText = formatForEdit(d_value);
}

QString ScientificSpinBox::formatForDisplay(double value) const
{
    if (qFuzzyIsNull(value)) {
        return QString("0");
    }
    
    // Determine precision
    int precision = d_displayPrecision;
    if (precision < 0) {
        // Use a reasonable default precision for initial display
        precision = 5;
    }
    
    // Format in scientific notation
    QString scientificText = QString::number(value, 'e', precision);
    
    // Convert to readable display format: "1.23e-4" → "1.23 × 10^(-4)"
    return applySuperscript(scientificText);
}

QString ScientificSpinBox::formatForEdit(double value) const
{
    if (qFuzzyIsNull(value)) {
        return QString("0");
    }
    
    // Use scientific notation with appropriate precision
    int precision = d_displayPrecision;
    if (precision < 0) {
        // Use a reasonable default for editing
        precision = 6;
    }
    
    return QString::number(value, 'e', precision);
}

int ScientificSpinBox::detectPrecision(const QString &text) const
{
    QString cleanText = text.trimmed();
    
    // Handle display format like "1.23 × 10^(-4)" first
    cleanText = removeSuperscript(cleanText);
    
    // Find decimal point
    int decimalPos = cleanText.indexOf('.');
    if (decimalPos == -1) {
        return 0; // No decimal places
    }
    
    // Find exponent
    int ePos = cleanText.indexOf('e', Qt::CaseInsensitive);
    if (ePos == -1) {
        // Fixed notation - count digits after decimal
        QString afterDecimal = cleanText.mid(decimalPos + 1);
        // Remove any trailing zeros for actual precision
        while (afterDecimal.endsWith('0') && afterDecimal.length() > 1) {
            afterDecimal.chop(1);
        }
        return afterDecimal.length();
    }
    
    // Scientific notation - count digits between decimal and 'e'
    QString mantissaAfterDecimal = cleanText.mid(decimalPos + 1, ePos - decimalPos - 1);
    // Remove trailing zeros to get actual precision
    while (mantissaAfterDecimal.endsWith('0') && mantissaAfterDecimal.length() > 1) {
        mantissaAfterDecimal.chop(1);
    }
    return mantissaAfterDecimal.length();
}

double ScientificSpinBox::calculateStepSize() const
{
    // For dynamic precision mode, always use proportional stepping
    // regardless of d_singleStep setting
    
    // Special case for zero value
    if (qFuzzyIsNull(d_value)) {
        return 0.1;
    }
    
    // Use 1% of the current value for dynamic stepping
    return std::abs(d_value) * 0.01;
}

QString ScientificSpinBox::applySuperscript(const QString &text) const
{
    QString result = text;
    
    // Replace 'e' or 'E' with ' × 10^' using regular characters in parentheses for better readability
    QRegularExpression ePattern("[eE]([+-]?\\d+)");
    QRegularExpressionMatch match = ePattern.match(result);
    
    if (match.hasMatch()) {
        QString exponent = match.captured(1);
        
        // Use regular-sized characters with ^ and parentheses for better readability
        // e.g., "1.23 × 10^(-4)" instead of tiny superscript
        if (exponent.startsWith('+')) {
            exponent = exponent.mid(1); // Remove leading +
        }
        
        // Use parentheses for negative exponents to make them clearer
        if (exponent.startsWith('-')) {
            result.replace(match.captured(0), QString(" × 10^(%1)").arg(exponent));
        } else {
            result.replace(match.captured(0), QString(" × 10^%1").arg(exponent));
        }
    }
    
    return result;
}

QString ScientificSpinBox::removeSuperscript(const QString &text) const
{
    QString result = text;
    
    // Handle the new readable format: "1.23 × 10^(-4)" or "1.23 × 10^4"
    QRegularExpression pattern("\\s*×\\s*10\\^\\(([+-]?\\d+)\\)");
    QRegularExpressionMatch match = pattern.match(result);
    
    if (match.hasMatch()) {
        QString exponent = match.captured(1);
        result.replace(match.captured(0), QString("e%1").arg(exponent));
    } else {
        // Handle format without parentheses: "1.23 × 10^4"
        QRegularExpression simplePattern("\\s*×\\s*10\\^([+-]?\\d+)");
        QRegularExpressionMatch simpleMatch = simplePattern.match(result);
        
        if (simpleMatch.hasMatch()) {
            QString exponent = simpleMatch.captured(1);
            result.replace(simpleMatch.captured(0), QString("e%1").arg(exponent));
        }
    }
    
    // Fallback: handle old Unicode superscript format if present
    result.replace(QString("⁰"), QString("0"));
    result.replace(QString("¹"), QString("1"));
    result.replace(QString("²"), QString("2"));
    result.replace(QString("³"), QString("3"));
    result.replace(QString("⁴"), QString("4"));
    result.replace(QString("⁵"), QString("5"));
    result.replace(QString("⁶"), QString("6"));
    result.replace(QString("⁷"), QString("7"));
    result.replace(QString("⁸"), QString("8"));
    result.replace(QString("⁹"), QString("9"));
    result.replace(QString("⁺"), QString("+"));
    result.replace(QString("⁻"), QString("-"));
    
    // Replace remaining ' × 10' patterns with 'e'
    result.replace(QString(" × 10"), QString("e"));
    result.replace(QString(" x 10"), QString("e"));
    result.replace(QString("×10"), QString("e"));
    result.replace(QString("x10"), QString("e"));
    
    return result;
}

bool ScientificSpinBox::isValidInput(const QString &text) const
{
    int pos = 0;
    QString testText = text;
    return validate(testText, pos) != QValidator::Invalid;
}

void ScientificSpinBox::emitValueChanged()
{
    emit valueChanged(d_value);
}
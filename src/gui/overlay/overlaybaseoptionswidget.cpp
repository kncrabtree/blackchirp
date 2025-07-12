#include "overlaybaseoptionswidget.h"

#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QRegularExpression>
#include <limits>

OverlayBaseOptionsWidget::OverlayBaseOptionsWidget(const QStringList &plotNames, double xRangeMin, double xRangeMax, QWidget *parent)
    : QWidget(parent), d_xRangeMin(xRangeMin), d_xRangeMax(xRangeMax)
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
    
    // Sanitized filename preview
    p_sanitizedLabelDisplay = new QLabel(this);
    p_sanitizedLabelDisplay->setStyleSheet("color: #666666; font-style: italic; font-size: 11px;");
    p_sanitizedLabelDisplay->setWordWrap(false);
    p_sanitizedLabelDisplay->setMinimumWidth(200); // Prevent text wrapping
    layout->addRow("Storage Name:", p_sanitizedLabelDisplay);
    
    // Connect label changes to update sanitized preview (label doesn't emit settingsChanged)
    connect(p_labelLineEdit, &QLineEdit::textChanged, this, &OverlayBaseOptionsWidget::onLabelChanged);
    
    // Plot ID
    p_plotIdComboBox = new QComboBox(this);
    layout->addRow("Plot ID:", p_plotIdComboBox);
    
    // Y Scale
    p_yScaleSpinBox = new QDoubleSpinBox(this);
    p_yScaleSpinBox->setRange(-1e10, 1e10);
    p_yScaleSpinBox->setDecimals(4);
    p_yScaleSpinBox->setSingleStep(1.0);
    p_yScaleSpinBox->setKeyboardTracking(false); // Prevent updates while typing
    layout->addRow("Y Scale:", p_yScaleSpinBox);
    
    // Y Offset
    p_yOffsetSpinBox = new QDoubleSpinBox(this);
    p_yOffsetSpinBox->setRange(-1e10, 1e10);
    p_yOffsetSpinBox->setDecimals(4);
    p_yOffsetSpinBox->setSingleStep(1.0);
    p_yOffsetSpinBox->setKeyboardTracking(false); // Prevent updates while typing
    layout->addRow("Y Offset:", p_yOffsetSpinBox);
    
    // X Offset
    p_xOffsetSpinBox = new QDoubleSpinBox(this);
    p_xOffsetSpinBox->setRange(-1e10, 1e10);
    p_xOffsetSpinBox->setDecimals(4);
    p_xOffsetSpinBox->setSingleStep(1.0);
    p_xOffsetSpinBox->setKeyboardTracking(false); // Prevent updates while typing
    layout->addRow("X Offset:", p_xOffsetSpinBox);
    
    // Min Frequency Limit
    QWidget *minFreqWidget = new QWidget(this);
    QHBoxLayout *minFreqLayout = new QHBoxLayout(minFreqWidget);
    minFreqLayout->setContentsMargins(0, 0, 0, 0);
    p_minFreqCheckBox = new QCheckBox("Enable", this);
    p_minFreqSpinBox = new QDoubleSpinBox(this);
    p_minFreqSpinBox->setRange(-1e10, 1e10);
    p_minFreqSpinBox->setDecimals(4);
    p_minFreqSpinBox->setSingleStep(1.0);
    p_minFreqSpinBox->setSuffix(" MHz");
    p_minFreqSpinBox->setKeyboardTracking(false); // Prevent updates while typing
    minFreqLayout->addWidget(p_minFreqCheckBox);
    minFreqLayout->addWidget(p_minFreqSpinBox);
    minFreqLayout->addStretch();
    connect(p_minFreqCheckBox, &QCheckBox::toggled, p_minFreqSpinBox, &QDoubleSpinBox::setEnabled);
    layout->addRow("Min Frequency:", minFreqWidget);
    
    // Max Frequency Limit
    QWidget *maxFreqWidget = new QWidget(this);
    QHBoxLayout *maxFreqLayout = new QHBoxLayout(maxFreqWidget);
    maxFreqLayout->setContentsMargins(0, 0, 0, 0);
    p_maxFreqCheckBox = new QCheckBox("Enable", this);
    p_maxFreqSpinBox = new QDoubleSpinBox(this);
    p_maxFreqSpinBox->setRange(-1e10, 1e10);
    p_maxFreqSpinBox->setDecimals(4);
    p_maxFreqSpinBox->setSingleStep(1.0);
    p_maxFreqSpinBox->setSuffix(" MHz");
    p_maxFreqSpinBox->setKeyboardTracking(false); // Prevent updates while typing
    maxFreqLayout->addWidget(p_maxFreqCheckBox);
    maxFreqLayout->addWidget(p_maxFreqSpinBox);
    maxFreqLayout->addStretch();
    connect(p_maxFreqCheckBox, &QCheckBox::toggled, p_maxFreqSpinBox, &QDoubleSpinBox::setEnabled);
    layout->addRow("Max Frequency:", maxFreqWidget);
    
    setLayout(layout);
    
    // Connect all non-label widgets to emit settingsChanged signal for real-time updates
    connect(p_plotIdComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &OverlayBaseOptionsWidget::settingsChanged);
    
    connect(p_yScaleSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &OverlayBaseOptionsWidget::settingsChanged);
    connect(p_yOffsetSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &OverlayBaseOptionsWidget::settingsChanged);
    connect(p_xOffsetSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &OverlayBaseOptionsWidget::settingsChanged);
    
    connect(p_minFreqCheckBox, &QCheckBox::toggled,
            this, &OverlayBaseOptionsWidget::settingsChanged);
    connect(p_minFreqSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &OverlayBaseOptionsWidget::settingsChanged);
    
    connect(p_maxFreqCheckBox, &QCheckBox::toggled,
            this, &OverlayBaseOptionsWidget::settingsChanged);
    connect(p_maxFreqSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &OverlayBaseOptionsWidget::settingsChanged);
    
    // Initialize sanitized label display
    onLabelChanged();
}

void OverlayBaseOptionsWidget::initializeDefaults()
{
    // Set default values
    p_labelLineEdit->clear(); // Empty label
    p_yScaleSpinBox->setValue(1.0);
    p_yOffsetSpinBox->setValue(0.0);
    p_xOffsetSpinBox->setValue(0.0);
    
    // Set frequency limits from xRange (disabled by default)
    p_minFreqCheckBox->setChecked(false);
    p_minFreqSpinBox->setValue(d_xRangeMin);
    p_minFreqSpinBox->setEnabled(false);
    
    p_maxFreqCheckBox->setChecked(false);
    p_maxFreqSpinBox->setValue(d_xRangeMax);
    p_maxFreqSpinBox->setEnabled(false);
    
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
    return sanitizeLabel(p_labelLineEdit->text());
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
    // Trigger sanitization preview update
    onLabelChanged();
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
    
    // Check frequency limit consistency
    if (getMinFreqEnabled() && getMaxFreqEnabled()) {
        double minFreq = getMinFreqValue();
        double maxFreq = getMaxFreqValue();
        if (minFreq >= maxFreq) {
            errors << "Minimum frequency must be less than maximum frequency";
        }
    }
    
    // Check for finite frequency values
    if (getMinFreqEnabled() && !qIsFinite(getMinFreqValue())) {
        errors << "Minimum frequency must be a finite number";
    }
    if (getMaxFreqEnabled() && !qIsFinite(getMaxFreqValue())) {
        errors << "Maximum frequency must be a finite number";
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
    overlay->setMinFreqLimit(getMinFreqEnabled(), getMinFreqValue());
    overlay->setMaxFreqLimit(getMaxFreqEnabled(), getMaxFreqValue());
}

bool OverlayBaseOptionsWidget::getMinFreqEnabled() const
{
    return p_minFreqCheckBox->isChecked();
}

double OverlayBaseOptionsWidget::getMinFreqValue() const
{
    return p_minFreqSpinBox->value();
}

bool OverlayBaseOptionsWidget::getMaxFreqEnabled() const
{
    return p_maxFreqCheckBox->isChecked();
}

double OverlayBaseOptionsWidget::getMaxFreqValue() const
{
    return p_maxFreqSpinBox->value();
}

void OverlayBaseOptionsWidget::setMinFreqLimit(bool enabled, double value)
{
    p_minFreqCheckBox->setChecked(enabled);
    p_minFreqSpinBox->setValue(value);
}

void OverlayBaseOptionsWidget::setMaxFreqLimit(bool enabled, double value)
{
    p_maxFreqCheckBox->setChecked(enabled);
    p_maxFreqSpinBox->setValue(value);
}

void OverlayBaseOptionsWidget::onLabelChanged()
{
    QString label = p_labelLineEdit->text();
    QString sanitized = sanitizeLabel(label);
    
    if (sanitized.isEmpty()) {
        p_sanitizedLabelDisplay->setText("(filename will be generated)");
    } else if (sanitized == label) {
        p_sanitizedLabelDisplay->setText(sanitized);
    } else {
        p_sanitizedLabelDisplay->setText(QString("<b>%1</b><br/>"
                                                "<span style='color: #CC6600;'>Changed from: %2</span>")
                                                .arg(sanitized, label));
    }
}

QString OverlayBaseOptionsWidget::sanitizeLabel(const QString& label) const
{
    QString sanitized = label;
    
    // Remove or replace characters that are problematic for filenames
    sanitized.replace(QRegularExpression("[/\\\\:*?\"<>|]"), "_");
    
    // Trim whitespace
    sanitized = sanitized.trimmed();
    
    // Ensure it's not empty
    if (sanitized.isEmpty() && !label.isEmpty()) {
        sanitized = "overlay";
    }
    
    return sanitized;
}

#include "overlaybaseoptionswidget.h"

#include <QFormLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QRegularExpression>
#include <QPushButton>
#include <QRegularExpressionValidator>
#include <limits>
#include <gui/style/themecolors.h>

OverlayBaseOptionsWidget::OverlayBaseOptionsWidget(const QStringList &plotNames, const Ft &currentFt, QWidget *parent)
    : QWidget(parent), SettingsStorage(BC::Key::OverlayBaseOptions::key), d_currentFt(currentFt), d_hasFtData(!currentFt.isEmpty()),
      p_commentValidator(nullptr)
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
    auto mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(4); // Reduced spacing between flat GroupBoxes
    
    // Group 1: Overlay Identity
    auto identityGroup = new QGroupBox("Overlay Identity", this);
    identityGroup->setFlat(true); // Flat appearance for nested GroupBox
    auto identityLayout = new QFormLayout(identityGroup);
    identityLayout->setContentsMargins(3, 2, 3, 3); // Reduced margins for flat GroupBox
    identityLayout->setVerticalSpacing(3); // Reduced spacing for compact appearance
    
    // Label
    p_labelLineEdit = new QLineEdit(this);
    p_labelLineEdit->setPlaceholderText("Enter overlay label");
    identityLayout->addRow("Label:", p_labelLineEdit);
    
    // Sanitized filename preview
    p_sanitizedLabelDisplay = new QLabel(this);
    p_sanitizedLabelDisplay->setStyleSheet(QString("color: %1; font-style: italic; font-size: 11px;").arg(ThemeColors::getCSSColor(ThemeColors::SubtleText, this)));
    p_sanitizedLabelDisplay->setWordWrap(false);
    p_sanitizedLabelDisplay->setMinimumWidth(200); // Prevent text wrapping
    identityLayout->addRow("Storage Name:", p_sanitizedLabelDisplay);
    
    // Connect label changes to update sanitized preview (label doesn't emit settingsChanged)
    connect(p_labelLineEdit, &QLineEdit::textChanged, this, &OverlayBaseOptionsWidget::onLabelChanged);
    
    // Comment field with semicolon validation
    p_commentLineEdit = new QLineEdit(this);
    p_commentLineEdit->setPlaceholderText("Enter description or comment");
    p_commentValidator = new QRegularExpressionValidator(QRegularExpression("[^;]*"), this);
    p_commentLineEdit->setValidator(p_commentValidator);
    p_commentLineEdit->setToolTip("Comments cannot contain semicolons (;) due to file format constraints");
    identityLayout->addRow("Comment:", p_commentLineEdit);
    
    // Plot ID
    p_plotIdComboBox = new QComboBox(this);
    identityLayout->addRow("Plot ID:", p_plotIdComboBox);
    
    // Group 2: Scale & Position
    auto scaleGroup = new QGroupBox("Scale & Position", this);
    scaleGroup->setFlat(true); // Flat appearance for nested GroupBox
    auto scaleLayout = new QVBoxLayout(scaleGroup);
    scaleLayout->setContentsMargins(3, 2, 3, 3); // Reduced margins for flat GroupBox
    scaleLayout->setSpacing(3); // Reduced spacing for compact appearance
    
    // Y Scale with Invert button
    auto yScaleRow = new QHBoxLayout();
    yScaleRow->addWidget(new QLabel("Y Scale:", this));
    p_yScaleInputWidget = new ScientificInputWidget(this);
    p_yScaleInputWidget->setSingleStep(1.0);
    p_yScaleInputWidget->setKeyboardTracking(false); // Prevent updates while typing
    yScaleRow->addWidget(p_yScaleInputWidget);
    p_invertButton = new QPushButton("Invert", this);
    p_invertButton->setIcon(ThemeColors::createThemedIcon(":/icons/arrows-up-down.svg", ThemeColors::IconSecondary, this));
    p_invertButton->setMaximumWidth(60);
    yScaleRow->addWidget(p_invertButton);
    yScaleRow->addStretch();
    connect(p_invertButton, &QPushButton::clicked, this, &OverlayBaseOptionsWidget::onInvertClicked);
    scaleLayout->addLayout(yScaleRow);
    
    // Autoscale controls (positioned right below Y Scale)
    auto autoscaleRow = new QHBoxLayout();
    autoscaleRow->addWidget(new QLabel("Autoscale:", this));
    p_autoscalePercentageSpinBox = new QDoubleSpinBox(this);
    p_autoscalePercentageSpinBox->setRange(0.1, 1000.0);
    p_autoscalePercentageSpinBox->setDecimals(1);
    p_autoscalePercentageSpinBox->setSingleStep(1.0);
    p_autoscalePercentageSpinBox->setSuffix("%");
    p_autoscalePercentageSpinBox->setValue(20.0); // Default 20%
    p_autoscalePercentageSpinBox->setKeyboardTracking(false);
    autoscaleRow->addWidget(p_autoscalePercentageSpinBox);
    p_autoscaleButton = new QPushButton("Autoscale", this);
    p_autoscaleButton->setIcon(ThemeColors::createThemedIcon(":/icons/arrows-pointing-out.svg", ThemeColors::IconSecondary, this));
    p_autoscaleButton->setMaximumWidth(80);
    autoscaleRow->addWidget(p_autoscaleButton);
    autoscaleRow->addStretch();
    connect(p_autoscaleButton, &QPushButton::clicked, this, &OverlayBaseOptionsWidget::onAutoscaleClicked);
    scaleLayout->addLayout(autoscaleRow);
    
    // Y and X Offsets (side-by-side for compact layout)
    auto offsetRow = new QHBoxLayout();
    offsetRow->addWidget(new QLabel("Y Offset:", this));
    p_yOffsetSpinBox = new QDoubleSpinBox(this);
    p_yOffsetSpinBox->setRange(-1e10, 1e10);
    p_yOffsetSpinBox->setDecimals(4);
    p_yOffsetSpinBox->setSingleStep(1.0);
    p_yOffsetSpinBox->setKeyboardTracking(false); // Prevent updates while typing
    offsetRow->addWidget(p_yOffsetSpinBox);
    
    offsetRow->addWidget(new QLabel("X Offset:", this));
    p_xOffsetSpinBox = new QDoubleSpinBox(this);
    p_xOffsetSpinBox->setRange(-1e10, 1e10);
    p_xOffsetSpinBox->setDecimals(4);
    p_xOffsetSpinBox->setSingleStep(1.0);
    p_xOffsetSpinBox->setKeyboardTracking(false); // Prevent updates while typing
    offsetRow->addWidget(p_xOffsetSpinBox);
    offsetRow->addStretch();
    scaleLayout->addLayout(offsetRow);
    
    // Group 3: Frequency Limits (Collapsible)
    auto frequencyGroup = new QGroupBox("Frequency Limits", this);
    frequencyGroup->setFlat(true); // Flat appearance for nested GroupBox
    frequencyGroup->setCheckable(true);
    frequencyGroup->setChecked(false);
    auto frequencyGroupLayout = new QVBoxLayout(frequencyGroup);
    frequencyGroupLayout->setContentsMargins(3, 2, 3, 3); // Reduced margins for flat GroupBox
    
    // Collapsible content widget
    auto frequencyContentWidget = new QWidget(this);
    auto frequencyLayout = new QGridLayout(frequencyContentWidget);
    frequencyLayout->setContentsMargins(3, 2, 3, 3); // Reduced margins for flat GroupBox
    frequencyLayout->setSpacing(3); // Reduced spacing for compact appearance
    
    // Min Frequency Limit
    p_minFreqCheckBox = new QCheckBox("Enable", this);
    p_minFreqSpinBox = new QDoubleSpinBox(this);
    p_minFreqSpinBox->setRange(-1e10, 1e10);
    p_minFreqSpinBox->setDecimals(4);
    p_minFreqSpinBox->setSingleStep(1.0);
    p_minFreqSpinBox->setSuffix(" MHz");
    p_minFreqSpinBox->setKeyboardTracking(false); // Prevent updates while typing
    connect(p_minFreqCheckBox, &QCheckBox::toggled, p_minFreqSpinBox, &QDoubleSpinBox::setEnabled);
    
    auto minFreqLabel = new QLabel("Min Frequency:", this);
    minFreqLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    frequencyLayout->addWidget(minFreqLabel, 0, 0);
    frequencyLayout->addWidget(p_minFreqCheckBox, 0, 1);
    frequencyLayout->addWidget(p_minFreqSpinBox, 0, 2);
    
    // Max Frequency Limit
    p_maxFreqCheckBox = new QCheckBox("Enable", this);
    p_maxFreqSpinBox = new QDoubleSpinBox(this);
    p_maxFreqSpinBox->setRange(-1e10, 1e10);
    p_maxFreqSpinBox->setDecimals(4);
    p_maxFreqSpinBox->setSingleStep(1.0);
    p_maxFreqSpinBox->setSuffix(" MHz");
    p_maxFreqSpinBox->setKeyboardTracking(false); // Prevent updates while typing
    connect(p_maxFreqCheckBox, &QCheckBox::toggled, p_maxFreqSpinBox, &QDoubleSpinBox::setEnabled);
    
    auto maxFreqLabel = new QLabel("Max Frequency:", this);
    maxFreqLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    frequencyLayout->addWidget(maxFreqLabel, 1, 0);
    frequencyLayout->addWidget(p_maxFreqCheckBox, 1, 1);
    frequencyLayout->addWidget(p_maxFreqSpinBox, 1, 2);
    
    frequencyGroupLayout->addWidget(frequencyContentWidget);
    
    // Connect GroupBox toggled signal to hide/show content widget for true collapsible behavior
    connect(frequencyGroup, &QGroupBox::toggled, frequencyContentWidget, &QWidget::setVisible);
    
    // Initially hide content since group starts unchecked
    frequencyContentWidget->setVisible(false);
    
    // Add groups to main layout
    mainLayout->addWidget(identityGroup);
    mainLayout->addWidget(scaleGroup);
    mainLayout->addWidget(frequencyGroup);
    mainLayout->addStretch();
    
    setLayout(mainLayout);
    
    // Connect all non-label widgets to emit settingsChanged signal for real-time updates
    connect(p_plotIdComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &OverlayBaseOptionsWidget::settingsChanged);
    connect(p_commentLineEdit, &QLineEdit::editingFinished,
            this, &OverlayBaseOptionsWidget::settingsChanged);
    
    connect(p_yScaleInputWidget, QOverload<double>::of(&ScientificInputWidget::valueChanged),
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
    p_commentLineEdit->clear(); // Empty comment
    p_yScaleInputWidget->setValue(1.0);
    p_yOffsetSpinBox->setValue(0.0);
    p_xOffsetSpinBox->setValue(0.0);
    
    // Load autoscale percentage from settings
    p_autoscalePercentageSpinBox->setValue(get(BC::Key::OverlayBaseOptions::autoscalePercentage, DEFAULT_AUTOSCALE_PERCENTAGE));
    
    // Register getter for autoscale percentage
    registerGetter(BC::Key::OverlayBaseOptions::autoscalePercentage, p_autoscalePercentageSpinBox, &QDoubleSpinBox::value);
    
    // Set frequency limits from Ft data (disabled by default)
    auto xRange = d_hasFtData ? d_currentFt.xRange() : qMakePair(0.0, 1000.0);
    d_xRangeMin = xRange.first;
    d_xRangeMax = xRange.second;
    
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

QString OverlayBaseOptionsWidget::getComment() const
{
    return p_commentLineEdit->text();
}

double OverlayBaseOptionsWidget::getYScale() const
{
    return p_yScaleInputWidget->value();
}

double OverlayBaseOptionsWidget::getYOffset() const
{
    return p_yOffsetSpinBox->value();
}

double OverlayBaseOptionsWidget::getXOffset() const
{
    return p_xOffsetSpinBox->value();
}

double OverlayBaseOptionsWidget::getAutoscalePercentage() const
{
    return p_autoscalePercentageSpinBox->value();
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

void OverlayBaseOptionsWidget::setComment(const QString &comment)
{
    p_commentLineEdit->setText(comment);
}

void OverlayBaseOptionsWidget::setYScale(double yScale)
{
    p_yScaleInputWidget->setValue(yScale);
}

void OverlayBaseOptionsWidget::setYOffset(double yOffset)
{
    p_yOffsetSpinBox->setValue(yOffset);
}

void OverlayBaseOptionsWidget::setXOffset(double xOffset)
{
    p_xOffsetSpinBox->setValue(xOffset);
}

void OverlayBaseOptionsWidget::setAutoscalePercentage(double percentage)
{
    p_autoscalePercentageSpinBox->setValue(percentage);
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
    overlay->setComment(getComment());
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
    
    // Emit signal for label changes to trigger validation
    emit labelChanged();
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

void OverlayBaseOptionsWidget::setOverlayReference(std::shared_ptr<OverlayBase> overlay)
{
    d_overlayRef = overlay;
}

void OverlayBaseOptionsWidget::onAutoscaleClicked()
{
    // Check if we have the necessary data
    if (!d_overlayRef || !d_hasFtData) {
        return; // Can't autoscale without overlay and Ft data
    }
    
    // Get the maximum Y values
    double overlayYMax = d_overlayRef->yMax();
    double ftYMax = d_currentFt.yMax();
    
    // Check for valid values
    if (overlayYMax <= 0.0 || ftYMax <= 0.0) {
        return; // Can't calculate with zero or negative max values
    }
    
    // Calculate the new Y scale using the percentage
    double percentage = getAutoscalePercentage();
    double targetHeight = ftYMax * (percentage / 100.0);
    double newYScale = targetHeight / overlayYMax;
    
    // Set the new value (this will trigger settingsChanged signal)
    setYScale(newYScale);
}

void OverlayBaseOptionsWidget::onInvertClicked()
{
    // Simply multiply current Y scale by -1
    double currentYScale = getYScale();
    setYScale(-currentYScale);
}

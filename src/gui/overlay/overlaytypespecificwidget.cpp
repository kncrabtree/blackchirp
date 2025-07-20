#include "overlaytypespecificwidget.h"

OverlayTypeSpecificWidget::OverlayTypeSpecificWidget(const Ft &currentFt, QWidget *parent)
    : QWidget(parent),
      d_context(Context::Creation),
      d_currentFt(currentFt),
      p_sourceFileConfigBox(nullptr),
      p_sourceFileSettingsBox(nullptr),
      p_overlaySettingsBox(nullptr),
      d_sourceFileValid(false),
      d_sourceFileEnabled(true),
      d_settingsValid(false)
{
    // setupUI() must be called after construction since it calls virtual methods
}

bool OverlayTypeSpecificWidget::validateSourceFile()
{
    // Clear previous error message
    d_sourceFileErrorMessage.clear();
    
    // Call derived class implementation
    bool isValid = validateSourceFileImpl();
    
    // Update base class state
    bool stateChanged = (d_sourceFileValid != isValid);
    d_sourceFileValid = isValid;
    
    // Emit signal if state changed
    if (stateChanged) {
        emit sourceFileChanged();
    }
    
    return isValid;
}

bool OverlayTypeSpecificWidget::validateSettings()
{
    // Clear previous error message
    d_settingsErrorMessage.clear();
    
    // Call derived class implementation
    bool isValid = validateSettingsImpl();
    
    // Update base class state
    bool stateChanged = (d_settingsValid != isValid);
    d_settingsValid = isValid;
    
    // Emit signal if state changed (or always for settings to trigger UI updates)
    if (stateChanged) {
        emit settingsChanged();
    }
    
    return isValid;
}

void OverlayTypeSpecificWidget::configureForContext()
{
    // Call appropriate context-specific configuration
    if (isCreationContext()) {
        configureForCreationContext();
    } else if (isSettingsContext()) {
        configureForSettingsContext();
    }
    
    // Update validation states after context configuration
    updateSourceFileControls();
}

void OverlayTypeSpecificWidget::updateSourceFileControls()
{
    // Configure source file configuration box based on context
    if (isCreationContext()) {
        // Creation context: source file config is always enabled and not checkable
        p_sourceFileConfigBox->setCheckable(false);
        p_sourceFileConfigBox->setEnabled(true);
        p_sourceFileConfigBox->setTitle("Source File Configuration");
    } else if (isSettingsContext()) {
        // Settings context: source file config is checkable and optional
        p_sourceFileConfigBox->setCheckable(true);
        p_sourceFileConfigBox->setTitle("Source File Configuration (Optional)");

        auto b = p_sourceFileConfigBox->signalsBlocked();
        p_sourceFileConfigBox->blockSignals(true);
        p_sourceFileConfigBox->setChecked(d_sourceFileEnabled); // Use current state
        p_sourceFileConfigBox->blockSignals(b);
    }
    
    // Enable/disable source file settings based on validation and context
    bool sourceEnabled = isCreationContext() || d_sourceFileEnabled;
    bool settingsEnabled;
    
    if (isCreationContext()) {
        // In creation mode: settings only enabled if source file is valid
        settingsEnabled = sourceEnabled && d_sourceFileValid;
    } else {
        // In settings mode: settings enabled when source file config is checked
        settingsEnabled = d_sourceFileEnabled;
    }
    
    p_sourceFileSettingsBox->setEnabled(settingsEnabled);
    
    // Validate source file if source is enabled
    if (sourceEnabled) {
        validateSourceFile();
    }
}

void OverlayTypeSpecificWidget::onSourceFileConfigToggled(bool enabled)
{
    d_sourceFileEnabled = enabled;
    updateSourceFileControls();
    
    // Note: sourceFileEnabled is contextual and should not be persisted
    emit settingsChanged();
}

void OverlayTypeSpecificWidget::setupUI()
{
    // Create main layout
    auto mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(6);
    
    // Create the three QGroupBoxes
    p_sourceFileConfigBox = new QGroupBox("Source File Configuration", this);
    p_sourceFileSettingsBox = new QGroupBox("Source File Settings", this);
    p_overlaySettingsBox = new QGroupBox("Type-Specific Settings", this);
    
    // Let derived classes populate the group boxes
    createSourceFileConfigUI(p_sourceFileConfigBox);
    createSourceFileSettingsUI(p_sourceFileSettingsBox);
    createTypeSpecificSettingsUI(p_overlaySettingsBox);
    
    // Add group boxes to main layout
    mainLayout->addWidget(p_sourceFileConfigBox);
    mainLayout->addWidget(p_sourceFileSettingsBox);
    mainLayout->addWidget(p_overlaySettingsBox);
    
    // Connect source file config box toggle
    connect(p_sourceFileConfigBox, &QGroupBox::toggled,
            this, &OverlayTypeSpecificWidget::onSourceFileConfigToggled);
    
    // Setup connections after UI is created
    setupConnections();
}

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
    d_sourceFileValid = isValid;

    // Update settings group box enabled state without calling updateSourceFileControls()
    // (which would re-enter validateSourceFile and cause a stack overflow)
    if (p_sourceFileSettingsBox) {
        bool sourceEnabled = isCreationContext() || d_sourceFileEnabled;
        if (isCreationContext())
            p_sourceFileSettingsBox->setEnabled(sourceEnabled && d_sourceFileValid);
        else
            p_sourceFileSettingsBox->setEnabled(d_sourceFileEnabled);
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
    
    // Smart visibility and state management for Source File Settings section
    bool sourceEnabled = isCreationContext() || d_sourceFileEnabled;
    bool settingsVisible;
    bool settingsEnabled;
    
    if (isCreationContext()) {
        // Creation context: settings always visible, enabled when source file is valid
        settingsVisible = true;
        settingsEnabled = sourceEnabled && d_sourceFileValid;
    } else {
        // Settings context: settings only visible when source file config is enabled
        settingsVisible = d_sourceFileEnabled;
        settingsEnabled = d_sourceFileEnabled;
    }
    
    // Apply smart visibility
    p_sourceFileSettingsBox->setVisible(settingsVisible);
    p_sourceFileSettingsBox->setEnabled(settingsEnabled);
    
    // Smart visibility for Type-Specific settings section
    p_overlaySettingsBox->setVisible(hasTypeSpecificSettings());
    
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

void OverlayTypeSpecificWidget::configureGroupBoxAppearance(QGroupBox* groupBox)
{
    if (!groupBox) {
        return;
    }

    // Flat, single-level container: no nested frames. The logical
    // groupings inside are carried by SettingsTable section rows.
    groupBox->setFlat(true);
    groupBox->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);

    if (auto layout = groupBox->layout()) {
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(4);
    }
}

void OverlayTypeSpecificWidget::setupUI()
{
    // Create main layout with reduced spacing for compactness
    auto mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(4); // Reduced from 6 to 4 for tighter layout

    // Flat, frameless single-level containers. They retain QGroupBox
    // type so the source-file-config state machine (setCheckable /
    // setChecked / setTitle / toggled in updateSourceFileControls,
    // validateSourceFile, onSourceFileConfigToggled, configureForContext)
    // keeps identical observable behavior. The nested QGroupBox scaffold
    // is gone; subclasses populate SettingsTables instead.
    p_sourceFileConfigBox = new QGroupBox("Source File Configuration", this);
    p_sourceFileSettingsBox = new QGroupBox("Source File Settings", this);
    p_overlaySettingsBox = new QGroupBox("Type-Specific Settings", this);

    // updateSourceFileControls() drives the source-file-config title
    // ("Source File Configuration" / "Source File Configuration
    // (Optional)") and its checkable state; the containers stay flat
    // and rely on the SettingsTable section rows their subclasses build
    // for the in-region labelling that the nested boxes used to carry.
    configureGroupBoxAppearance(p_sourceFileConfigBox);
    configureGroupBoxAppearance(p_sourceFileSettingsBox);
    configureGroupBoxAppearance(p_overlaySettingsBox);

    // Let derived classes populate the containers with SettingsTables
    createSourceFileConfigUI(p_sourceFileConfigBox);
    createSourceFileSettingsUI(p_sourceFileSettingsBox);
    createTypeSpecificSettingsUI(p_overlaySettingsBox);

    // Add containers to main layout with intelligent stretch
    mainLayout->addWidget(p_sourceFileConfigBox, 0); // Fixed size for file selection
    mainLayout->addWidget(p_sourceFileSettingsBox, 0); // Fixed size for settings
    mainLayout->addWidget(p_overlaySettingsBox, 1); // Takes remaining space for previews/large content

    // Connect source file config box toggle
    connect(p_sourceFileConfigBox, &QGroupBox::toggled,
            this, &OverlayTypeSpecificWidget::onSourceFileConfigToggled);

    // Setup connections after UI is created
    setupConnections();
}

void OverlayTypeSpecificWidget::updatePathDisplayAndTooltip(QLineEdit* lineEdit, const QString &fullPath)
{
    if (!lineEdit) {
        return;
    }
    
    // Store the full path for validation purposes (separate from display)
    d_fullSourceFilePath = fullPath;
    
    // Set tooltip to show full path
    lineEdit->setToolTip(fullPath);
    
    // Display abbreviated path if too long (show end of path)
    const int maxDisplayChars = 50;
    if (fullPath.length() <= maxDisplayChars) {
        lineEdit->setText(fullPath);
    } else {
        // Show "...end_of_path" format
        QString abbreviated = "..." + fullPath.right(maxDisplayChars - 3);
        lineEdit->setText(abbreviated);
    }
}

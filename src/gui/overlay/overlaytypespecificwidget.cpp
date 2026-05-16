#include "overlaytypespecificwidget.h"

#include <QCheckBox>

#include <gui/widget/settingstable.h>

OverlayTypeSpecificWidget::OverlayTypeSpecificWidget(const Ft &currentFt, QWidget *parent)
    : QWidget(parent),
      d_context(Context::Creation),
      d_currentFt(currentFt),
      p_sourceFileConfigTable(nullptr),
      p_sourceConfigBox(nullptr),
      d_sourceConfigSection(-1),
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
    // Configure source file configuration section based on context
    if (isCreationContext()) {
        // Creation context: source file config is always enabled and not
        // checkable (a plain heading; the file picker must stay usable).
        setSourceConfigCheckable(false);
        setSourceConfigTitle("Source File Configuration");
        p_sourceFileConfigTable->setBoundRowsEnabled(d_sourceConfigSection, true);
    } else if (isSettingsContext()) {
        // Settings context: source file config is checkable and optional.
        setSourceConfigCheckable(true);
        setSourceConfigTitle("Source File Configuration (Optional)");
        setSourceConfigChecked(d_sourceFileEnabled); // Use current state
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

    // Let the subclass re-assert any dynamic row visibility it owns
    // (e.g. catalog parsed-file detail rows) after the section's
    // context state has been applied.
    refreshSourceFileConfigState();
}

void OverlayTypeSpecificWidget::onSourceFileConfigToggled(bool enabled)
{
    d_sourceFileEnabled = enabled;
    updateSourceFileControls();
    
    // Note: sourceFileEnabled is contextual and should not be persisted
    emit settingsChanged();
}

bool OverlayTypeSpecificWidget::isSourceConfigEnabled() const
{
    // A non-checkable (Creation) section is always "on"; the file
    // picker is unconditionally available there.
    if (isCreationContext())
        return true;
    return p_sourceConfigBox && p_sourceConfigBox->isChecked();
}

void OverlayTypeSpecificWidget::setSourceConfigChecked(bool checked)
{
    if (!p_sourceConfigBox)
        return;

    // Signal-blocked so neither the toggle relay nor the window-grow
    // fires for this programmatic, context-setup state application;
    // visibility is then reconciled without resizing.
    const bool b = p_sourceConfigBox->signalsBlocked();
    p_sourceConfigBox->blockSignals(true);
    p_sourceConfigBox->setChecked(checked);
    p_sourceConfigBox->blockSignals(b);
    p_sourceFileConfigTable->applySectionVisibility(d_sourceConfigSection);
}

void OverlayTypeSpecificWidget::setSourceConfigCheckable(bool checkable)
{
    p_sourceFileConfigTable->setSectionCheckable(d_sourceConfigSection,
                                                 checkable);
}

void OverlayTypeSpecificWidget::setSourceConfigTitle(const QString &title)
{
    p_sourceFileConfigTable->setSectionTitle(d_sourceConfigSection, title);
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

    // The source-file-config tier is a flat SettingsTable led by a
    // checkable section row; updateSourceFileControls() retitles it and
    // switches it between the Creation (plain heading) and Settings
    // (checkable) renderings via the setSourceConfig* helpers. The other
    // two tiers stay flat, frameless QGroupBoxes.
    p_sourceFileConfigTable = new SettingsTable(this);
    d_sourceConfigSection = p_sourceFileConfigTable->addCheckableSectionRow(
        "Source File Configuration", false, &p_sourceConfigBox);

    p_sourceFileSettingsBox = new QGroupBox("Source File Settings", this);
    p_overlaySettingsBox = new QGroupBox("Type-Specific Settings", this);

    configureGroupBoxAppearance(p_sourceFileSettingsBox);
    configureGroupBoxAppearance(p_overlaySettingsBox);

    // The subclass appends its file-selection / status rows (and any
    // dynamic detail rows) after the section row; everything it adds is
    // bound to the section so it collapses with it.
    const int firstRow = p_sourceFileConfigTable->rowCount();
    createSourceFileConfigUI(p_sourceFileConfigTable);
    const int lastRow = p_sourceFileConfigTable->rowCount();
    for (int r = firstRow; r < lastRow; ++r)
        d_sourceConfigRows.append(r);
    p_sourceFileConfigTable->bindSectionRows(d_sourceConfigSection,
                                             d_sourceConfigRows);

    createSourceFileSettingsUI(p_sourceFileSettingsBox);
    createTypeSpecificSettingsUI(p_overlaySettingsBox);

    // Add containers to main layout with intelligent stretch
    mainLayout->addWidget(p_sourceFileConfigTable, 0); // Fixed size for file selection
    mainLayout->addWidget(p_sourceFileSettingsBox, 0); // Fixed size for settings
    mainLayout->addWidget(p_overlaySettingsBox, 1); // Takes remaining space for previews/large content

    // Relay the section checkbox toggle through to the source-file
    // state machine, exactly as the QGroupBox::toggled connection did.
    connect(p_sourceConfigBox, &QCheckBox::toggled,
            this, &OverlayTypeSpecificWidget::sourceConfigToggled);
    connect(this, &OverlayTypeSpecificWidget::sourceConfigToggled,
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

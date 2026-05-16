#include "overlaytypespecificwidget.h"

#include <QCheckBox>

#include <gui/widget/settingstable.h>

OverlayTypeSpecificWidget::OverlayTypeSpecificWidget(const Ft &currentFt, QWidget *parent)
    : QWidget(parent),
      d_context(Context::Creation),
      d_currentFt(currentFt),
      p_settingsTable(nullptr),
      p_sourceConfigBox(nullptr),
      d_sourceConfigSection(-1),
      d_sourceFileSettingsSection(-1),
      d_typeSpecificSection(-1),
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

    // Update the Source File Settings tier's enabled state without
    // calling updateSourceFileControls() (which would re-enter
    // validateSourceFile and cause a stack overflow).
    if (d_sourceFileSettingsSection >= 0) {
        bool sourceEnabled = isCreationContext() || d_sourceFileEnabled;
        if (isCreationContext())
            p_settingsTable->setBoundRowsEnabled(
                d_sourceFileSettingsSection, sourceEnabled && d_sourceFileValid);
        else
            p_settingsTable->setBoundRowsEnabled(
                d_sourceFileSettingsSection, d_sourceFileEnabled);
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
        p_settingsTable->setBoundRowsEnabled(d_sourceConfigSection, true);
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
    
    // Apply smart visibility to the Source File Settings tier
    p_settingsTable->setSectionVisible(d_sourceFileSettingsSection,
                                       settingsVisible);
    p_settingsTable->setBoundRowsEnabled(d_sourceFileSettingsSection,
                                         settingsEnabled);

    // Type-Specific tier presence (static per overlay type)
    if (d_typeSpecificSection >= 0)
        p_settingsTable->setSectionVisible(d_typeSpecificSection,
                                           hasTypeSpecificSettings());
    
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
    p_settingsTable->applySectionVisibility(d_sourceConfigSection);
}

void OverlayTypeSpecificWidget::setSourceConfigCheckable(bool checkable)
{
    p_settingsTable->setSectionCheckable(d_sourceConfigSection,
                                                 checkable);
}

void OverlayTypeSpecificWidget::setSourceConfigTitle(const QString &title)
{
    p_settingsTable->setSectionTitle(d_sourceConfigSection, title);
}

void OverlayTypeSpecificWidget::setupUI()
{
    auto mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);

    // One table carries all three tiers as section rows, matching the
    // Base Options / Curve Appearance panels. The minimum width keeps
    // the status / path cells from clipping at the dialog's content
    // size (mirrors OverlayBaseOptionsWidget).
    p_settingsTable = new SettingsTable(this);
    p_settingsTable->setMinimumWidth(380);

    auto bindTier = [this](int section, auto populate) {
        const int first = p_settingsTable->rowCount();
        populate(p_settingsTable);
        const int last = p_settingsTable->rowCount();
        QList<int> rows;
        for (int r = first; r < last; ++r)
            rows.append(r);
        p_settingsTable->bindSectionRows(section, rows);
        return rows;
    };

    // Source File Configuration — checkable; updateSourceFileControls()
    // switches it between the Creation (plain heading) and Settings
    // (checkable) renderings via the setSourceConfig* helpers.
    d_sourceConfigSection = p_settingsTable->addCheckableSectionRow(
        "Source File Configuration", false, &p_sourceConfigBox);
    d_sourceConfigRows = bindTier(d_sourceConfigSection,
        [this](SettingsTable *t) { populateSourceFileConfigRows(t); });

    // Source File Settings — plain heading; shown/hidden and
    // enabled/disabled as a unit by the state machine.
    d_sourceFileSettingsSection =
        p_settingsTable->addSectionRow("Source File Settings");
    bindTier(d_sourceFileSettingsSection,
        [this](SettingsTable *t) { populateSourceFileSettingsRows(t); });

    // Type-Specific Settings — added only when the subclass actually
    // has any (BCExp has none); a subclass may retitle the section.
    if (hasTypeSpecificSettings()) {
        d_typeSpecificSection =
            p_settingsTable->addSectionRow("Type-Specific Settings");
        bindTier(d_typeSpecificSection,
            [this](SettingsTable *t) { populateTypeSpecificRows(t); });
    }

    mainLayout->addWidget(p_settingsTable);

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

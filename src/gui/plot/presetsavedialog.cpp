#include "presetsavedialog.h"
#include "curveappearancepresetmanager.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QRadioButton>
#include <QLineEdit>
#include <QComboBox>
#include <QLabel>
#include <QPushButton>
#include <QButtonGroup>
#include <QDialogButtonBox>
#include <QFrame>

PresetSaveDialog::PresetSaveDialog(const QString &suggestedName, 
                                   CurveAppearancePresetManager *presetManager,
                                   QWidget *parent)
    : QDialog(parent), d_suggestedName(suggestedName), p_presetManager(presetManager)
{
    setWindowTitle("Save Curve Appearance Preset");
    setModal(true);
    setupUI();
    populateExistingPresets();
    
    // Set default mode based on whether there are existing presets
    if (p_existingPresetCombo->count() > 0) {
        p_createNewRadio->setChecked(true);
    } else {
        p_createNewRadio->setChecked(true);
        p_overwriteExistingRadio->setEnabled(false);
    }
    
    onModeChanged();
    updateOkButtonState();
}

void PresetSaveDialog::setupUI()
{
    p_mainLayout = new QVBoxLayout(this);
    
    // Title label
    QLabel *titleLabel = new QLabel("Choose how to save the current curve appearance:", this);
    titleLabel->setWordWrap(true);
    p_mainLayout->addWidget(titleLabel);
    
    p_mainLayout->addSpacing(10);
    
    // Mode selection radio buttons
    p_modeGroup = new QButtonGroup(this);
    
    p_createNewRadio = new QRadioButton("Create new preset", this);
    p_overwriteExistingRadio = new QRadioButton("Overwrite existing preset", this);
    
    p_modeGroup->addButton(p_createNewRadio, 0);
    p_modeGroup->addButton(p_overwriteExistingRadio, 1);
    
    p_mainLayout->addWidget(p_createNewRadio);
    p_mainLayout->addWidget(p_overwriteExistingRadio);
    
    p_mainLayout->addSpacing(15);
    
    // Create new preset section
    QFrame *newPresetFrame = new QFrame(this);
    newPresetFrame->setFrameStyle(QFrame::StyledPanel);
    newPresetFrame->setLineWidth(1);
    
    QFormLayout *newPresetLayout = new QFormLayout(newPresetFrame);
    newPresetLayout->setContentsMargins(10, 10, 10, 10);
    
    p_newNameEdit = new QLineEdit(d_suggestedName, this);
    p_newNameEdit->setPlaceholderText("Enter new preset name...");
    newPresetLayout->addRow("Preset name:", p_newNameEdit);
    
    p_mainLayout->addWidget(newPresetFrame);
    
    // Overwrite existing preset section
    QFrame *overwriteFrame = new QFrame(this);
    overwriteFrame->setFrameStyle(QFrame::StyledPanel);
    overwriteFrame->setLineWidth(1);
    
    QFormLayout *overwriteLayout = new QFormLayout(overwriteFrame);
    overwriteLayout->setContentsMargins(10, 10, 10, 10);
    
    p_existingPresetCombo = new QComboBox(this);
    p_existingPresetCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    overwriteLayout->addRow("Select preset:", p_existingPresetCombo);
    
    p_mainLayout->addWidget(overwriteFrame);
    
    p_mainLayout->addSpacing(20);
    
    // Dialog buttons
    QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    p_okButton = buttonBox->button(QDialogButtonBox::Ok);
    p_cancelButton = buttonBox->button(QDialogButtonBox::Cancel);
    
    p_okButton->setText("Save Preset");
    p_cancelButton->setText("Cancel");
    
    p_mainLayout->addWidget(buttonBox);
    
    // Connect signals
    connect(p_createNewRadio, &QRadioButton::toggled, this, &PresetSaveDialog::onModeChanged);
    connect(p_overwriteExistingRadio, &QRadioButton::toggled, this, &PresetSaveDialog::onModeChanged);
    connect(p_newNameEdit, &QLineEdit::textChanged, this, &PresetSaveDialog::onNewNameChanged);
    connect(p_existingPresetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &PresetSaveDialog::onExistingPresetChanged);
    
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    
    // Set reasonable minimum size
    setMinimumWidth(400);
}

void PresetSaveDialog::populateExistingPresets()
{
    if (!p_presetManager) {
        return;
    }
    
    p_existingPresetCombo->clear();
    
    QStringList presetNames = p_presetManager->getPresetNames();
    for (const QString &name : presetNames) {
        auto preset = p_presetManager->getPreset(name);
        
        // Add icon to distinguish default vs custom presets
        QString displayName = name;
        if (preset.isDefault) {
            displayName += " (default)";
        }
        
        p_existingPresetCombo->addItem(displayName, name);
    }
    
    // Disable overwrite option if no presets exist
    p_overwriteExistingRadio->setEnabled(p_existingPresetCombo->count() > 0);
}

QString PresetSaveDialog::getPresetName() const
{
    if (p_createNewRadio->isChecked()) {
        return p_newNameEdit->text().trimmed();
    } else {
        return p_existingPresetCombo->currentData().toString();
    }
}

bool PresetSaveDialog::isOverwriteMode() const
{
    return p_overwriteExistingRadio->isChecked();
}

void PresetSaveDialog::onModeChanged()
{
    bool createNewMode = p_createNewRadio->isChecked();
    
    // Enable/disable appropriate controls
    p_newNameEdit->setEnabled(createNewMode);
    p_existingPresetCombo->setEnabled(!createNewMode);
    
    // Update OK button state
    updateOkButtonState();
    
    // Focus the active input
    if (createNewMode) {
        p_newNameEdit->setFocus();
        p_newNameEdit->selectAll();
    } else {
        p_existingPresetCombo->setFocus();
    }
}

void PresetSaveDialog::onExistingPresetChanged()
{
    updateOkButtonState();
}

void PresetSaveDialog::onNewNameChanged()
{
    updateOkButtonState();
}

void PresetSaveDialog::updateOkButtonState()
{
    bool enabled = false;
    
    if (p_createNewRadio->isChecked()) {
        // For new preset, need non-empty name
        enabled = !p_newNameEdit->text().trimmed().isEmpty();
    } else if (p_overwriteExistingRadio->isChecked()) {
        // For overwrite, need valid selection
        enabled = p_existingPresetCombo->currentIndex() >= 0;
    }
    
    p_okButton->setEnabled(enabled);
}
#include "runtimehardwareconfigdialog.h"
#include "runtimehardwareconfigdialog_ui.h"
#include <QTreeWidgetItem>
#include <QListWidgetItem>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QComboBox>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QGroupBox>
#include <QHash>
#include <QButtonGroup>
#include <QRadioButton>
#include <QCheckBox>
#include <QMessageBox>
#include <QDialogButtonBox>
#include <data/bcglobals.h>
#include <hardware/core/hardwareregistry.h>
#include <hardware/core/hardwareprofilemanager.h>
#include <gui/style/themecolors.h>

RuntimeHardwareConfigDialog::RuntimeHardwareConfigDialog(QWidget *parent)
    : QDialog(parent),
      pu_ui(new Ui::RuntimeHardwareConfigDialog)
{
    pu_ui->setupUi(this);
    
    // Apply theme-aware styling to validation status label
    pu_ui->applyValidationStatusStyling(this);
    
    // Phase 2: Populate configuration overview with actual hardware data
    populateConfigurationOverview();
    
    // Phase 3: Populate hardware browser and connect selection handling
    populateHardwareBrowser();
    
    // Connect dialog buttons with custom logic for Phase 4.3 state management
    connect(pu_ui->buttonBox, &QDialogButtonBox::accepted, this, &RuntimeHardwareConfigDialog::onDialogAccepted);
    connect(pu_ui->buttonBox, &QDialogButtonBox::rejected, this, &RuntimeHardwareConfigDialog::onDialogRejected);
    
    // Connect hardware browser selection changes
    connect(pu_ui->hardwareBrowserList, &QListWidget::currentItemChanged,
            this, &RuntimeHardwareConfigDialog::onHardwareBrowserSelectionChanged);
    
    // Initialize both original and preview state from current runtime configuration
    d_originalRuntimeConfig = RuntimeHardwareConfig::constInstance().getCurrentHardware();
    d_previewRuntimeConfig = d_originalRuntimeConfig;
    
    // Initialize validation status
    validatePreviewConfiguration();
}

RuntimeHardwareConfigDialog::~RuntimeHardwareConfigDialog()
{
    delete pu_ui;
}

void RuntimeHardwareConfigDialog::populateConfigurationOverview()
{
    // Clear the tree
    pu_ui->configOverviewTree->clear();
    
    // Use preview configuration instead of current runtime configuration
    auto currentHw = d_previewRuntimeConfig;
    
    // Handle empty configuration
    if (currentHw.empty()) {
        auto* item = new QTreeWidgetItem(pu_ui->configOverviewTree);
        item->setText(0, "No hardware configured");
        item->setFlags(item->flags() & ~Qt::ItemIsSelectable);
        return;
    }
    
    // Group hardware by type to determine single vs multi-instance
    QHash<QString, QStringList> hardwareByType;
    for(auto it = currentHw.begin(); it != currentHw.end(); ++it) {
        auto [hwType, label] = BC::Key::parseKey(it->first);
        QString displayText = QString("%1 (%2)").arg(label, it->second);
        hardwareByType[hwType].append(displayText);
    }
    
    // Build tree structure based on single vs multi-instance
    for(auto typeIt = hardwareByType.begin(); typeIt != hardwareByType.end(); ++typeIt) {
        const QString& hwType = typeIt.key();
        const QStringList& instances = typeIt.value();
        
        if (instances.size() == 1) {
            // Single instance: "HardwareType: label (implementation)"
            auto* item = new QTreeWidgetItem(pu_ui->configOverviewTree);
            item->setText(0, QString("%1: %2").arg(hwType, instances.first()));
        } else {
            // Multi-instance: Parent "HardwareType" with children "label (implementation)"
            auto* parentItem = new QTreeWidgetItem(pu_ui->configOverviewTree);
            parentItem->setText(0, hwType);
            parentItem->setExpanded(true);
            
            for(const QString& instance : instances) {
                auto* childItem = new QTreeWidgetItem(parentItem);
                childItem->setText(0, instance);
            }
        }
    }
    
    // Ensure tree is fully expanded
    pu_ui->configOverviewTree->expandAll();
}

void RuntimeHardwareConfigDialog::refreshConfigurationOverview()
{
    populateConfigurationOverview();
}

void RuntimeHardwareConfigDialog::populateHardwareBrowser()
{
    // Clear the list
    pu_ui->hardwareBrowserList->clear();
    
    // Get all available hardware types from the registry
    auto hardwareTypes = HardwareRegistry::instance().getHardwareTypes();
    
    // Get current hardware configuration to count active instances (use preview state)
    auto currentHw = d_previewRuntimeConfig;
    
    // Count instances by hardware type
    QHash<QString, int> typeCounts;
    for(auto it = currentHw.begin(); it != currentHw.end(); ++it) {
        auto [hwType, label] = BC::Key::parseKey(it->first);
        typeCounts[hwType]++;
    }
    
    // Sort hardware types for consistent display
    hardwareTypes.sort();
    
    // Populate the list with hardware types and counts
    for(const QString& hwType : hardwareTypes) {
        int count = typeCounts.value(hwType, 0);
        QString displayText = QString("%1 (%2)").arg(hwType).arg(count);
        
        auto* item = new QListWidgetItem(displayText, pu_ui->hardwareBrowserList);
        
        // Store the hardware type for easy retrieval
        item->setData(Qt::UserRole, hwType);
        
        // Apply visual styling based on configuration status
        if (count > 0) {
            // Configured hardware: bold text
            QFont font = item->font();
            font.setBold(true);
            item->setFont(font);
        } else {
            // Unconfigured hardware: normal text (default)
        }
    }
}

void RuntimeHardwareConfigDialog::onHardwareBrowserSelectionChanged(QListWidgetItem* current, QListWidgetItem* previous)
{
    Q_UNUSED(previous)
    
    if (current == nullptr) {
        d_currentHardwareType.clear();
        updateSelectionDisplay(QString());
        return;
    }
    
    // Extract hardware type from the selected item and store it
    QString selectedHardwareType = current->data(Qt::UserRole).toString();
    d_currentHardwareType = selectedHardwareType;
    updateSelectionDisplay(selectedHardwareType);
}

void RuntimeHardwareConfigDialog::updateSelectionDisplay(const QString& selectedHardwareType)
{
    // Update the main configuration panel title 
    if (selectedHardwareType.isEmpty()) {
        pu_ui->configurationLabel->setText("Configuration");
    } else {
        pu_ui->configurationLabel->setText(QString("%1 Configuration").arg(selectedHardwareType));
    }
    
    // Use the new dynamic UI creation method
    updateRightPanelForHardwareType(selectedHardwareType);
}

void RuntimeHardwareConfigDialog::updateRightPanelForHardwareType(const QString& hardwareType)
{
    // Clear the existing content in the right panel
    if (pu_ui->configurationContentWidget->layout()) {
        QLayoutItem* item;
        while ((item = pu_ui->configurationContentWidget->layout()->takeAt(0)) != nullptr) {
            // Properly disconnect all signals from the widget before deletion to prevent crashes
            if (item->widget()) {
                item->widget()->disconnect();
            }
            delete item->widget();
            delete item;
        }
        delete pu_ui->configurationContentWidget->layout();
    }
    
    // Create a new layout for the right panel content
    auto* layout = new QVBoxLayout(pu_ui->configurationContentWidget);
    
    // Handle case where no hardware type is selected
    if (hardwareType.isEmpty()) {
        auto* noSelectionLabel = new QLabel("No hardware type selected", pu_ui->configurationContentWidget);
        noSelectionLabel->setAlignment(Qt::AlignCenter);
        noSelectionLabel->setStyleSheet(QString("QLabel { color: %1; font-style: italic; }")
            .arg(ThemeColors::getCSSColor(ThemeColors::SubtleText, this)));
        layout->addWidget(noSelectionLabel);
        layout->addStretch();
        pu_ui->configurationContentWidget->setLayout(layout);
        return;
    }
    
    // Determine if this is single-instance or multi-instance hardware type
    bool isMultiInstance = HardwareRegistry::isMultiInstanceType(hardwareType);
    
    // Create main group box for the hardware profile management
    auto* profileGroupBox = new QGroupBox(QString("%1 Profiles").arg(hardwareType), pu_ui->configurationContentWidget);
    auto* groupLayout = new QVBoxLayout(profileGroupBox);
    
    // Available Profiles label
    auto* availableProfilesLabel = new QLabel("Available Profiles:");
    availableProfilesLabel->setStyleSheet("QLabel { font-weight: bold; }");
    groupLayout->addWidget(availableProfilesLabel);
    
    // Create checkable list widget for profile selection
    auto* profilesList = new QListWidget();
    profilesList->setObjectName("profilesList");
    profilesList->setMaximumHeight(200);
    
    // Get all profiles for this hardware type from HardwareProfileManager
    auto& profileManager = HardwareProfileManager::instance();
    QStringList allProfiles = profileManager.getAllProfiles(hardwareType);
    
    // Create button group for radio button behavior (single-instance only)
    QButtonGroup* buttonGroup = nullptr;
    if (!isMultiInstance) {
        buttonGroup = new QButtonGroup(this);
        buttonGroup->setExclusive(true);
    }
    
    // Populate profiles list with appropriate selection widgets
    for (const QString& profileLabel : allProfiles) {
        QString implementation = profileManager.getImplementation(hardwareType, profileLabel);
        QString displayText = QString("%1 (%2)").arg(profileLabel, implementation);
        
        auto* listItem = new QListWidgetItem(profilesList);
        listItem->setText(displayText);
        listItem->setData(Qt::UserRole, profileLabel); // Store label for easy retrieval
        
        // Create appropriate selection widget based on instance type
        QWidget* selectionWidget = nullptr;
        if (isMultiInstance) {
            // Multi-instance: use checkboxes
            auto* checkbox = new QCheckBox(displayText);
            checkbox->setObjectName(QString("profile_%1_checkbox").arg(profileLabel));
            
            // Check if this profile is active in preview configuration
            QString profileKey = BC::Key::hwKey(hardwareType, profileLabel);
            bool isActive = d_previewRuntimeConfig.find(profileKey) != d_previewRuntimeConfig.end();
            checkbox->setChecked(isActive);
            
            // Connect to selection change handler
            connect(checkbox, &QCheckBox::toggled, this, [this, hardwareType]() {
                onProfileSelectionChanged(hardwareType);
            });
            
            selectionWidget = checkbox;
            
        } else {
            // Single-instance: use radio buttons
            auto* radioButton = new QRadioButton(displayText);
            radioButton->setObjectName(QString("profile_%1_radio").arg(profileLabel));
            
            // Check if this profile is active in preview configuration
            QString profileKey = BC::Key::hwKey(hardwareType, profileLabel);
            bool isActive = d_previewRuntimeConfig.find(profileKey) != d_previewRuntimeConfig.end();
            radioButton->setChecked(isActive);
            
            // Add to button group
            buttonGroup->addButton(radioButton);
            
            // Connect to selection change handler
            connect(radioButton, &QRadioButton::toggled, this, [this, hardwareType]() {
                onProfileSelectionChanged(hardwareType);
            });
            
            selectionWidget = radioButton;
        }
        
        // Set the widget for this list item
        profilesList->setItemWidget(listItem, selectionWidget);
    }
    
    // Handle case where no profiles exist
    if (allProfiles.isEmpty()) {
        auto* noProfilesItem = new QListWidgetItem("No profiles available", profilesList);
        noProfilesItem->setFlags(noProfilesItem->flags() & ~Qt::ItemIsSelectable);
        noProfilesItem->setForeground(ThemeColors::getThemeAwareColor(ThemeColors::SubtleText, this));
        
        auto* noProfilesLabel = new QLabel("No profiles available");
        noProfilesLabel->setAlignment(Qt::AlignCenter);
        noProfilesLabel->setStyleSheet(QString("QLabel { color: %1; font-style: italic; }")
            .arg(ThemeColors::getCSSColor(ThemeColors::SubtleText, this)));
        profilesList->setItemWidget(noProfilesItem, noProfilesLabel);
    }
    
    groupLayout->addWidget(profilesList);
    
    // Add/Remove buttons (horizontal layout)
    auto* buttonLayout = new QHBoxLayout();
    
    auto* addProfileButton = new QPushButton("Add Profile");
    addProfileButton->setObjectName("addProfileButton");
    connect(addProfileButton, &QPushButton::clicked, this, [this, hardwareType]() {
        onAddProfile(hardwareType);
    });
    
    auto* removeProfileButton = new QPushButton("Remove Profile");
    removeProfileButton->setObjectName("removeProfileButton");
    connect(removeProfileButton, &QPushButton::clicked, this, [this, hardwareType]() {
        onRemoveProfile(hardwareType);
    });
    
    buttonLayout->addWidget(addProfileButton);
    buttonLayout->addWidget(removeProfileButton);
    buttonLayout->addStretch();
    
    groupLayout->addLayout(buttonLayout);
    
    // Add the group box to the main layout
    layout->addWidget(profileGroupBox);
    layout->addStretch(); // Add stretch to push content to top
    
    pu_ui->configurationContentWidget->setLayout(layout);
}

void RuntimeHardwareConfigDialog::onProfileSelectionChanged(const QString& hardwareType)
{
    if (hardwareType.isEmpty()) {
        return;
    }
    
    // Find the profiles list widget
    auto* profilesList = pu_ui->configurationContentWidget->findChild<QListWidget*>("profilesList");
    if (!profilesList) {
        return;
    }
    
    // Clear existing entries for this hardware type in preview config
    auto it = d_previewRuntimeConfig.begin();
    while (it != d_previewRuntimeConfig.end()) {
        auto [hwType, label] = BC::Key::parseKey(it->first);
        if (hwType == hardwareType) {
            it = d_previewRuntimeConfig.erase(it);
        } else {
            ++it;
        }
    }
    
    // Add selected profiles to preview configuration
    auto& profileManager = HardwareProfileManager::instance();
    bool isMultiInstance = HardwareRegistry::isMultiInstanceType(hardwareType);
    
    for (int i = 0; i < profilesList->count(); ++i) {
        auto* listItem = profilesList->item(i);
        QString profileLabel = listItem->data(Qt::UserRole).toString();
        
        if (profileLabel.isEmpty()) {
            continue; // Skip items without profile data (like "No profiles available")
        }
        
        // Check if this profile is selected
        bool isSelected = false;
        if (isMultiInstance) {
            auto* checkbox = qobject_cast<QCheckBox*>(profilesList->itemWidget(listItem));
            if (checkbox) {
                isSelected = checkbox->isChecked();
            }
        } else {
            auto* radioButton = qobject_cast<QRadioButton*>(profilesList->itemWidget(listItem));
            if (radioButton) {
                isSelected = radioButton->isChecked();
            }
        }
        
        // Add to preview configuration if selected
        if (isSelected) {
            QString implementation = profileManager.getImplementation(hardwareType, profileLabel);
            QString profileKey = BC::Key::hwKey(hardwareType, profileLabel);
            d_previewRuntimeConfig[profileKey] = implementation;
        }
    }
    
    // Update preview display
    updatePreviewConfiguration();
}

void RuntimeHardwareConfigDialog::updatePreviewConfiguration()
{
    // Refresh the configuration overview (left panel) to show preview state
    populateConfigurationOverview();
    
    // Refresh the hardware browser to show updated instance counts
    populateHardwareBrowser();
    
    // Validate the configuration and update status
    validatePreviewConfiguration();
}

void RuntimeHardwareConfigDialog::onAddProfile(const QString& hardwareType)
{
    if (hardwareType.isEmpty()) {
        return;
    }
    
    // Get available implementations for this hardware type
    QStringList implementations = HardwareRegistry::instance().getImplementations(hardwareType);
    if (implementations.isEmpty()) {
        QMessageBox::warning(this, "Add Profile", "No implementations available for " + hardwareType);
        return;
    }
    
    // Create a simple dialog for adding a profile
    QDialog addDialog(this);
    addDialog.setWindowTitle("Add " + hardwareType + " Profile");
    addDialog.setModal(true);
    addDialog.resize(400, 200);
    
    auto* layout = new QVBoxLayout(&addDialog);
    
    // Implementation selection
    auto* formLayout = new QFormLayout();
    
    auto* implementationCombo = new QComboBox();
    implementationCombo->addItems(implementations);
    formLayout->addRow("Implementation:", implementationCombo);
    
    // Label input
    auto* labelEdit = new QLineEdit();
    labelEdit->setPlaceholderText("Enter unique label for this profile");
    
    // Generate a default label suggestion
    auto& profileManager = HardwareProfileManager::instance();
    QString defaultLabel = profileManager.generateDefaultLabel(hardwareType);
    labelEdit->setText(defaultLabel);
    
    formLayout->addRow("Label:", labelEdit);
    
    layout->addLayout(formLayout);
    
    // Validation label
    auto* validationLabel = new QLabel();
    validationLabel->setStyleSheet(QString("QLabel { color: %1; }")
        .arg(ThemeColors::getCSSColor(ThemeColors::StatusError, this)));
    validationLabel->hide();
    layout->addWidget(validationLabel);
    
    // Button box
    auto* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &addDialog);
    layout->addWidget(buttonBox);
    
    // Connect buttons
    connect(buttonBox, &QDialogButtonBox::accepted, &addDialog, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, &addDialog, &QDialog::reject);
    
    // Real-time label validation
    connect(labelEdit, &QLineEdit::textChanged, [&](const QString& text) {
        auto validationError = profileManager.validateLabel(text);
        if (validationError != HardwareProfileManager::Valid) {
            QString errorMsg;
            switch (validationError) {
                case HardwareProfileManager::Empty:
                    errorMsg = "Label cannot be empty";
                    break;
                case HardwareProfileManager::TooLong:
                    errorMsg = "Label too long (max 64 characters)";
                    break;
                case HardwareProfileManager::InvalidCharacters:
                    errorMsg = "Label contains invalid characters";
                    break;
                case HardwareProfileManager::StartsWithNumber:
                    errorMsg = "Label cannot start with a number";
                    break;
                case HardwareProfileManager::StartsWithUnderscore:
                    errorMsg = "Label cannot start with underscore";
                    break;
                case HardwareProfileManager::ContainsDots:
                    errorMsg = "Label cannot contain dots";
                    break;
                default:
                    errorMsg = "Invalid label";
                    break;
            }
            validationLabel->setText(errorMsg);
            validationLabel->show();
            buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
        } else if (!profileManager.isLabelAvailable(hardwareType, text)) {
            validationLabel->setText("Label already exists for this hardware type");
            validationLabel->show();
            buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
        } else {
            validationLabel->hide();
            buttonBox->button(QDialogButtonBox::Ok)->setEnabled(true);
        }
    });
    
    // Trigger initial validation
    labelEdit->textChanged(labelEdit->text());
    
    // Show dialog and handle result
    if (addDialog.exec() == QDialog::Accepted) {
        QString implementation = implementationCombo->currentText();
        QString label = labelEdit->text().trimmed();
        
        // Create the profile
        QString actualLabel = profileManager.createHardwareProfile(hardwareType, implementation, label);
        
        if (!actualLabel.isEmpty()) {
            // Success - refresh the right panel to show the new profile
            updateRightPanelForHardwareType(hardwareType);
            
            // For single-instance hardware, automatically activate the new profile in preview
            if (!HardwareRegistry::isMultiInstanceType(hardwareType)) {
                // Clear existing entries for this hardware type in preview
                auto it = d_previewRuntimeConfig.begin();
                while (it != d_previewRuntimeConfig.end()) {
                    auto [hwType, label] = BC::Key::parseKey(it->first);
                    if (hwType == hardwareType) {
                        it = d_previewRuntimeConfig.erase(it);
                    } else {
                        ++it;
                    }
                }
                
                // Add the new profile to preview configuration
                QString profileKey = BC::Key::hwKey(hardwareType, actualLabel);
                d_previewRuntimeConfig[profileKey] = implementation;
                
                // Update preview display
                updatePreviewConfiguration();
            }
        } else {
            QMessageBox::warning(this, "Add Profile", "Failed to create profile. Please try again.");
        }
    }
}

void RuntimeHardwareConfigDialog::onRemoveProfile(const QString& hardwareType)
{
    if (hardwareType.isEmpty()) {
        return;
    }
    
    // Find the profiles list widget
    auto* profilesList = pu_ui->configurationContentWidget->findChild<QListWidget*>("profilesList");
    if (!profilesList) {
        return;
    }
    
    // Find which profiles are selected for removal (based on QListWidget selection, not activation state)
    QStringList profilesToRemove;
    auto& profileManager = HardwareProfileManager::instance();
    
    // Get selected items from the list widget
    QList<QListWidgetItem*> selectedItems = profilesList->selectedItems();
    
    if (selectedItems.isEmpty()) {
        // No items selected in the list widget
        QMessageBox::information(this, "Remove Profile", "Please select at least one profile to remove from the list.");
        return;
    }
    
    // Extract profile labels from selected items
    for (QListWidgetItem* item : selectedItems) {
        QString profileLabel = item->data(Qt::UserRole).toString();
        
        if (!profileLabel.isEmpty()) {
            profilesToRemove.append(profileLabel);
        }
    }
    
    // Confirm deletion
    QString message;
    if (profilesToRemove.size() == 1) {
        message = QString("Are you sure you want to delete the profile '%1'?\n\nThis action cannot be undone.")
                     .arg(profilesToRemove.first());
    } else {
        message = QString("Are you sure you want to delete %1 profiles?\n\nProfiles to delete:\n• %2\n\nThis action cannot be undone.")
                     .arg(profilesToRemove.size())
                     .arg(profilesToRemove.join("\n• "));
    }
    
    int result = QMessageBox::warning(this, "Remove Profile", message,
                                      QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
    
    if (result == QMessageBox::Yes) {
        // Remove profiles
        bool success = true;
        for (const QString& label : profilesToRemove) {
            if (!profileManager.deleteHardwareProfile(hardwareType, label)) {
                success = false;
                qWarning() << "Failed to delete profile" << hardwareType << label;
            } else {
                // Phase 4.3: Remove from BOTH preview and original runtime configs
                // This handles the edge case where deleted profiles must be removed permanently
                QString profileKey = BC::Key::hwKey(hardwareType, label);
                d_previewRuntimeConfig.erase(profileKey);
                d_originalRuntimeConfig.erase(profileKey);
            }
        }
        
        if (success) {
            // Refresh the right panel to show updated profile list
            updateRightPanelForHardwareType(hardwareType);
            
            // Update preview configuration display
            updatePreviewConfiguration();
        } else {
            QMessageBox::warning(this, "Remove Profile", "Some profiles could not be deleted. Please try again.");
        }
    }
}

void RuntimeHardwareConfigDialog::onDialogAccepted()
{
    // Apply the preview configuration to the runtime configuration
    auto& runtimeConfig = RuntimeHardwareConfig::instance();
    
    if (runtimeConfig.applyConfiguration(d_previewRuntimeConfig)) {
        // Configuration applied successfully
        accept();
    } else {
        // Configuration failed to apply
        QMessageBox::warning(this, "Configuration Error", 
                           "Failed to apply hardware configuration. Please check your settings and try again.");
    }
}

void RuntimeHardwareConfigDialog::onDialogRejected()
{
    // Phase 4.3: Restore original runtime configuration
    // Note: Profile creation/deletion has already been persisted immediately,
    // so we only need to restore the runtime configuration state minus any deleted profiles
    
    auto& runtimeConfig = RuntimeHardwareConfig::instance();
    
    // Apply the original configuration (which may have fewer profiles due to deletions)
    // This handles the edge case where deleted profiles must not be restored
    runtimeConfig.applyConfiguration(d_originalRuntimeConfig);
    
    // Close dialog without applying preview changes
    reject();
}

void RuntimeHardwareConfigDialog::validatePreviewConfiguration()
{
    // Use the new static validation method to validate the preview configuration
    QStringList validationErrors = RuntimeHardwareConfig::validateHardwareConfiguration(d_previewRuntimeConfig);
    
    if (validationErrors.isEmpty()) {
        // Configuration is valid
        updateValidationStatus("Configuration is valid", "Success");
        // Enable Apply button
        pu_ui->buttonBox->button(QDialogButtonBox::Ok)->setEnabled(true);
    } else {
        // Configuration has errors
        QString errorMessage;
        if (validationErrors.size() == 1) {
            errorMessage = QString("Error: %1").arg(validationErrors.first());
        } else {
            errorMessage = QString("Error: %1 (+%2 more)").arg(validationErrors.first()).arg(validationErrors.size() - 1);
        }
        updateValidationStatus(errorMessage, "Error");
        // Disable Apply button
        pu_ui->buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);
    }
}

void RuntimeHardwareConfigDialog::updateValidationStatus(const QString& message, const QString& state)
{
    pu_ui->validationStatusLabel->setText(message);
    
    // Apply ThemeColors styling based on state
    QString styleSheet;
    if (state == "Success") {
        styleSheet = QString("QLabel { color: %1; }")
            .arg(ThemeColors::getCSSColor(ThemeColors::StatusSuccess, this));
    } else if (state == "Error") {
        styleSheet = QString("QLabel { color: %1; }")
            .arg(ThemeColors::getCSSColor(ThemeColors::StatusError, this));
    } else if (state == "Info") {
        styleSheet = QString("QLabel { color: %1; font-style: italic; }")
            .arg(ThemeColors::getCSSColor(ThemeColors::StatusInfo, this));
    } else {
        // Default styling
        styleSheet = QString("QLabel { color: %1; }")
            .arg(ThemeColors::getCSSColor(ThemeColors::StatusSuccess, this));
    }
    
    pu_ui->validationStatusLabel->setStyleSheet(styleSheet);
}
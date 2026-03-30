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
#include <QSet>
#include <QButtonGroup>
#include <QRadioButton>
#include <QCheckBox>
#include <QMessageBox>
#include <QDialogButtonBox>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QTextEdit>
#include <QMetaEnum>
#include <QSettings>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCoreApplication>
#include <data/bcglobals.h>
#include <data/settings/hardwarekeys.h>
#include <data/storage/applicationconfigmanager.h>
#include <hardware/core/communication/communicationprotocol.h>
#include <hardware/core/hardwareregistry.h>
#include <hardware/core/hardwareprofilemanager.h>
#include <hardware/library/vendorlibrary.h>
#include <hardware/library/spectrumlibrary.h>
#include <hardware/library/labjacklibrary.h>
#include <gui/style/themecolors.h>

RuntimeHardwareConfigDialog::RuntimeHardwareConfigDialog(QWidget *parent)
    : QDialog(parent),
      pu_ui(new Ui::RuntimeHardwareConfigDialog),
      p_currentLibrary(nullptr)
{
    pu_ui->setupUi(this);

    // Apply theme-aware styling to validation status label
    pu_ui->applyValidationStatusStyling(this);

    // Ensure system profiles exist before reading configuration
    HardwareProfileManager::instance().ensureSystemProfiles();

    // Initialize both original and preview state from current runtime configuration FIRST
    d_originalRuntimeConfig = RuntimeHardwareConfig::constInstance().getCurrentHardware();
    d_previewRuntimeConfig = d_originalRuntimeConfig;

    // Initialize threaded preview config from stored overrides
    for (auto& [hwKey, impl] : d_originalRuntimeConfig) {
        auto override = RuntimeHardwareConfig::constInstance().getThreaded(hwKey);
        if (override.has_value())
            d_previewThreadedConfig[hwKey] = *override;
    }

    // Auto-activate system profiles for required types that have no active entry
    {
        auto& profileManager = HardwareProfileManager::instance();
        HardwareRegistry& registry = HardwareRegistry::instance();
        for (const QString& hwType : registry.getHardwareTypes()) {
            if (!RuntimeHardwareConfig::isHardwareRequired(hwType)) {
                continue;
            }
            bool hasEntry = false;
            for (auto& [key, impl] : d_previewRuntimeConfig) {
                auto [type, label] = BC::Key::parseKey(key);
                if (type == hwType) {
                    hasEntry = true;
                    break;
                }
            }
            if (!hasEntry) {
                QString impl = profileManager.getImplementation(hwType, QStringLiteral("virtual"));
                if (!impl.isEmpty()) {
                    d_previewRuntimeConfig[BC::Key::hwKey(hwType, QStringLiteral("virtual"))] = impl;
                }
            }
        }
    }
    
    // Phase 2: Populate configuration overview with actual hardware data
    populateConfigurationOverview();
    
    // Phase 3: Populate hardware browser and connect selection handling
    populateHardwareBrowser();
    
    // Phase 3.5: Initialize Library Status tab
    initializeLibraryStatusTab();
    
    // Connect dialog buttons with custom logic for Phase 4.3 state management
    connect(pu_ui->buttonBox, &QDialogButtonBox::accepted, this, &RuntimeHardwareConfigDialog::onDialogAccepted);
    connect(pu_ui->buttonBox, &QDialogButtonBox::rejected, this, &RuntimeHardwareConfigDialog::onDialogRejected);
    
    // Connect hardware browser selection changes
    connect(pu_ui->hardwareBrowserList, &QListWidget::currentItemChanged,
            this, &RuntimeHardwareConfigDialog::onHardwareBrowserSelectionChanged);

    // Connect left panel overview tree to drive middle/right panel selection
    connect(pu_ui->configOverviewTree, &QTreeWidget::currentItemChanged,
            this, [this](QTreeWidgetItem* current, QTreeWidgetItem*) {
        if (!current)
            return;
        QString hwType = current->data(0, Qt::UserRole).toString();
        if (hwType.isEmpty())
            return;
        // Find and select the matching item in the hardware browser list
        for (int i = 0; i < pu_ui->hardwareBrowserList->count(); ++i) {
            auto* item = pu_ui->hardwareBrowserList->item(i);
            if (item->data(Qt::UserRole).toString() == hwType) {
                pu_ui->hardwareBrowserList->setCurrentItem(item);
                break;
            }
        }
    });
    
    // Initialize validation status
    validatePreviewConfiguration();
    
    // Initialize staging indicators for libraries
    updateAllLibraryStagingIndicators();
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
            item->setData(0, Qt::UserRole, hwType);
        } else {
            // Multi-instance: Parent "HardwareType" with children "label (implementation)"
            auto* parentItem = new QTreeWidgetItem(pu_ui->configOverviewTree);
            parentItem->setText(0, hwType);
            parentItem->setData(0, Qt::UserRole, hwType);
            parentItem->setExpanded(true);

            for(const QString& instance : instances) {
                auto* childItem = new QTreeWidgetItem(parentItem);
                childItem->setText(0, instance);
                childItem->setData(0, Qt::UserRole, hwType);
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
    // Preserve current selection to prevent right panel from closing
    QString currentSelectedHardwareType = d_currentHardwareType;
    
    // Block signals to prevent unwanted selection change events during repopulation
    pu_ui->hardwareBrowserList->blockSignals(true);
    
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
    QListWidgetItem* itemToSelect = nullptr;
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

        // Gray out LIF hardware types when LIF module is disabled
        if (RuntimeHardwareConfig::isLifHardwareType(hwType)
            && !ApplicationConfigManager::instance().isLifEnabled())
        {
            item->setForeground(ThemeColors::getThemeAwareColor(ThemeColors::SubtleText, this));
            QFont f = item->font();
            f.setItalic(true);
            item->setFont(f);
            item->setText(item->text() + " [LIF disabled]");
        }

        // Remember this item if it matches the previously selected hardware type
        if (hwType == currentSelectedHardwareType) {
            itemToSelect = item;
        }
    }
    
    // Restore selection if we had one previously
    if (itemToSelect != nullptr) {
        pu_ui->hardwareBrowserList->setCurrentItem(itemToSelect);
    }
    
    // Re-enable signals
    pu_ui->hardwareBrowserList->blockSignals(false);
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
            // Let Qt handle signal cleanup automatically when parent is destroyed
            if (item->widget()) {
                delete item->widget();
            }
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
    
    // Check if this is a LIF hardware type that's currently disabled
    if (RuntimeHardwareConfig::isLifHardwareType(hardwareType)
        && !ApplicationConfigManager::instance().isLifEnabled())
    {
        auto* disabledLabel = new QLabel(
            QString("The %1 hardware type is currently inactive because the LIF module "
                    "is disabled.\n\nTo use LIF hardware, enable the LIF Module in "
                    "Settings > Application Settings and restart the application.")
            .arg(hardwareType),
            pu_ui->configurationContentWidget);
        disabledLabel->setWordWrap(true);
        disabledLabel->setAlignment(Qt::AlignTop | Qt::AlignLeft);
        disabledLabel->setStyleSheet(QString("QLabel { color: %1; font-style: italic; padding: 12px; }")
            .arg(ThemeColors::getCSSColor(ThemeColors::SubtleText, this)));
        layout->addWidget(disabledLabel);
        layout->addStretch();
        pu_ui->configurationContentWidget->setLayout(layout);
        return;
    }

    // Determine if this is single-instance or multi-instance hardware type
    bool isMultiInstance = HardwareRegistry::isMultiInstanceType(hardwareType);
    bool isRequired = RuntimeHardwareConfig::isHardwareRequired(hardwareType);

    // For single-instance types, add an Enable checkbox at the top
    if (!isMultiInstance) {
        // Determine if this type is currently enabled (has any entry in preview config)
        bool isEnabled = false;
        for (auto& [key, impl] : d_previewRuntimeConfig) {
            auto [type, lbl] = BC::Key::parseKey(key);
            if (type == hardwareType) {
                isEnabled = true;
                break;
            }
        }

        auto* enableCheckbox = new QCheckBox(QString("Enable %1").arg(hardwareType), pu_ui->configurationContentWidget);
        enableCheckbox->setObjectName("enableHardwareCheckbox");
        enableCheckbox->setChecked(isEnabled);

        if (isRequired) {
            enableCheckbox->setEnabled(false);
            enableCheckbox->setChecked(true);
            enableCheckbox->setToolTip(QString("%1 is required for BlackChirp to operate and cannot be disabled.")
                                           .arg(hardwareType));
        }

        layout->addWidget(enableCheckbox);

        // Connect enable checkbox (use toggled since we handle programmatic + user changes the same)
        connect(enableCheckbox, &QCheckBox::toggled, this, [this, hardwareType](bool checked) {
            onEnableToggled(hardwareType, checked);
        });
    }

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

    // For single-instance, determine if hardware is currently enabled
    bool singleInstanceEnabled = true;
    if (!isMultiInstance) {
        singleInstanceEnabled = false;
        for (auto& [key, impl] : d_previewRuntimeConfig) {
            auto [type, lbl] = BC::Key::parseKey(key);
            if (type == hardwareType) {
                singleInstanceEnabled = true;
                break;
            }
        }
    }

    // Populate profiles list with appropriate selection widgets
    for (const QString& profileLabel : allProfiles) {
        QString implementation = profileManager.getImplementation(hardwareType, profileLabel);
        bool isSysProfile = HardwareProfileManager::isSystemProfile(hardwareType, profileLabel);
        QString displayText = QString("%1 (%2)").arg(profileLabel, implementation);
        if (isSysProfile) {
            displayText += " [system]";
        }

        auto* listItem = new QListWidgetItem(profilesList);
        listItem->setData(Qt::UserRole, profileLabel); // Store label for easy retrieval
        // Don't set text on listItem to avoid duplication with custom widget text

        // Apply visual distinction for system profiles
        if (isSysProfile) {
            listItem->setForeground(ThemeColors::getThemeAwareColor(ThemeColors::SubtleText, this));
            QFont f = listItem->font();
            f.setItalic(true);
            listItem->setFont(f);
        }

        // Use built-in QListWidgetItem check state for profile activation.
        // This renders a checkbox indicator as part of the item decoration;
        // clicking the indicator toggles the check, while clicking the text
        // area only selects the row (for Advanced settings).
        listItem->setText(displayText);
        listItem->setFlags(listItem->flags() | Qt::ItemIsUserCheckable);

        QString profileKey = BC::Key::hwKey(hardwareType, profileLabel);
        bool isActive = d_previewRuntimeConfig.find(profileKey) != d_previewRuntimeConfig.end();
        if (!isMultiInstance && !singleInstanceEnabled)
            isActive = false;
        listItem->setCheckState(isActive ? Qt::Checked : Qt::Unchecked);
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

    // For single-instance, connect the enable checkbox to gray out profile items
    if (!isMultiInstance) {
        auto* enableCheckbox = pu_ui->configurationContentWidget->findChild<QCheckBox*>("enableHardwareCheckbox");
        if (enableCheckbox) {
            connect(enableCheckbox, &QCheckBox::toggled, this, [profilesList](bool checked) {
                for (int i = 0; i < profilesList->count(); ++i) {
                    auto* item = profilesList->item(i);
                    if (item->data(Qt::UserRole).toString().isEmpty())
                        continue;
                    auto flags = item->flags();
                    if (checked)
                        flags |= Qt::ItemIsUserCheckable;
                    else
                        flags &= ~Qt::ItemIsUserCheckable;
                    item->setFlags(flags);
                }
            });
        }
    }

    // Handle check state changes on list items.
    // For multi-instance: multiple items can be checked.
    // For single-instance: enforce mutual exclusivity (like radio buttons).
    connect(profilesList, &QListWidget::itemChanged, this,
            [this, hardwareType, profilesList, isMultiInstance](QListWidgetItem* changedItem) {
        if (isMultiInstance) {
            onProfileCheckboxClicked(hardwareType);
        } else {
            // Enforce radio-button behavior: uncheck all others when one is checked
            if (changedItem->checkState() == Qt::Checked) {
                QSignalBlocker blocker(profilesList);
                for (int i = 0; i < profilesList->count(); ++i) {
                    auto* item = profilesList->item(i);
                    if (item != changedItem && item->checkState() == Qt::Checked)
                        item->setCheckState(Qt::Unchecked);
                }
            }
            onProfileRadioClicked(hardwareType);
        }
    });

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

    // Initially disable Remove button (no selection)
    removeProfileButton->setEnabled(false);

    // Connect to list selection changes to enable/disable Remove button
    connect(profilesList, &QListWidget::itemSelectionChanged, this, [removeProfileButton, profilesList]() {
        removeProfileButton->setEnabled(!profilesList->selectedItems().isEmpty());
    });

    buttonLayout->addWidget(addProfileButton);
    buttonLayout->addWidget(removeProfileButton);
    buttonLayout->addStretch();

    groupLayout->addLayout(buttonLayout);

    // Add the group box to the main layout
    layout->addWidget(profileGroupBox);

    // Add collapsible Advanced section
    auto* advancedButton = new QPushButton(QString::fromUtf8("\u25b6 Advanced"), pu_ui->configurationContentWidget);
    advancedButton->setObjectName("advancedToggleButton");
    advancedButton->setFlat(true);
    advancedButton->setStyleSheet("QPushButton { text-align: left; font-weight: bold; }");

    auto* advancedContainer = new QWidget(pu_ui->configurationContentWidget);
    advancedContainer->setObjectName("advancedContainer");
    advancedContainer->setVisible(false);

    auto* advancedLayout = new QVBoxLayout(advancedContainer);
    advancedLayout->setContentsMargins(4, 2, 4, 2);

    // Determine the profile whose Advanced settings should be displayed.
    // Uses the highlighted (selected) list item so the user can click any
    // profile row to view/edit its Advanced settings without toggling its
    // checked state.
    auto getActiveHwKey = [hardwareType, profilesList]() -> QString {
        auto selected = profilesList->selectedItems();
        if (!selected.isEmpty()) {
            QString profileLabel = selected.first()->data(Qt::UserRole).toString();
            if (!profileLabel.isEmpty())
                return BC::Key::hwKey(hardwareType, profileLabel);
        }
        return QString();
    };

    bool typeDefault = getTypeDefaultThreaded(hardwareType);
    QString activeHwKey = getActiveHwKey();
    bool threadedChecked = typeDefault;
    if (!activeHwKey.isEmpty()) {
        auto it = d_previewThreadedConfig.find(activeHwKey);
        if (it != d_previewThreadedConfig.end())
            threadedChecked = it->second;
    }

    QString checkboxLabel = tr("Run in own thread");
    if (typeDefault)
        checkboxLabel += tr(" (recommended)");
    auto* threadedCheckbox = new QCheckBox(checkboxLabel, advancedContainer);
    threadedCheckbox->setObjectName("threadedCheckbox");
    threadedCheckbox->setChecked(threadedChecked);
    threadedCheckbox->setEnabled(!activeHwKey.isEmpty());

    advancedLayout->addWidget(threadedCheckbox);

    // Python script path field — shown when the selected profile uses a Python implementation
    QLineEdit* pythonScriptEdit = nullptr;
#ifdef BC_PYTHON_HARDWARE
    {
        auto* pythonScriptWidget = new QWidget(advancedContainer);
        pythonScriptWidget->setObjectName("pythonScriptWidget");
        auto* scriptLayout = new QHBoxLayout(pythonScriptWidget);
        scriptLayout->setContentsMargins(0, 0, 0, 0);

        auto* scriptLabel = new QLabel(tr("Python Script:"), pythonScriptWidget);
        pythonScriptEdit = new QLineEdit(pythonScriptWidget);
        pythonScriptEdit->setObjectName("pythonScriptEdit");
        pythonScriptEdit->setPlaceholderText(tr("Path to Python hardware script..."));

        auto* browseButton = new QPushButton(tr("Browse..."), pythonScriptWidget);
        connect(browseButton, &QPushButton::clicked, this, [this, pythonScriptEdit]() {
            QString path = QFileDialog::getOpenFileName(this, tr("Select Python Script"),
                                                         QString(), tr("Python Files (*.py)"));
            if (!path.isEmpty())
                pythonScriptEdit->setText(path);
        });

        scriptLayout->addWidget(scriptLabel);
        scriptLayout->addWidget(pythonScriptEdit, 1);
        scriptLayout->addWidget(browseButton);

        // Helper: check whether the selected profile is a Python implementation
        auto isSelectedPython = [hardwareType, profilesList]() -> bool {
            auto selected = profilesList->selectedItems();
            if (selected.isEmpty()) return false;
            QString label = selected.first()->data(Qt::UserRole).toString();
            if (label.isEmpty()) return false;
            QString impl = HardwareProfileManager::instance().getImplementation(hardwareType, label);
            return impl.contains(QStringLiteral("Python"));
        };

        // Initialize visibility and value from the current selection
        bool isPython = isSelectedPython();
        pythonScriptWidget->setVisible(isPython);
        if (isPython && !activeHwKey.isEmpty()) {
            auto it = d_previewPythonScriptConfig.find(activeHwKey);
            if (it != d_previewPythonScriptConfig.end()) {
                pythonScriptEdit->setText(it->second);
            } else {
                auto [type, label] = BC::Key::parseKey(activeHwKey);
                pythonScriptEdit->setText(
                    HardwareProfileManager::instance().getPythonScriptPath(type, label));
            }
        }

        // Wire text changes into preview config
        connect(pythonScriptEdit, &QLineEdit::textChanged, this,
                [this, getActiveHwKey](const QString& text) {
            QString hwKey = getActiveHwKey();
            if (hwKey.isEmpty()) return;
            d_previewPythonScriptConfig[hwKey] = text;
        });

        advancedLayout->addWidget(pythonScriptWidget);
    }
#endif

    advancedContainer->setLayout(advancedLayout);

    // Toggle expand/collapse
    connect(advancedButton, &QPushButton::clicked, this, [advancedButton, advancedContainer](bool) {
        bool visible = !advancedContainer->isVisible();
        advancedContainer->setVisible(visible);
        advancedButton->setText(visible ? QString::fromUtf8("\u25bc Advanced")
                                        : QString::fromUtf8("\u25b6 Advanced"));
    });

    // Update Advanced section when profile selection changes in the list
    // Capture the parent widget so we can toggle visibility of the Python script row
    QWidget* pythonScriptWidget = advancedContainer->findChild<QWidget*>("pythonScriptWidget");
    connect(profilesList, &QListWidget::itemSelectionChanged, this,
            [this, hardwareType, profilesList, threadedCheckbox, typeDefault, getActiveHwKey,
             pythonScriptEdit, pythonScriptWidget]() {
        QString hwKey = getActiveHwKey();
        if (hwKey.isEmpty()) {
            threadedCheckbox->setEnabled(false);
            threadedCheckbox->setChecked(typeDefault);
            if (pythonScriptWidget)
                pythonScriptWidget->setVisible(false);
            return;
        }
        threadedCheckbox->setEnabled(true);
        auto it = d_previewThreadedConfig.find(hwKey);
        bool checked = (it != d_previewThreadedConfig.end()) ? it->second : typeDefault;
        // Block signals to avoid triggering the toggled connection below
        QSignalBlocker blocker(threadedCheckbox);
        threadedCheckbox->setChecked(checked);

        if (pythonScriptEdit) {
            // Show/hide based on whether selected profile is a Python implementation
            auto [type, label] = BC::Key::parseKey(hwKey);
            QString impl = HardwareProfileManager::instance().getImplementation(type, label);
            bool isPython = impl.contains(QStringLiteral("Python"));
            if (pythonScriptWidget)
                pythonScriptWidget->setVisible(isPython);

            if (isPython) {
                QSignalBlocker scriptBlocker(pythonScriptEdit);
                auto sit = d_previewPythonScriptConfig.find(hwKey);
                if (sit != d_previewPythonScriptConfig.end()) {
                    pythonScriptEdit->setText(sit->second);
                } else {
                    pythonScriptEdit->setText(
                        HardwareProfileManager::instance().getPythonScriptPath(type, label));
                }
            }
        }
    });

    // Wire checkbox changes into d_previewThreadedConfig
    connect(threadedCheckbox, &QCheckBox::toggled, this,
            [this, hardwareType, profilesList, getActiveHwKey](bool checked) {
        QString hwKey = getActiveHwKey();
        if (hwKey.isEmpty()) return;
        d_previewThreadedConfig[hwKey] = checked;
    });

    layout->addWidget(advancedButton);
    layout->addWidget(advancedContainer);
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

    for (int i = 0; i < profilesList->count(); ++i) {
        auto* listItem = profilesList->item(i);
        QString profileLabel = listItem->data(Qt::UserRole).toString();
        
        if (profileLabel.isEmpty()) {
            continue; // Skip items without profile data (like "No profiles available")
        }
        
        // Check if this profile is selected (checked)
        bool isSelected = (listItem->checkState() == Qt::Checked);
        
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

void RuntimeHardwareConfigDialog::onEnableToggled(const QString& hardwareType, bool enabled)
{
    if (hardwareType.isEmpty()) {
        return;
    }

    if (enabled) {
        // Find the currently selected radio button's profile label
        auto* profilesList = pu_ui->configurationContentWidget->findChild<QListWidget*>("profilesList");
        if (profilesList) {
            auto& profileManager = HardwareProfileManager::instance();
            for (int i = 0; i < profilesList->count(); ++i) {
                auto* listItem = profilesList->item(i);
                QString profileLabel = listItem->data(Qt::UserRole).toString();
                if (profileLabel.isEmpty()) {
                    continue;
                }
                if (listItem->checkState() == Qt::Checked) {
                    QString implementation = profileManager.getImplementation(hardwareType, profileLabel);
                    QString profileKey = BC::Key::hwKey(hardwareType, profileLabel);
                    d_previewRuntimeConfig[profileKey] = implementation;
                    break;
                }
            }

            // If no radio is checked, fall back to first profile (or "virtual" system profile)
            bool hasEntry = false;
            for (auto& [key, impl] : d_previewRuntimeConfig) {
                auto [type, lbl] = BC::Key::parseKey(key);
                if (type == hardwareType) {
                    hasEntry = true;
                    break;
                }
            }
            if (!hasEntry) {
                // Check for "virtual" system profile first
                QString impl = profileManager.getImplementation(hardwareType, QStringLiteral("virtual"));
                if (!impl.isEmpty()) {
                    d_previewRuntimeConfig[BC::Key::hwKey(hardwareType, QStringLiteral("virtual"))] = impl;
                } else if (!profileManager.getAllProfiles(hardwareType).isEmpty()) {
                    // Fall back to first available profile
                    QString firstLabel = profileManager.getAllProfiles(hardwareType).first();
                    impl = profileManager.getImplementation(hardwareType, firstLabel);
                    if (!impl.isEmpty()) {
                        d_previewRuntimeConfig[BC::Key::hwKey(hardwareType, firstLabel)] = impl;
                    }
                }
            }
        }
    } else {
        // Remove all entries for this hardware type from preview config
        auto it = d_previewRuntimeConfig.begin();
        while (it != d_previewRuntimeConfig.end()) {
            auto [hwType, lbl] = BC::Key::parseKey(it->first);
            if (hwType == hardwareType) {
                it = d_previewRuntimeConfig.erase(it);
            } else {
                ++it;
            }
        }
    }

    updatePreviewConfiguration();
}

void RuntimeHardwareConfigDialog::onProfileRadioClicked(const QString& hardwareType)
{
    if (hardwareType.isEmpty()) {
        return;
    }

    // Check if the enable checkbox is checked (only update if enabled)
    auto* enableCheckbox = pu_ui->configurationContentWidget->findChild<QCheckBox*>("enableHardwareCheckbox");
    if (enableCheckbox && !enableCheckbox->isChecked()) {
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
        auto [hwType, lbl] = BC::Key::parseKey(it->first);
        if (hwType == hardwareType) {
            it = d_previewRuntimeConfig.erase(it);
        } else {
            ++it;
        }
    }

    // Add the checked profile
    auto& profileManager = HardwareProfileManager::instance();
    for (int i = 0; i < profilesList->count(); ++i) {
        auto* listItem = profilesList->item(i);
        QString profileLabel = listItem->data(Qt::UserRole).toString();
        if (profileLabel.isEmpty())
            continue;
        if (listItem->checkState() == Qt::Checked) {
            QString implementation = profileManager.getImplementation(hardwareType, profileLabel);
            QString profileKey = BC::Key::hwKey(hardwareType, profileLabel);
            d_previewRuntimeConfig[profileKey] = implementation;
            break;
        }
    }

    updatePreviewConfiguration();
}

void RuntimeHardwareConfigDialog::onProfileCheckboxClicked(const QString& hardwareType)
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
        auto [hwType, lbl] = BC::Key::parseKey(it->first);
        if (hwType == hardwareType) {
            it = d_previewRuntimeConfig.erase(it);
        } else {
            ++it;
        }
    }

    // Add all checked profiles for this hardware type
    auto& profileManager = HardwareProfileManager::instance();
    for (int i = 0; i < profilesList->count(); ++i) {
        auto* listItem = profilesList->item(i);
        QString profileLabel = listItem->data(Qt::UserRole).toString();
        if (profileLabel.isEmpty()) {
            continue;
        }
        if (listItem->checkState() == Qt::Checked) {
            QString implementation = profileManager.getImplementation(hardwareType, profileLabel);
            QString profileKey = BC::Key::hwKey(hardwareType, profileLabel);
            d_previewRuntimeConfig[profileKey] = implementation;
        }
    }

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
    
    // Sort implementations alphabetically for better user experience
    implementations.sort();
    
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

    // Protocol selection (shown only when multiple protocols are supported)
    auto* protocolLabel = new QLabel("Protocol:");
    auto* protocolCombo = new QComboBox();
    formLayout->addRow(protocolLabel, protocolCombo);

    auto updateProtocolCombo = [&](const QString& impl) {
        auto protocols = HardwareRegistry::instance().getSupportedProtocols(hardwareType, impl);
        protocolCombo->clear();
        auto commEnum = QMetaEnum::fromType<CommunicationProtocol::CommType>();
        for (auto p : protocols)
            protocolCombo->addItem(QString(commEnum.valueToKey(static_cast<int>(p))),
                                   static_cast<int>(p));
        bool multiProtocol = protocols.size() > 1;
        protocolLabel->setVisible(multiProtocol);
        protocolCombo->setVisible(multiProtocol);
    };
    updateProtocolCombo(implementationCombo->currentText());
    connect(implementationCombo, &QComboBox::currentTextChanged, updateProtocolCombo);

    // Label input
    auto* labelEdit = new QLineEdit();
    labelEdit->setPlaceholderText("Enter unique label for this profile");
    
    // Generate a default label suggestion
    auto& profileManager = HardwareProfileManager::instance();
    QString defaultLabel = profileManager.generateDefaultLabel(hardwareType);
    labelEdit->setText(defaultLabel);
    
    formLayout->addRow("Label:", labelEdit);

    layout->addLayout(formLayout);

    // Configuration parameters (shown when implementation has HwConfigParams)
    auto* configParamsGroup = new QGroupBox("Configuration Parameters");
    auto* configParamsLayout = new QFormLayout(configParamsGroup);
    configParamsGroup->hide();

    // Map from param key -> widget, rebuilt when implementation changes
    QHash<QString, QWidget*> paramWidgets;

    auto updateConfigParams = [&](const QString& impl) {
        // Clear existing widgets
        while (configParamsLayout->rowCount() > 0)
            configParamsLayout->removeRow(0);
        paramWidgets.clear();

        auto params = HardwareRegistry::instance().getConfigParams(hardwareType, impl);
        configParamsGroup->setVisible(!params.isEmpty());

        for (const auto& param : params) {
            int typeId = param.defaultValue.userType();
            QWidget* widget = nullptr;

            if (typeId == QMetaType::Int) {
                auto* sb = new QSpinBox();
                sb->setValue(param.defaultValue.toInt());
                if (param.minimum.isValid()) sb->setMinimum(param.minimum.toInt());
                if (param.maximum.isValid()) sb->setMaximum(param.maximum.toInt());
                widget = sb;
            } else if (typeId == QMetaType::UInt) {
                auto* sb = new QSpinBox();
                sb->setValue(static_cast<int>(param.defaultValue.toUInt()));
                if (param.minimum.isValid()) sb->setMinimum(static_cast<int>(param.minimum.toUInt()));
                if (param.maximum.isValid()) sb->setMaximum(static_cast<int>(param.maximum.toUInt()));
                widget = sb;
            } else if (typeId == QMetaType::Double) {
                auto* dsb = new QDoubleSpinBox();
                dsb->setValue(param.defaultValue.toDouble());
                if (param.minimum.isValid()) dsb->setMinimum(param.minimum.toDouble());
                if (param.maximum.isValid()) dsb->setMaximum(param.maximum.toDouble());
                widget = dsb;
            } else if (typeId == QMetaType::Bool) {
                auto* cb = new QCheckBox();
                cb->setChecked(param.defaultValue.toBool());
                widget = cb;
            } else {
                auto* le = new QLineEdit();
                le->setText(param.defaultValue.toString());
                widget = le;
            }

            if (widget) {
                paramWidgets[param.key] = widget;
                configParamsLayout->addRow(param.label + ":", widget);
            }
        }
    };
    updateConfigParams(implementationCombo->currentText());
    connect(implementationCombo, &QComboBox::currentTextChanged, updateConfigParams);

    layout->addWidget(configParamsGroup);

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

        // Write selected protocol and config params to settings before hardware
        // object is created, so the constructor finds the correct values.
        // These must be written directly under the hwKey group (not under an
        // implementation subgroup), because HardwareObject's SettingsStorage
        // root is hwKey and it reads commType/params from that level.
        {
            auto selectedProtocol = static_cast<CommunicationProtocol::CommType>(
                protocolCombo->currentData().toInt());
            QString settingsKey = BC::Key::hwKey(hardwareType, label);
            QSettings s(QCoreApplication::organizationName(), QCoreApplication::applicationName());
            s.beginGroup(settingsKey);
            s.setValue(BC::Key::HW::commType, static_cast<int>(selectedProtocol));

            // Write config params (e.g., numChannels, tunable) so the base class
            // constructor can read them via SettingsStorage/getOrSetDefault.
            for (auto it = paramWidgets.cbegin(); it != paramWidgets.cend(); ++it) {
                QVariant val;
                if (auto* sb = qobject_cast<QSpinBox*>(it.value()))
                    val = sb->value();
                else if (auto* dsb = qobject_cast<QDoubleSpinBox*>(it.value()))
                    val = dsb->value();
                else if (auto* cb = qobject_cast<QCheckBox*>(it.value()))
                    val = cb->isChecked();
                else if (auto* le = qobject_cast<QLineEdit*>(it.value()))
                    val = le->text();

                if (val.isValid())
                    s.setValue(it.key(), val);
            }

            s.endGroup();
            s.sync();
        }

        // Create the profile
        QString actualLabel = profileManager.createHardwareProfile(hardwareType, implementation, label);
        
        if (!actualLabel.isEmpty()) {
            // Automatically activate the new profile if it's the first profile of this hardware type
            // Check if any profile already exists for this hardware type in preview configuration
            bool hasExistingProfile = false;
            for (auto it = d_previewRuntimeConfig.begin(); it != d_previewRuntimeConfig.end(); ++it) {
                auto [hwType, label] = BC::Key::parseKey(it->first);
                if (hwType == hardwareType) {
                    hasExistingProfile = true;
                    break;
                }
            }
            
            // Only add to preview if this is the first profile for this hardware type
            if (!hasExistingProfile) {
                QString profileKey = BC::Key::hwKey(hardwareType, actualLabel);
                d_previewRuntimeConfig[profileKey] = implementation;
            }
            
            // Offer to copy template script for Python hardware implementations
#ifdef BC_PYTHON_HARDWARE
            if (implementation.startsWith(QStringLiteral("Python"))) {
                // Derive template filename: "PythonAwg" -> "python_awg_template.py"
                QString typePart = implementation.mid(6); // Remove "Python" prefix
                QString templateFilename = QStringLiteral("python_%1_template.py").arg(typePart.toLower());

                // Search for template using same paths as findHostScript()
                QString appDir = QCoreApplication::applicationDirPath();
                QString templatePath;
                QStringList searchPaths = {
                    appDir + "/" + templateFilename,
                    appDir + "/../share/blackchirp/" + templateFilename
                };
                for (const auto& path : searchPaths) {
                    if (QFile::exists(path)) {
                        templatePath = path;
                        break;
                    }
                }

                if (!templatePath.isEmpty()) {
                    auto result = QMessageBox::question(
                        this, tr("Python Template Script"),
                        tr("Would you like to create a copy of the template script to customize?"),
                        QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);

                    if (result == QMessageBox::Yes) {
                        // Use the application's configured save path as the initial directory
                        SettingsStorage ss;
                        QString initialDir = ss.get(BC::Key::savePath, QString(""));
                        QString suggestedName = QStringLiteral("my_%1.py").arg(typePart.toLower());
                        if (!initialDir.isEmpty())
                            suggestedName = QDir(initialDir).filePath(suggestedName);
                        QString savePath = QFileDialog::getSaveFileName(
                            this, tr("Save Python Script"),
                            suggestedName,
                            tr("Python Scripts (*.py)"));

                        if (!savePath.isEmpty()) {
                            if (QFile::exists(savePath))
                                QFile::remove(savePath);

                            if (QFile::copy(templatePath, savePath)) {
                                // Make the copy writable (QFile::copy preserves source permissions)
                                QFile::setPermissions(savePath,
                                    QFile::permissions(savePath) | QFileDevice::WriteOwner);

                                QString profileKey = BC::Key::hwKey(hardwareType, actualLabel);
                                d_previewPythonScriptConfig[profileKey] = savePath;
                            } else {
                                QMessageBox::warning(this, tr("Python Template Script"),
                                    tr("Failed to copy template script to the selected location."));
                            }
                        }
                    }
                }
            }
#endif

            // Refresh the right panel to show the new profile with correct radio button state
            updateRightPanelForHardwareType(hardwareType);

            // Select the newly created profile in the list
            auto* profilesList = pu_ui->configurationContentWidget->findChild<QListWidget*>("profilesList");
            if (profilesList) {
                for (int i = 0; i < profilesList->count(); ++i) {
                    if (profilesList->item(i)->data(Qt::UserRole).toString() == actualLabel) {
                        profilesList->setCurrentRow(i);
                        break;
                    }
                }
            }

            // Update preview display
            updatePreviewConfiguration();
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
    
    // Extract profile labels from selected items, blocking system profiles
    for (QListWidgetItem* item : selectedItems) {
        QString profileLabel = item->data(Qt::UserRole).toString();

        if (profileLabel.isEmpty()) {
            continue;
        }

        // Block removal of system profiles
        if (HardwareProfileManager::isSystemProfile(hardwareType, profileLabel)) {
            QMessageBox::information(this, "Cannot Remove Profile",
                QString("The '%1' profile for %2 is a system profile and cannot be removed.\n\n"
                        "System profiles ensure BlackChirp can always start with working hardware.")
                    .arg(profileLabel, hardwareType));
            return;
        }

        profilesToRemove.append(profileLabel);
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

        // If the hardware type now has no active profile and a system profile exists,
        // auto-activate it so the user is never left in an unrecoverable grayed-out state.
        bool hasActiveProfile = false;
        for (auto& [key, impl] : d_previewRuntimeConfig) {
            auto [hwType, lbl] = BC::Key::parseKey(key);
            if (hwType == hardwareType) { hasActiveProfile = true; break; }
        }
        if (!hasActiveProfile) {
            QString sysImpl = profileManager.getImplementation(hardwareType, "virtual");
            if (!sysImpl.isEmpty()) {
                d_previewRuntimeConfig[BC::Key::hwKey(hardwareType, "virtual")] = sysImpl;
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
        // Apply threading overrides from preview config
        for (auto& [hwKey, threaded] : d_previewThreadedConfig)
            runtimeConfig.setThreaded(hwKey, threaded);

        // Apply Python script path overrides
        for (auto& [hwKey, scriptPath] : d_previewPythonScriptConfig) {
            auto [type, label] = BC::Key::parseKey(hwKey);
            if (!type.isEmpty() && !label.isEmpty())
                HardwareProfileManager::instance().setPythonScriptPath(type, label, scriptPath);
        }

        // Save all hardware profiles to persistent storage now that dialog is finished
        auto& profileManager = HardwareProfileManager::instance();
        profileManager.saveProfiles();
        
        // Phase 3.5.3: Library configuration changes are staged and will be applied
        // by HardwareManager::syncWithRuntimeConfig() which is called by MainWindow
        // after dialog close. This ensures library changes are applied before 
        // hardware synchronization begins.
        
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
    
    // Phase 3.5.2: Revert all library staged changes
    revertAllLibraryChanges();
    
    // Save all hardware profiles to persistent storage - profiles were created/deleted during session
    auto& profileManager = HardwareProfileManager::instance();
    profileManager.saveProfiles();
    
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

// Phase 3.5: Library Status Tab Implementation

void RuntimeHardwareConfigDialog::initializeLibraryStatusTab()
{
    // Connect library overview table selection changes
    connect(pu_ui->libraryOverviewTable, &QTableWidget::currentItemChanged,
            this, &RuntimeHardwareConfigDialog::onLibrarySelectionChanged);
    
    // Connect configuration controls
    connect(pu_ui->userLibraryPathEdit, &QLineEdit::textChanged,
            this, [this]() { onLibraryPathChanged(d_currentLibraryKey); });
    
    connect(pu_ui->additionalPathsEdit, &QLineEdit::textChanged,
            this, [this]() { onLibraryPathChanged(d_currentLibraryKey); });
    
    connect(pu_ui->autoDiscoveryCheckBox, &QCheckBox::toggled,
            this, [this]() { onLibraryPathChanged(d_currentLibraryKey); });
    
    connect(pu_ui->browseLibraryButton, &QPushButton::clicked,
            this, &RuntimeHardwareConfigDialog::onBrowseLibraryPath);
    
    connect(pu_ui->testLoadButton, &QPushButton::clicked,
            this, &RuntimeHardwareConfigDialog::onTestLoadLibrary);
    
    connect(pu_ui->refreshLibraryButton, &QPushButton::clicked,
            this, &RuntimeHardwareConfigDialog::refreshLibraryStatus);
    
    // Initialize installation guidance with platform-specific generic instructions
    pu_ui->installationGuidanceText->setHtml(getGenericInstallationGuidance());
    
    // Initialize library status display
    refreshLibraryStatus();
}

void RuntimeHardwareConfigDialog::refreshLibraryStatus()
{
    // Store current selection to restore later
    QTableWidgetItem* currentSelection = pu_ui->libraryOverviewTable->currentItem();
    QString selectedLibraryName;
    if (currentSelection != nullptr) {
        // Get the library name from the first column of the current row
        QTableWidgetItem* nameItem = pu_ui->libraryOverviewTable->item(currentSelection->row(), 0);
        if (nameItem != nullptr) {
            selectedLibraryName = nameItem->data(Qt::UserRole).toString();
        }
    }
    
    // Get references to all vendor libraries
    QList<QPair<QString, VendorLibrary*>> libraries;
    libraries.append({"Spectrum M4i", &SpectrumLibrary::instance()});
    libraries.append({"LabJack USB", &LabjackLibrary::instance()});
    
    // Initialize table structure if it's empty (first run)
    if (pu_ui->libraryOverviewTable->rowCount() == 0) {
        pu_ui->libraryOverviewTable->setRowCount(libraries.size());
        
        // Create table items once and set up basic properties
        for (int row = 0; row < libraries.size(); ++row) {
            const QString& displayName = libraries[row].first;
            
            // Library name (column 0)
            auto* nameItem = new QTableWidgetItem(displayName);
            nameItem->setFlags(nameItem->flags() & ~Qt::ItemIsEditable);
            nameItem->setData(Qt::UserRole, displayName);
            pu_ui->libraryOverviewTable->setItem(row, 0, nameItem);
            
            // Status (column 1)
            auto* statusItem = new QTableWidgetItem();
            statusItem->setFlags(statusItem->flags() & ~Qt::ItemIsEditable);
            pu_ui->libraryOverviewTable->setItem(row, 1, statusItem);
            
            // Version (column 2)
            auto* versionItem = new QTableWidgetItem();
            versionItem->setFlags(versionItem->flags() & ~Qt::ItemIsEditable);
            pu_ui->libraryOverviewTable->setItem(row, 2, versionItem);
            
            // Load path (column 3)
            auto* pathItem = new QTableWidgetItem();
            pathItem->setFlags(pathItem->flags() & ~Qt::ItemIsEditable);
            pu_ui->libraryOverviewTable->setItem(row, 3, pathItem);
        }
        
        // Set up column header stretch settings once
        pu_ui->libraryOverviewTable->horizontalHeader()->setStretchLastSection(true);
    }
    
    // Update existing table items with current library status
    QTableWidgetItem* itemToReselect = nullptr;
    
    for (int row = 0; row < libraries.size() && row < pu_ui->libraryOverviewTable->rowCount(); ++row) {
        const QString& displayName = libraries[row].first;
        VendorLibrary* library = libraries[row].second;
        
        // Update status (column 1)
        QString statusText = getLibraryStatusText(*library);
        QTableWidgetItem* statusItem = pu_ui->libraryOverviewTable->item(row, 1);
        if (statusItem != nullptr) {
            statusItem->setText(statusText);
            
            // Apply status-based color coding
            if (library->isAvailable()) {
                statusItem->setForeground(ThemeColors::getThemeAwareColor(ThemeColors::StatusSuccess, this));
            } else {
                statusItem->setForeground(ThemeColors::getThemeAwareColor(ThemeColors::StatusError, this));
            }
        }
        
        // Update version (column 2)
        QString versionText = getLibraryVersion(*library);
        QTableWidgetItem* versionItem = pu_ui->libraryOverviewTable->item(row, 2);
        if (versionItem != nullptr) {
            versionItem->setText(versionText);
        }
        
        // Update load path (column 3)
        QString pathText = library->loadedLibraryPath();
        if (pathText.isEmpty()) {
            pathText = "Not loaded";
        }
        QTableWidgetItem* pathItem = pu_ui->libraryOverviewTable->item(row, 3);
        if (pathItem != nullptr) {
            pathItem->setText(pathText);
        }
        
        // Check if this row should be reselected
        if (displayName == selectedLibraryName) {
            itemToReselect = pu_ui->libraryOverviewTable->item(row, 0); // Select first column
        }
    }
    
    // Restore selection if we had one previously
    if (itemToReselect != nullptr) {
        pu_ui->libraryOverviewTable->setCurrentItem(itemToReselect);
    }
    
    // Adjust column widths only if this was the initial setup
    if (selectedLibraryName.isEmpty()) {
        pu_ui->libraryOverviewTable->resizeColumnsToContents();
    }
    
    // Update details if a library is currently selected
    if (p_currentLibrary != nullptr) {
        updateLibraryDetails(*p_currentLibrary);
        updateLibraryConfiguration(*p_currentLibrary);
    }
}

void RuntimeHardwareConfigDialog::onLibrarySelectionChanged(QTableWidgetItem* current, QTableWidgetItem* previous)
{
    Q_UNUSED(previous)
    
    if (current == nullptr) {
        p_currentLibrary = nullptr;
        d_currentLibraryKey.clear();
        // Clear details and configuration panels
        pu_ui->libraryDetailsText->clear();
        pu_ui->userLibraryPathEdit->clear();
        pu_ui->additionalPathsEdit->clear();
        pu_ui->autoDiscoveryCheckBox->setChecked(true);
        // Restore generic installation guidance
        pu_ui->installationGuidanceText->setHtml(getGenericInstallationGuidance());
        return;
    }
    
    // Get selected library
    int row = current->row();
    QString libraryDisplayName = pu_ui->libraryOverviewTable->item(row, 0)->data(Qt::UserRole).toString();
    
    VendorLibrary* library = nullptr;
    QString libraryKey;
    
    if (libraryDisplayName == "Spectrum M4i") {
        library = &SpectrumLibrary::instance();
        libraryKey = BC::Key::Spectrum::spectrumM4i;
    } else if (libraryDisplayName == "LabJack USB") {
        library = &LabjackLibrary::instance();
        libraryKey = BC::Key::LabJack::labjackU3;
    }
    
    if (library != nullptr) {
        p_currentLibrary = library;
        d_currentLibraryKey = libraryKey;
        
        updateLibraryDetails(*library);
        updateLibraryConfiguration(*library);
    }
}

void RuntimeHardwareConfigDialog::onLibraryPathChanged(const QString& libraryKey)
{
    if (libraryKey.isEmpty() || p_currentLibrary == nullptr) {
        return;
    }
    
    // Update library configuration based on UI input
    QString userPath = pu_ui->userLibraryPathEdit->text().trimmed();
    QString additionalPaths = pu_ui->additionalPathsEdit->text().trimmed();
    bool autoDiscovery = pu_ui->autoDiscoveryCheckBox->isChecked();
    
    // Apply settings to staged configuration (no immediate effect)
    p_currentLibrary->setStagedUserProvidedPath(userPath);
    
    // Parse semicolon-separated additional paths
    QStringList pathList;
    if (!additionalPaths.isEmpty()) {
        pathList = additionalPaths.split(';', Qt::SkipEmptyParts);
        for (QString& path : pathList) {
            path = path.trimmed();
        }
    }
    p_currentLibrary->setStagedSearchPaths(pathList);
    
    p_currentLibrary->setStagedAutoDiscoveryEnabled(autoDiscovery);
    
    // Update visual indicators for staged changes
    updateStagingIndicators();
    updateAllLibraryStagingIndicators();
}

void RuntimeHardwareConfigDialog::onBrowseLibraryPath()
{
    if (p_currentLibrary == nullptr) {
        return;
    }
    
    QString currentPath = pu_ui->userLibraryPathEdit->text();
    QString startDir = currentPath.isEmpty() ? QDir::homePath() : currentPath;

    QString selectedPath = QFileDialog::getExistingDirectory(
        this,
        QString("Select Directory Containing %1 Library").arg(getLibraryDisplayName(*p_currentLibrary)),
        startDir
    );

    if (!selectedPath.isEmpty()) {
        pu_ui->userLibraryPathEdit->setText(selectedPath);
    }
}

void RuntimeHardwareConfigDialog::onTestLoadLibrary()
{
    if (p_currentLibrary == nullptr) {
        return;
    }
    
    // Check if there are staged changes to test
    if (!p_currentLibrary->hasUnstagedChanges()) {
        // No staged changes - test with current active settings
        bool success = p_currentLibrary->reloadLibrary();
        
        QString title = QString("Test Load - %1").arg(getLibraryDisplayName(*p_currentLibrary));
        if (success) {
            QMessageBox::information(this, title, "Library loaded successfully with current active settings!");
        } else {
            QString errorMsg = QString("Failed to load library with current active settings:\n\n%1").arg(p_currentLibrary->errorString());
            QMessageBox::warning(this, title, errorMsg);
        }
        
        // Refresh the status display
        refreshLibraryStatus();
        return;
    }
    
    // We have staged changes - warn user and test with temporary application
    QString title = QString("Test Load - %1").arg(getLibraryDisplayName(*p_currentLibrary));
    int result = QMessageBox::question(this, title, 
        "This will temporarily apply your staged changes to test the library loading.\n\n"
        "The changes will be reverted after testing. Continue?",
        QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
    
    if (result != QMessageBox::Yes) {
        return;
    }
    
    // Store current active state for rollback (save staging state)
    QString originalStagedUserPath = p_currentLibrary->getStagedUserProvidedPath();
    QStringList originalStagedSearchPaths = p_currentLibrary->getStagedSearchPaths();
    bool originalStagedAutoDiscovery = p_currentLibrary->isStagedAutoDiscoveryEnabled();
    
    // Temporarily apply staged settings
    bool applySuccess = p_currentLibrary->applyChanges();
    
    if (!applySuccess) {
        QMessageBox::warning(this, title, 
            QString("Failed to apply staged changes for testing:\n\n%1").arg(p_currentLibrary->errorString()));
        return;
    }
    
    // Show test result
    if (p_currentLibrary->isAvailable()) {
        QMessageBox::information(this, title, "Library loaded successfully with staged settings!");
    } else {
        QString errorMsg = QString("Failed to load library with staged settings:\n\n%1").arg(p_currentLibrary->errorString());
        QMessageBox::warning(this, title, errorMsg);
    }
    
    // Restore original staged state (user's pending changes)
    p_currentLibrary->setStagedUserProvidedPath(originalStagedUserPath);
    p_currentLibrary->setStagedSearchPaths(originalStagedSearchPaths);
    p_currentLibrary->setStagedAutoDiscoveryEnabled(originalStagedAutoDiscovery);
    
    // Refresh the status display
    refreshLibraryStatus();
    updateStagingIndicators();
}

void RuntimeHardwareConfigDialog::updateLibraryDetails(VendorLibrary& library)
{
    QString details;
    
    // Library name and description
    details += QString("<h3>%1</h3>").arg(getLibraryDisplayName(library));
    details += QString("<p><b>Library Name:</b> %1</p>").arg(library.libraryName());
    
    // Status information
    details += QString("<p><b>Status:</b> %1</p>").arg(getLibraryStatusText(library));
    
    if (library.isAvailable()) {
        // Available - show success information
        details += QString("<p style='color: %1;'><b>✓ Library is available and ready</b></p>")
                    .arg(ThemeColors::getCSSColor(ThemeColors::StatusSuccess, this));
        
        QString loadedPath = library.loadedLibraryPath();
        if (!loadedPath.isEmpty()) {
            details += QString("<p><b>Loaded from:</b><br><code>%1</code></p>").arg(loadedPath);
        }
        
        // Version information
        QString version = getLibraryVersion(library);
        if (version != "Unknown") {
            details += QString("<p><b>Version:</b> %1</p>").arg(version);
        }
        
    } else {
        // Not available - show error information
        details += QString("<p style='color: %1;'><b>✗ Library is not available</b></p>")
                    .arg(ThemeColors::getCSSColor(ThemeColors::StatusError, this));
        
        QString errorMsg = library.errorString();
        if (!errorMsg.isEmpty()) {
            details += QString("<p><b>Error:</b> %1</p>").arg(errorMsg);
        }
        
        if (library.wasLoadingAttempted()) {
            details += "<p><b>Loading was attempted</b> - check configuration settings below.</p>";
        } else {
            details += "<p><b>Loading not attempted</b> - library will be loaded when needed.</p>";
        }
    }
    
    // Platform-specific library names
    QStringList platformNames = library.platformLibraryNames();
    if (!platformNames.isEmpty()) {
        details += "<p><b>Platform library names:</b><br>";
        for (const QString& name : platformNames) {
            details += QString("• <code>%1</code><br>").arg(name);
        }
        details += "</p>";
    }
    
    // Default search paths
    QStringList defaultPaths = library.defaultSearchPaths();
    if (!defaultPaths.isEmpty()) {
        details += "<p><b>Default search paths:</b><br>";
        for (const QString& path : defaultPaths) {
            details += QString("• <code>%1</code><br>").arg(path);
        }
        details += "</p>";
    }
    
    pu_ui->libraryDetailsText->setHtml(details);
}

void RuntimeHardwareConfigDialog::updateLibraryConfiguration(VendorLibrary& library)
{
    // Block signals to prevent recursive calls
    pu_ui->userLibraryPathEdit->blockSignals(true);
    pu_ui->additionalPathsEdit->blockSignals(true);
    pu_ui->autoDiscoveryCheckBox->blockSignals(true);
    
    // Load staged settings (what user is editing, not active configuration)
    pu_ui->userLibraryPathEdit->setText(library.getStagedUserProvidedPath());
    
    QStringList userPaths = library.getStagedSearchPaths();
    pu_ui->additionalPathsEdit->setText(userPaths.join(";"));
    
    pu_ui->autoDiscoveryCheckBox->setChecked(library.isStagedAutoDiscoveryEnabled());
    
    // Re-enable signals
    pu_ui->userLibraryPathEdit->blockSignals(false);
    pu_ui->additionalPathsEdit->blockSignals(false);
    pu_ui->autoDiscoveryCheckBox->blockSignals(false);
    
    // Enable controls
    pu_ui->userLibraryPathEdit->setEnabled(true);
    pu_ui->browseLibraryButton->setEnabled(true);
    pu_ui->additionalPathsEdit->setEnabled(true);
    pu_ui->autoDiscoveryCheckBox->setEnabled(true);
    pu_ui->testLoadButton->setEnabled(true);
    
    // Update installation guidance with library-specific instructions
    pu_ui->installationGuidanceText->setHtml(library.getInstallationInstructions());
    
    // Update visual indicators for staging state
    updateStagingIndicators();
}

QString RuntimeHardwareConfigDialog::getLibraryStatusText(VendorLibrary& library)
{
    if (library.isAvailable()) {
        return "Available";
    } else if (library.wasLoadingAttempted()) {
        return "Error";
    } else {
        return "Not Found";
    }
}

QString RuntimeHardwareConfigDialog::getLibraryDisplayName(VendorLibrary& library)
{
    // Map library instances to display names
    if (&library == &SpectrumLibrary::instance()) {
        return "Spectrum M4i";
    } else if (&library == &LabjackLibrary::instance()) {
        return "LabJack USB";
    } else {
        return library.libraryName();
    }
}

QString RuntimeHardwareConfigDialog::getLibraryVersion(VendorLibrary& library)
{
    if (!library.isAvailable()) {
        return "Unknown";
    }
    
    // Use the generic getVersionInfo() method first
    QString versionInfo = library.getVersionInfo();
    if (!versionInfo.isEmpty()) {
        return versionInfo;
    }
    
    // Fallback to specific library implementations for backward compatibility
    if (&library == &LabjackLibrary::instance()) {
        LabjackLibrary& ljLib = static_cast<LabjackLibrary&>(library);
        if (ljLib.LJUSB_GetLibraryVersion != nullptr) {
            try {
                float version = ljLib.LJUSB_GetLibraryVersion();
                return QString::number(version, 'f', 2);
            } catch (...) {
                // If version call fails, just return "Available"
                return "Available";
            }
        }
    }
    
    return "Available";
}

void RuntimeHardwareConfigDialog::updateStagingIndicators()
{
    if (p_currentLibrary == nullptr) {
        return;
    }
    
    // Update UI elements to show staging state
    bool hasChanges = p_currentLibrary->hasUnstagedChanges();
    
    // Update library configuration panel label to show staging state
    QString configLabel = "Library Configuration";
    if (hasChanges) {
        configLabel += " *";  // Add asterisk for pending changes
        pu_ui->libraryConfigPanelLabel->setStyleSheet(
            QString("QLabel { font-weight: bold; color: %1; }")
            .arg(ThemeColors::getCSSColor(ThemeColors::StatusInfo, this)));
    } else {
        pu_ui->libraryConfigPanelLabel->setStyleSheet("QLabel { font-weight: bold; }");
    }
    pu_ui->libraryConfigPanelLabel->setText(configLabel);
    
    // Update test load button text based on staging state
    if (hasChanges) {
        pu_ui->testLoadButton->setText("Test Load (Staged)");
        pu_ui->testLoadButton->setStyleSheet(
            QString("QPushButton { color: %1; font-weight: bold; }")
            .arg(ThemeColors::getCSSColor(ThemeColors::StatusInfo, this)));
    } else {
        pu_ui->testLoadButton->setText("Test Load");
        pu_ui->testLoadButton->setStyleSheet("");
    }
    
    // Update form controls to show modified state
    updateControlStagingIndicator(pu_ui->userLibraryPathEdit, 
        p_currentLibrary->getStagedUserProvidedPath() != p_currentLibrary->getActiveUserProvidedPath());
        
    updateControlStagingIndicator(pu_ui->additionalPathsEdit,
        p_currentLibrary->getStagedSearchPaths() != p_currentLibrary->getActiveSearchPaths());
        
    updateControlStagingIndicator(pu_ui->autoDiscoveryCheckBox,
        p_currentLibrary->isStagedAutoDiscoveryEnabled() != p_currentLibrary->isActiveAutoDiscoveryEnabled());
}

void RuntimeHardwareConfigDialog::updateControlStagingIndicator(QWidget* control, bool isModified)
{
    if (control == nullptr) {
        return;
    }
    
    if (isModified) {
        // Apply visual indication for modified controls
        control->setProperty("staging-modified", true);
        control->setStyleSheet(
            QString("QLineEdit { border: 2px solid %1; } QCheckBox { color: %1; font-weight: bold; }")
            .arg(ThemeColors::getCSSColor(ThemeColors::StatusInfo, this)));
    } else {
        // Remove visual indication
        control->setProperty("staging-modified", false);
        control->setStyleSheet("");
    }
}

void RuntimeHardwareConfigDialog::revertAllLibraryChanges()
{
    // Revert staged changes for all vendor libraries
    SpectrumLibrary::instance().revertChanges();
    LabjackLibrary::instance().revertChanges();
    
    // Update UI to reflect reverted state if library is currently selected
    if (p_currentLibrary != nullptr) {
        updateLibraryConfiguration(*p_currentLibrary);
    }
}

void RuntimeHardwareConfigDialog::updateAllLibraryStagingIndicators()
{
    // Update staging indicators for all libraries, including tab-level indicators
    bool hasAnyLibraryChanges = SpectrumLibrary::instance().hasUnstagedChanges() ||
                                LabjackLibrary::instance().hasUnstagedChanges();
    
    // Update the Library Status tab title to show staging state
    QString tabTitle = "Library Status";
    if (hasAnyLibraryChanges) {
        tabTitle += " *";  // Add asterisk to tab for global staging indicator
    }
    
    // Find and update the tab title
    int tabIndex = -1;
    for (int i = 0; i < pu_ui->mainTabWidget->count(); ++i) {
        if (pu_ui->mainTabWidget->widget(i) == pu_ui->libraryStatusTab) {
            tabIndex = i;
            break;
        }
    }
    
    if (tabIndex >= 0) {
        pu_ui->mainTabWidget->setTabText(tabIndex, tabTitle);
        
        // Apply styling to tab if there are changes
        if (hasAnyLibraryChanges) {
            pu_ui->mainTabWidget->tabBar()->setTabTextColor(tabIndex, 
                ThemeColors::getThemeAwareColor(ThemeColors::StatusInfo, this));
        } else {
            pu_ui->mainTabWidget->tabBar()->setTabTextColor(tabIndex, 
                palette().color(QPalette::Text));
        }
    }
    
    // Update current library's indicators if one is selected
    if (p_currentLibrary != nullptr) {
        updateStagingIndicators();
    }
}

QString RuntimeHardwareConfigDialog::getGenericInstallationGuidance() const
{
#ifdef Q_OS_LINUX
    return QString(
        "<p><b>Installing Vendor Libraries on Linux:</b></p>"
        "<ol>"
        "<li><b>Download:</b> Visit the vendor's official website to download the Linux driver packages</li>"
        "<li><b>Prerequisites:</b> Ensure you have necessary development tools:"
        "<pre>sudo apt update\nsudo apt install build-essential linux-headers-$(uname -r)</pre></li>"
        "<li><b>Installation:</b> Most vendors provide installation scripts that handle:"
        "<ul>"
        "<li>Kernel module compilation and loading</li>"
        "<li>Library installation to standard paths (typically <code>/usr/local/lib/</code> or <code>/opt/</code>)</li>"
        "<li>Device permissions and udev rules</li>"
        "</ul></li>"
        "<li><b>User Permissions:</b> Add your user to the appropriate groups for device access</li>"
        "<li><b>Library Path:</b> Update <code>LD_LIBRARY_PATH</code> if libraries are in non-standard locations</li>"
        "</ol>"
        "<p><b>Common Troubleshooting:</b></p>"
        "<ul>"
        "<li>Check kernel module loading: <code>lsmod | grep [vendor]</code></li>"
        "<li>Verify device files exist: <code>ls -la /dev/</code></li>"
        "<li>Update library cache: <code>sudo ldconfig</code></li>"
        "<li>Check USB device detection: <code>lsusb</code></li>"
        "</ul>"
    );
#elif defined(Q_OS_WIN)
    return QString(
        "<p><b>Installing Vendor Libraries on Windows:</b></p>"
        "<ol>"
        "<li><b>Download:</b> Visit the vendor's official website to download Windows driver packages</li>"
        "<li><b>Administrator Rights:</b> Always run installers as Administrator</li>"
        "<li><b>Installation:</b> Windows installers typically handle:"
        "<ul>"
        "<li>Driver installation and digital signature verification</li>"
        "<li>Library registration in system directories</li>"
        "<li>Device driver association</li>"
        "</ul></li>"
        "<li><b>Reboot:</b> Most hardware drivers require a system restart</li>"
        "<li><b>Verification:</b> Check Device Manager for proper device recognition</li>"
        "</ol>"
        "<p><b>Common Troubleshooting:</b></p>"
        "<ul>"
        "<li>Disable Driver Signature Enforcement if needed (advanced users)</li>"
        "<li>Check Windows Event Viewer for driver errors</li>"
        "<li>Ensure proper USB cable connections</li>"
        "<li>Try different USB ports if devices aren't detected</li>"
        "</ul>"
    );
#elif defined(Q_OS_MACOS)
    return QString(
        "<p><b>Installing Vendor Libraries on macOS:</b></p>"
        "<ol>"
        "<li><b>Download:</b> Visit the vendor's official website to download macOS driver packages</li>"
        "<li><b>Security:</b> Allow installation from identified developers in System Preferences > Security & Privacy</li>"
        "<li><b>Installation:</b> Run the .pkg installer and follow the setup wizard</li>"
        "<li><b>System Extensions:</b> Grant permission for system extensions when prompted</li>"
        "<li><b>Reboot:</b> Restart your Mac to load kernel extensions</li>"
        "</ol>"
        "<p><b>Apple Silicon Considerations:</b></p>"
        "<ul>"
        "<li>Ensure the driver supports ARM64 architecture</li>"
        "<li>Some vendors may require Rosetta 2 for Intel-compiled drivers</li>"
        "</ul>"
        "<p><b>Common Troubleshooting:</b></p>"
        "<ul>"
        "<li>Check System Preferences > Security & Privacy for blocked extensions</li>"
        "<li>Verify kernel extension loading: <code>kextstat</code></li>"
        "<li>Check USB device detection: <code>system_profiler SPUSBDataType</code></li>"
        "</ul>"
    );
#else
    return QString(
        "<p><b>Installing Vendor Libraries:</b></p>"
        "<p>Please visit the vendor's official website to download the appropriate driver package for your operating system.</p>"
        "<p>Most hardware vendors provide comprehensive installation instructions and support documentation specific to their products.</p>"
        "<p><b>General Guidelines:</b></p>"
        "<ul>"
        "<li>Always download drivers from official vendor websites</li>"
        "<li>Follow vendor-specific installation procedures</li>"
        "<li>Ensure your system meets hardware and software requirements</li>"
        "<li>Check vendor documentation for platform-specific considerations</li>"
        "</ul>"
    );
#endif
}

bool RuntimeHardwareConfigDialog::getTypeDefaultThreaded(const QString& hardwareType)
{
    // These types default to threaded based on their intermediate class constructors
    static const QSet<QString> threadedTypes = {
        QString(FtmwScope::staticMetaObject.className()),
        QString(AWG::staticMetaObject.className()),
        QString(IOBoard::staticMetaObject.className()),
        QString(LifScope::staticMetaObject.className()),
        QString(LifLaser::staticMetaObject.className()),
        QString(GpibController::staticMetaObject.className())
    };
    return threadedTypes.contains(hardwareType);
}

#include "runtimehardwareconfigdialog.h"
#include "runtimehardwareconfigdialog_ui.h"
#include "addprofiledialog.h"

#include <QTreeWidgetItem>
#include <QListWidgetItem>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QGroupBox>
#include <QCheckBox>
#include <QMessageBox>
#include <QDialogButtonBox>
#include <QSignalBlocker>

#include <data/bcglobals.h>
#include <data/settings/hardwarekeys.h>
#include <data/storage/applicationconfigmanager.h>
#include <data/loadout/loadoutmanager.h>
#include <hardware/core/hardwareregistry.h>
#include <hardware/core/hardwareprofilemanager.h>
#include <gui/style/themecolors.h>
#include <gui/widget/pythonsettingswidget.h>
#include <gui/widget/librarystatuswidget.h>

#include <QInputDialog>
#include <QSet>

using namespace Qt::StringLiterals;

// Returns the set of hwKeys that FTMW presets care about (AWG, FtmwDigitizer, Clock).
// Implementations are intentionally excluded; only the key identity matters for drift.
static QSet<QString> ftmwRelevantHwKeys(const std::map<QString, QString, std::less<>> &hwMap)
{
    static const QSet<QString> relevantTypes {
        FtmwDigitizer::staticMetaObject.className(),
        AWG::staticMetaObject.className(),
        Clock::staticMetaObject.className()
    };
    QSet<QString> keys;
    for (const auto &[hwKey, impl] : hwMap) {
        auto [type, label] = BC::Key::parseKey(hwKey);
        if (relevantTypes.contains(type))
            keys.insert(hwKey);
    }
    return keys;
}

RuntimeHardwareConfigDialog::RuntimeHardwareConfigDialog(QWidget *parent)
    : QDialog(parent),
      pu_ui(new Ui::RuntimeHardwareConfigDialog),
      p_libraryStatusWidget(nullptr)
{
    pu_ui->setupUi(this);

    setWindowIcon(ThemeColors::createThemedIcon(":/icons/bc_logo_trans.svg", ThemeColors::IconPrimary, this));

    // Apply theme-aware styling to validation status label
    pu_ui->applyValidationStatusStyling(this);

    // Ensure system profiles exist before reading configuration
    HardwareProfileManager::instance().ensureSystemProfiles();

    // Initialize both original and preview state from current runtime configuration FIRST
    d_originalRuntimeConfig = RuntimeHardwareConfig::constInstance().getCurrentHardware();
    d_previewRuntimeConfig = d_originalRuntimeConfig;

    // Track the active loadout (Save target and FTMW snapshot pointer; does not drive the preview)
    d_activeLoadoutName = LoadoutManager::instance().currentLoadoutName();

    // Seed the active loadout's hardware map from the current runtime config if it is empty.
    // This handles first-run and any loadout that was created before hardware was configured.
    {
        auto loadout = LoadoutManager::instance().getLoadout(d_activeLoadoutName);
        if (loadout.has_value() && loadout->hardwareMap.empty() && !d_originalRuntimeConfig.empty()) {
            loadout->hardwareMap = std::map<QString,QString,std::less<>>(
                d_originalRuntimeConfig.begin(), d_originalRuntimeConfig.end());
            LoadoutManager::instance().putLoadout(*loadout);
        }
    }

    // Initialize threaded overrides from stored values
    for (auto& [hwKey, impl] : d_originalRuntimeConfig) {
        auto override = RuntimeHardwareConfig::constInstance().getThreaded(hwKey);
        if (override.has_value())
            d_profileOverrides[hwKey].threaded = *override;
    }

    // Auto-activate system profiles for required types that have no active entry
    ensureRequiredTypes();

    // Populate configuration overview with actual hardware data
    populateConfigurationOverview();

    // Populate hardware browser and connect selection handling
    populateHardwareBrowser();

    // Initialize loadout list and wire loadout buttons
    connect(pu_ui->p_loadoutList, &QListWidget::currentItemChanged,
            this, [this](QListWidgetItem*, QListWidgetItem*) { onLoadoutListSelectionChanged(); });
    connect(pu_ui->p_loadoutActivate, &QPushButton::clicked,
            this, &RuntimeHardwareConfigDialog::onLoadoutActivate);
    connect(pu_ui->p_loadoutSave, &QPushButton::clicked,
            this, &RuntimeHardwareConfigDialog::onLoadoutSave);
    connect(pu_ui->p_loadoutSaveAs, &QPushButton::clicked,
            this, &RuntimeHardwareConfigDialog::onLoadoutSaveAs);
    connect(pu_ui->p_loadoutCopy, &QPushButton::clicked,
            this, &RuntimeHardwareConfigDialog::onLoadoutCopy);
    connect(pu_ui->p_loadoutDelete, &QPushButton::clicked,
            this, &RuntimeHardwareConfigDialog::onLoadoutDelete);
    populateLoadoutList();

    pu_ui->p_loadoutActivate->setIcon(ThemeColors::createThemedIconWithStates(
        ":/icons/bolt.svg", ThemeColors::IconPrimary, ThemeColors::DisabledText, this));
    pu_ui->p_loadoutSave->setIcon(ThemeColors::createThemedIconWithStates(
        ":/icons/archive-box.svg", ThemeColors::IconPrimary, ThemeColors::DisabledText, this));
    pu_ui->p_loadoutSaveAs->setIcon(ThemeColors::createThemedIconWithStates(
        ":/icons/arrow-up-on-square.svg", ThemeColors::IconPrimary, ThemeColors::DisabledText, this));
    pu_ui->p_loadoutCopy->setIcon(ThemeColors::createThemedIconWithStates(
        ":/icons/document-duplicate.svg", ThemeColors::IconPrimary, ThemeColors::DisabledText, this));
    pu_ui->p_loadoutDelete->setIcon(ThemeColors::createThemedIconWithStates(
        ":/icons/trash.svg", ThemeColors::StatusError, ThemeColors::DisabledText, this));

    // Initialize Library Status tab
    p_libraryStatusWidget = new LibraryStatusWidget(pu_ui->libraryStatusTab);
    pu_ui->libraryStatusTab->layout()->addWidget(p_libraryStatusWidget);
    connect(p_libraryStatusWidget, &LibraryStatusWidget::stagingStateChanged,
            this, &RuntimeHardwareConfigDialog::onLibraryStagingStateChanged);

    // Connect dialog buttons
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
    onLibraryStagingStateChanged(p_libraryStatusWidget->hasUnstagedChanges());
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
        listItem->setData(Qt::UserRole, profileLabel);

        // Apply visual distinction for system profiles
        if (isSysProfile) {
            listItem->setForeground(ThemeColors::getThemeAwareColor(ThemeColors::SubtleText, this));
            QFont f = listItem->font();
            f.setItalic(true);
            listItem->setFont(f);
        }

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
        auto it = d_profileOverrides.find(activeHwKey);
        if (it != d_profileOverrides.end() && it->second.threaded.has_value())
            threadedChecked = *it->second.threaded;
    }

    QString checkboxLabel = tr("Run in own thread");
    if (typeDefault)
        checkboxLabel += tr(" (recommended)");
    auto* threadedCheckbox = new QCheckBox(checkboxLabel, advancedContainer);
    threadedCheckbox->setObjectName("threadedCheckbox");
    threadedCheckbox->setChecked(threadedChecked);
    threadedCheckbox->setEnabled(!activeHwKey.isEmpty());

    advancedLayout->addWidget(threadedCheckbox);

    // Helper: derive default class name from implementation (e.g., "PythonAwg" -> "AwgDriver")
    auto defaultPythonClassName = [](const QString& impl) -> QString {
        if (!impl.startsWith(QStringLiteral("Python")))
            return QString();
        return impl.mid(6) + QStringLiteral("Driver");
    };

    // Helper: check whether the selected profile is a Python implementation
    auto isSelectedPython = [hardwareType, profilesList]() -> bool {
        auto selected = profilesList->selectedItems();
        if (selected.isEmpty()) return false;
        QString label = selected.first()->data(Qt::UserRole).toString();
        if (label.isEmpty()) return false;
        QString impl = HardwareProfileManager::instance().getImplementation(hardwareType, label);
        return impl.contains(QStringLiteral("Python"));
    };

    // Python settings widget
    auto* pythonWidget = new PythonSettingsWidget(advancedContainer);
    pythonWidget->setObjectName("pythonWidget");

    // Initialize visibility and values from the current selection
    bool isPython = isSelectedPython();
    pythonWidget->setVisible(isPython);
    if (isPython && !activeHwKey.isEmpty()) {
        auto [type, label] = BC::Key::parseKey(activeHwKey);
        QString impl = HardwareProfileManager::instance().getImplementation(type, label);

        auto oit = d_profileOverrides.find(activeHwKey);
        const ProfileOverrides* ov = (oit != d_profileOverrides.end()) ? &oit->second : nullptr;

        pythonWidget->setScriptPath(
            (ov && ov->pythonScript.has_value()) ? *ov->pythonScript
            : HardwareProfileManager::instance().getPythonScriptPath(type, label));

        pythonWidget->setClassNamePlaceholder(defaultPythonClassName(impl));
        pythonWidget->setClassName(
            (ov && ov->pythonClass.has_value()) ? *ov->pythonClass
            : HardwareProfileManager::instance().getPythonClassName(type, label));

        pythonWidget->setEnvPath(
            (ov && ov->pythonEnv.has_value()) ? *ov->pythonEnv
            : HardwareProfileManager::instance().getPythonEnvPath(type, label));
    }

    // Wire PythonSettingsWidget signals into preview configs
    connect(pythonWidget, &PythonSettingsWidget::scriptPathChanged, this,
            [this, getActiveHwKey](const QString& text) {
        QString hwKey = getActiveHwKey();
        if (hwKey.isEmpty()) return;
        d_profileOverrides[hwKey].pythonScript = text;
    });

    connect(pythonWidget, &PythonSettingsWidget::classNameChanged, this,
            [this, getActiveHwKey](const QString& text) {
        QString hwKey = getActiveHwKey();
        if (hwKey.isEmpty()) return;
        d_profileOverrides[hwKey].pythonClass = text;
    });

    connect(pythonWidget, &PythonSettingsWidget::envPathChanged, this,
            [this, getActiveHwKey](const QString& text) {
        QString hwKey = getActiveHwKey();
        if (hwKey.isEmpty()) return;
        d_profileOverrides[hwKey].pythonEnv = text;
    });

    advancedLayout->addWidget(pythonWidget);

    advancedContainer->setLayout(advancedLayout);

    // Toggle expand/collapse
    connect(advancedButton, &QPushButton::clicked, this, [advancedButton, advancedContainer](bool) {
        bool visible = !advancedContainer->isVisible();
        advancedContainer->setVisible(visible);
        advancedButton->setText(visible ? QString::fromUtf8("\u25bc Advanced")
                                        : QString::fromUtf8("\u25b6 Advanced"));
    });

    // Update Advanced section when profile selection changes in the list
    connect(profilesList, &QListWidget::itemSelectionChanged, this,
            [this, hardwareType, profilesList, threadedCheckbox, typeDefault, getActiveHwKey,
             pythonWidget, defaultPythonClassName]() {
        QString hwKey = getActiveHwKey();
        if (hwKey.isEmpty()) {
            threadedCheckbox->setEnabled(false);
            threadedCheckbox->setChecked(typeDefault);
            pythonWidget->setVisible(false);
            return;
        }
        threadedCheckbox->setEnabled(true);
        {
            auto it = d_profileOverrides.find(hwKey);
            bool checked = (it != d_profileOverrides.end() && it->second.threaded.has_value())
                           ? *it->second.threaded : typeDefault;
            QSignalBlocker blocker(threadedCheckbox);
            threadedCheckbox->setChecked(checked);
        }

        // Show/hide based on whether selected profile is a Python implementation
        auto [type, label] = BC::Key::parseKey(hwKey);
        QString impl = HardwareProfileManager::instance().getImplementation(type, label);
        bool isPythonImpl = impl.contains(QStringLiteral("Python"));
        pythonWidget->setVisible(isPythonImpl);

        if (isPythonImpl) {
            auto oit = d_profileOverrides.find(hwKey);
            const ProfileOverrides* ov = (oit != d_profileOverrides.end()) ? &oit->second : nullptr;

            pythonWidget->setScriptPath(
                (ov && ov->pythonScript.has_value()) ? *ov->pythonScript
                : HardwareProfileManager::instance().getPythonScriptPath(type, label));

            pythonWidget->setClassNamePlaceholder(defaultPythonClassName(impl));
            pythonWidget->setClassName(
                (ov && ov->pythonClass.has_value()) ? *ov->pythonClass
                : HardwareProfileManager::instance().getPythonClassName(type, label));

            pythonWidget->setEnvPath(
                (ov && ov->pythonEnv.has_value()) ? *ov->pythonEnv
                : HardwareProfileManager::instance().getPythonEnvPath(type, label));
        }
    });

    // Wire checkbox changes into profile overrides
    connect(threadedCheckbox, &QCheckBox::toggled, this,
            [this, hardwareType, profilesList, getActiveHwKey](bool checked) {
        QString hwKey = getActiveHwKey();
        if (hwKey.isEmpty()) return;
        d_profileOverrides[hwKey].threaded = checked;
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
    populateConfigurationOverview();
    populateHardwareBrowser();
    validatePreviewConfiguration();
    refreshLoadoutDirtyIndicator();
}

void RuntimeHardwareConfigDialog::onAddProfile(const QString& hardwareType)
{
    if (hardwareType.isEmpty()) {
        return;
    }

    AddProfileDialog addDialog(hardwareType, this);
    if (addDialog.exec() != QDialog::Accepted) {
        return;
    }

    QString implementation = addDialog.selectedImplementation();
    QString actualLabel = addDialog.profileLabel();
    QString pythonScript = addDialog.pythonScriptPath();

    // Automatically activate the new profile if it's the first profile of this hardware type
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

    // Store Python script path if a template was copied
    if (!pythonScript.isEmpty()) {
        QString profileKey = BC::Key::hwKey(hardwareType, actualLabel);
        d_profileOverrides[profileKey].pythonScript = pythonScript;
    }

    // Refresh the right panel to show the new profile with correct state
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
        message = QString("Are you sure you want to delete %1 profiles?\n\nProfiles to delete:\n%2\n\nThis action cannot be undone.")
                     .arg(profilesToRemove.size())
                     .arg(profilesToRemove.join("\n"));
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
                // Remove from BOTH preview and original runtime configs
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
    if (isPreviewDirty()) {
        QMessageBox msgBox(this);
        msgBox.setWindowTitle(u"Unsaved Changes"_s);
        msgBox.setText(u"The active loadout \"%1\" has unsaved changes."_s.arg(d_activeLoadoutName));
        auto *saveBtn   = msgBox.addButton(u"Save and apply"_s,         QMessageBox::AcceptRole);
        auto *applyBtn  = msgBox.addButton(u"Apply without saving"_s,   QMessageBox::DestructiveRole);
        auto *saveAsBtn = msgBox.addButton(u"Save to new loadout..."_s, QMessageBox::ActionRole);
        msgBox.addButton(QMessageBox::Cancel);
        msgBox.exec();
        const auto *clicked = msgBox.clickedButton();
        if (clicked == saveBtn)
            onLoadoutSave();
        else if (clicked == saveAsBtn)
            onLoadoutSaveAs();
        else if (clicked != applyBtn)
            return;
    }

    // Apply the preview configuration to the runtime configuration
    auto& runtimeConfig = RuntimeHardwareConfig::instance();

    if (runtimeConfig.applyConfiguration(d_previewRuntimeConfig)) {
        // Apply per-profile overrides
        for (auto& [hwKey, ov] : d_profileOverrides) {
            if (ov.threaded.has_value())
                runtimeConfig.setThreaded(hwKey, *ov.threaded);
            auto [type, label] = BC::Key::parseKey(hwKey);
            if (!type.isEmpty() && !label.isEmpty()) {
                auto& pm = HardwareProfileManager::instance();
                if (ov.pythonScript.has_value())
                    pm.setPythonScriptPath(type, label, *ov.pythonScript);
                if (ov.pythonClass.has_value())
                    pm.setPythonClassName(type, label, *ov.pythonClass);
                if (ov.pythonEnv.has_value())
                    pm.setPythonEnvPath(type, label, *ov.pythonEnv);
            }
        }

        // Save all hardware profiles to persistent storage now that dialog is finished
        auto& profileManager = HardwareProfileManager::instance();
        profileManager.saveProfiles();

        // Library configuration changes are staged and will be applied
        // by HardwareManager::syncWithRuntimeConfig() which is called by MainWindow
        // after dialog close.

        // Record the active loadout as the persisted current loadout
        LoadoutManager::instance().setCurrentLoadoutName(d_activeLoadoutName);

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
    auto& runtimeConfig = RuntimeHardwareConfig::instance();

    // Apply the original configuration to undo preview changes. Skip when the
    // originals are empty (e.g. first-run cancel): applying an empty map would
    // deactivate every profile in HardwareProfileManager and override the
    // virtual-defaults fallback in activateMissingSystemProfiles().
    if (!d_originalRuntimeConfig.empty())
        runtimeConfig.applyConfiguration(d_originalRuntimeConfig);

    // Revert all library staged changes
    p_libraryStatusWidget->revertAllChanges();

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
        styleSheet = QString("QLabel { color: %1; }")
            .arg(ThemeColors::getCSSColor(ThemeColors::StatusSuccess, this));
    }

    pu_ui->validationStatusLabel->setStyleSheet(styleSheet);
}

void RuntimeHardwareConfigDialog::onLibraryStagingStateChanged(bool hasChanges)
{
    // Update the Library Status tab title to show staging state
    int tabIndex = -1;
    for (int i = 0; i < pu_ui->mainTabWidget->count(); ++i) {
        if (pu_ui->mainTabWidget->widget(i) == pu_ui->libraryStatusTab) {
            tabIndex = i;
            break;
        }
    }

    if (tabIndex >= 0) {
        QString tabTitle = hasChanges ? "Library Status *" : "Library Status";
        pu_ui->mainTabWidget->setTabText(tabIndex, tabTitle);

        if (hasChanges) {
            pu_ui->mainTabWidget->tabBar()->setTabTextColor(tabIndex,
                ThemeColors::getThemeAwareColor(ThemeColors::StatusInfo, this));
        } else {
            pu_ui->mainTabWidget->tabBar()->setTabTextColor(tabIndex,
                palette().color(QPalette::Text));
        }
    }
}

bool RuntimeHardwareConfigDialog::getTypeDefaultThreaded(const QString& hardwareType)
{
    // These types default to threaded based on their intermediate class constructors
    static const QSet<QString> threadedTypes = {
        QString(FtmwDigitizer::staticMetaObject.className()),
        QString(AWG::staticMetaObject.className()),
        QString(IOBoard::staticMetaObject.className()),
        QString(LifDigitizer::staticMetaObject.className()),
        QString(LifLaser::staticMetaObject.className()),
        QString(GpibController::staticMetaObject.className())
    };
    return threadedTypes.contains(hardwareType);
}

void RuntimeHardwareConfigDialog::ensureRequiredTypes()
{
    auto &profileManager = HardwareProfileManager::instance();
    HardwareRegistry &registry = HardwareRegistry::instance();
    for (const QString &hwType : registry.getHardwareTypes()) {
        if (!RuntimeHardwareConfig::isHardwareRequired(hwType))
            continue;
        bool hasEntry = false;
        for (auto &[key, impl] : d_previewRuntimeConfig) {
            auto [type, label] = BC::Key::parseKey(key);
            if (type == hwType) {
                hasEntry = true;
                break;
            }
        }
        if (!hasEntry) {
            const QString impl = profileManager.getImplementation(hwType, QStringLiteral("virtual"));
            if (!impl.isEmpty())
                d_previewRuntimeConfig[BC::Key::hwKey(hwType, QStringLiteral("virtual"))] = impl;
        }
    }
}

QString RuntimeHardwareConfigDialog::selectedLoadoutName() const
{
    const auto *item = pu_ui->p_loadoutList->currentItem();
    return item ? item->data(Qt::UserRole).toString() : QString{};
}

bool RuntimeHardwareConfigDialog::isPreviewDirty() const
{
    const auto loadout = LoadoutManager::instance().getLoadout(d_activeLoadoutName);
    if (!loadout.has_value())
        return !d_previewRuntimeConfig.empty();
    return d_previewRuntimeConfig != loadout->hardwareMap;
}

void RuntimeHardwareConfigDialog::populateLoadoutList()
{
    QSignalBlocker blocker(pu_ui->p_loadoutList);
    const auto sel = selectedLoadoutName();
    pu_ui->p_loadoutList->clear();
    const bool dirty = isPreviewDirty();
    for (const auto &n : LoadoutManager::instance().loadoutNames()) {
        QString display = n;
        if (n == d_activeLoadoutName) {
            display += u" (active)"_s;
            if (dirty)
                display += u"*"_s;
        }
        auto *item = new QListWidgetItem(display, pu_ui->p_loadoutList);
        item->setData(Qt::UserRole, n);
    }
    const QString toSelect = sel.isEmpty() ? d_activeLoadoutName : sel;
    for (int i = 0; i < pu_ui->p_loadoutList->count(); ++i) {
        if (pu_ui->p_loadoutList->item(i)->data(Qt::UserRole).toString() == toSelect) {
            pu_ui->p_loadoutList->setCurrentRow(i);
            break;
        }
    }
    updateLoadoutButtonStates();
}

void RuntimeHardwareConfigDialog::refreshLoadoutDirtyIndicator()
{
    const bool dirty = isPreviewDirty();
    for (int i = 0; i < pu_ui->p_loadoutList->count(); ++i) {
        auto *item = pu_ui->p_loadoutList->item(i);
        if (item->data(Qt::UserRole).toString() == d_activeLoadoutName) {
            QString display = d_activeLoadoutName + u" (active)"_s;
            if (dirty)
                display += u"*"_s;
            item->setText(display);
            break;
        }
    }
    updateLoadoutButtonStates();
}

void RuntimeHardwareConfigDialog::switchToLoadout(const QString &name)
{
    d_activeLoadoutName = name;
    auto loadout = LoadoutManager::instance().getLoadout(name);
    if (loadout.has_value() && !loadout->hardwareMap.empty()) {
        d_previewRuntimeConfig.clear();
        d_previewRuntimeConfig.insert(loadout->hardwareMap.begin(), loadout->hardwareMap.end());
    } else
        d_previewRuntimeConfig = d_originalRuntimeConfig;

    ensureRequiredTypes();

    d_currentHardwareType.clear();
    populateConfigurationOverview();
    populateHardwareBrowser();
    updateSelectionDisplay(QString{});
    validatePreviewConfiguration();
    populateLoadoutList();
}

void RuntimeHardwareConfigDialog::updateLoadoutButtonStates()
{
    const QString sel = selectedLoadoutName();
    const bool hasSel = !sel.isEmpty();
    const bool selIsActive = (sel == d_activeLoadoutName);
    pu_ui->p_loadoutActivate->setEnabled(hasSel && !selIsActive);
    pu_ui->p_loadoutSave->setEnabled(isPreviewDirty());
    pu_ui->p_loadoutSaveAs->setEnabled(true);
    pu_ui->p_loadoutCopy->setEnabled(hasSel);
    pu_ui->p_loadoutDelete->setEnabled(hasSel && !selIsActive);
}

void RuntimeHardwareConfigDialog::onLoadoutListSelectionChanged()
{
    updateLoadoutButtonStates();
}

void RuntimeHardwareConfigDialog::onLoadoutActivate()
{
    const QString name = selectedLoadoutName();
    if (name.isEmpty() || name == d_activeLoadoutName)
        return;

    if (isPreviewDirty()) {
        QMessageBox msgBox(this);
        msgBox.setWindowTitle(u"Unsaved Changes"_s);
        msgBox.setText(u"The active loadout \"%1\" has unsaved changes."_s.arg(d_activeLoadoutName));
        auto *saveBtn    = msgBox.addButton(u"Save and activate"_s,    QMessageBox::AcceptRole);
        auto *discardBtn = msgBox.addButton(u"Discard and activate"_s, QMessageBox::DestructiveRole);
        msgBox.addButton(QMessageBox::Cancel);
        msgBox.exec();
        const auto *clicked = msgBox.clickedButton();
        if (clicked == saveBtn)
            onLoadoutSave();
        else if (clicked != discardBtn)
            return;
    }

    switchToLoadout(name);
}

void RuntimeHardwareConfigDialog::onLoadoutSave()
{
    auto &lm = LoadoutManager::instance();

    HardwareLoadout loadout;
    loadout.name = d_activeLoadoutName;
    loadout.hardwareMap = std::map<QString,QString,std::less<>>(d_previewRuntimeConfig.begin(), d_previewRuntimeConfig.end());

    const auto existing = lm.getLoadout(d_activeLoadoutName);
    if (existing.has_value()) {
        const bool drift = ftmwRelevantHwKeys(loadout.hardwareMap) != ftmwRelevantHwKeys(existing->hardwareMap);

        if (!drift) {
            loadout.ftmwPresets = existing->ftmwPresets;
            loadout.currentFtmwPresetName = existing->currentFtmwPresetName;
        } else if (!lm.ftmwPresetNames(d_activeLoadoutName, false).isEmpty()) {
            QMessageBox msgBox(this);
            msgBox.setWindowTitle(u"Hardware Configuration Changed"_s);
            msgBox.setText(
                u"The AWG, digitizer, or clock hardware has changed for loadout \"%1\". "
                u"The existing FTMW presets may no longer be compatible."_s.arg(d_activeLoadoutName));
            auto *discardBtn = msgBox.addButton(u"Discard FTMW presets and save"_s, QMessageBox::DestructiveRole);
            auto *saveAsBtn  = msgBox.addButton(u"Save As instead"_s, QMessageBox::ResetRole);
            msgBox.addButton(QMessageBox::Cancel);
            msgBox.setDefaultButton(QMessageBox::Cancel);
            msgBox.exec();

            const auto *clicked = msgBox.clickedButton();
            if (clicked == discardBtn) {
                lm.clearFtmwPresets(d_activeLoadoutName);
            } else if (clicked == saveAsBtn) {
                onLoadoutSaveAs();
                return;
            } else {
                return;
            }
        } else {
            // Drift with no named presets: clear __LastUsed__ defensively
            lm.clearFtmwPresets(d_activeLoadoutName);
        }
    }

    loadout.lastModified = QDateTime::currentDateTimeUtc();
    lm.putLoadout(loadout);
    refreshLoadoutDirtyIndicator();
}

void RuntimeHardwareConfigDialog::onLoadoutSaveAs()
{
    bool ok = false;
    const QString name = QInputDialog::getText(
        this, u"Save Loadout As"_s, u"Loadout name:"_s,
        QLineEdit::Normal, {}, &ok).trimmed();

    if (!ok || name.isEmpty())
        return;

    if (LoadoutManager::instance().loadoutExists(name)) {
        const auto reply = QMessageBox::question(
            this, u"Overwrite Loadout"_s,
            u"A loadout named '%1' already exists. Overwrite it?"_s.arg(name),
            QMessageBox::Yes | QMessageBox::No);
        if (reply != QMessageBox::Yes)
            return;
    }

    const QString prevName = d_activeLoadoutName;

    HardwareLoadout loadout;
    loadout.name = name;
    loadout.hardwareMap = std::map<QString,QString,std::less<>>(d_previewRuntimeConfig.begin(), d_previewRuntimeConfig.end());
    loadout.lastModified = QDateTime::currentDateTimeUtc();
    LoadoutManager::instance().putLoadout(loadout);

    d_activeLoadoutName = name;
    populateLoadoutList();

    // Offer FTMW preset copy when the previous loadout shares hardware and has named presets
    auto &lm = LoadoutManager::instance();
    const auto prevLoadout = lm.getLoadout(prevName);
    if (prevLoadout.has_value()) {
        const auto namedPresets = lm.ftmwPresetNames(prevName, false);
        if (!namedPresets.isEmpty() &&
            ftmwRelevantHwKeys(loadout.hardwareMap) == ftmwRelevantHwKeys(prevLoadout->hardwareMap)) {
            const auto copyReply = QMessageBox::question(
                this, u"Copy FTMW Presets"_s,
                u"Copy FTMW presets from \"%1\" to \"%2\"?"_s.arg(prevName, name),
                QMessageBox::Yes | QMessageBox::No);
            if (copyReply == QMessageBox::Yes) {
                for (const auto &pName : namedPresets) {
                    auto preset = lm.getFtmwPreset(prevName, pName);
                    if (preset.has_value())
                        lm.putFtmwPreset(name, pName, *preset);
                }
                const auto curPreset = lm.currentFtmwPresetName(prevName);
                if (!curPreset.isEmpty()
                    && curPreset != BC::Store::LM::lastUsedFtmwPresetName)
                    lm.setCurrentFtmwPresetName(name, curPreset);
            }
        }
    }
}

void RuntimeHardwareConfigDialog::onLoadoutCopy()
{
    const QString sourceName = selectedLoadoutName();
    if (sourceName.isEmpty())
        return;

    const auto sourceLoadout = LoadoutManager::instance().getLoadout(sourceName);
    if (!sourceLoadout.has_value())
        return;

    bool ok = false;
    const QString name = QInputDialog::getText(
        this, u"Copy Loadout"_s, u"New loadout name:"_s,
        QLineEdit::Normal, {}, &ok).trimmed();
    if (!ok || name.isEmpty())
        return;

    if (LoadoutManager::instance().loadoutExists(name)) {
        const auto reply = QMessageBox::question(
            this, u"Overwrite Loadout"_s,
            u"A loadout named '%1' already exists. Overwrite it?"_s.arg(name),
            QMessageBox::Yes | QMessageBox::No);
        if (reply != QMessageBox::Yes)
            return;
    }

    HardwareLoadout newLoadout;
    newLoadout.name = name;
    newLoadout.hardwareMap = sourceLoadout->hardwareMap;
    LoadoutManager::instance().putLoadout(newLoadout);

    auto &lm = LoadoutManager::instance();
    const auto namedPresets = lm.ftmwPresetNames(sourceName, false);
    if (!namedPresets.isEmpty()) {
        const auto copyReply = QMessageBox::question(
            this, u"Copy FTMW Presets"_s,
            u"Copy FTMW presets from \"%1\" to \"%2\"?"_s.arg(sourceName, name),
            QMessageBox::Yes | QMessageBox::No);
        if (copyReply == QMessageBox::Yes) {
            for (const auto &pName : namedPresets) {
                auto preset = lm.getFtmwPreset(sourceName, pName);
                if (preset.has_value())
                    lm.putFtmwPreset(name, pName, *preset);
            }
            const auto curPreset = lm.currentFtmwPresetName(sourceName);
            if (!curPreset.isEmpty() && curPreset != BC::Store::LM::lastUsedFtmwPresetName)
                lm.setCurrentFtmwPresetName(name, curPreset);
        }
    }

    populateLoadoutList();
}

void RuntimeHardwareConfigDialog::onLoadoutDelete()
{
    const QString name = selectedLoadoutName();
    if (name.isEmpty() || name == d_activeLoadoutName)
        return;

    const auto reply = QMessageBox::question(
        this, u"Delete Loadout"_s,
        u"Delete loadout '%1'? This cannot be undone."_s.arg(name),
        QMessageBox::Yes | QMessageBox::No);
    if (reply != QMessageBox::Yes)
        return;

    LoadoutManager::instance().removeLoadout(name);
    populateLoadoutList();
}

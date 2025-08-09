#include "runtimehardwareconfigdialog.h"
#include "runtimehardwareconfigdialog_ui.h"
#include <QTreeWidgetItem>
#include <QListWidgetItem>
#include <QLabel>
#include <QVBoxLayout>
#include <QHash>
#include <data/bcglobals.h>
#include <hardware/core/hardwareregistry.h>

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
    
    // Connect dialog buttons
    connect(pu_ui->buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(pu_ui->buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    
    // Connect hardware browser selection changes
    connect(pu_ui->hardwareBrowserList, &QListWidget::currentItemChanged,
            this, &RuntimeHardwareConfigDialog::onHardwareBrowserSelectionChanged);
}

RuntimeHardwareConfigDialog::~RuntimeHardwareConfigDialog()
{
    delete pu_ui;
}

void RuntimeHardwareConfigDialog::populateConfigurationOverview()
{
    // Clear the tree
    pu_ui->configOverviewTree->clear();
    
    // Get all configured hardware
    auto currentHw = RuntimeHardwareConfig::constInstance().getCurrentHardware();
    
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
    
    // Get current hardware configuration to count active instances
    auto currentHw = RuntimeHardwareConfig::constInstance().getCurrentHardware();
    
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
        updateSelectionDisplay(QString());
        return;
    }
    
    // Extract hardware type from the selected item
    QString selectedHardwareType = current->data(Qt::UserRole).toString();
    updateSelectionDisplay(selectedHardwareType);
}

void RuntimeHardwareConfigDialog::updateSelectionDisplay(const QString& selectedHardwareType)
{
    // Clear the existing content in the right panel
    if (pu_ui->configurationContentWidget->layout()) {
        QLayoutItem* item;
        while ((item = pu_ui->configurationContentWidget->layout()->takeAt(0)) != nullptr) {
            delete item->widget();
            delete item;
        }
        delete pu_ui->configurationContentWidget->layout();
    }
    
    // Create a new layout for the right panel content
    auto* layout = new QVBoxLayout(pu_ui->configurationContentWidget);
    
    // Add placeholder label showing selected hardware type
    auto* selectionLabel = new QLabel(pu_ui->configurationContentWidget);
    selectionLabel->setAlignment(Qt::AlignCenter);
    
    if (selectedHardwareType.isEmpty()) {
        selectionLabel->setText("No hardware selected");
        selectionLabel->setStyleSheet("QLabel { color: gray; font-style: italic; }");
    } else {
        selectionLabel->setText(QString("Selected: %1").arg(selectedHardwareType));
        selectionLabel->setStyleSheet("QLabel { font-weight: bold; }");
    }
    
    layout->addWidget(selectionLabel);
    layout->addStretch(); // Add stretch to center the label
    
    pu_ui->configurationContentWidget->setLayout(layout);
}
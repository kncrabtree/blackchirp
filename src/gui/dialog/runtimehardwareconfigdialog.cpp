#include "runtimehardwareconfigdialog.h"
#include "runtimehardwareconfigdialog_ui.h"
#include <QTreeWidgetItem>
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
    
    // Connect dialog buttons
    connect(pu_ui->buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(pu_ui->buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
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
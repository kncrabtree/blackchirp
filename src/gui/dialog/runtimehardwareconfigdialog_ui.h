#ifndef RUNTIMEHARDWARECONFIGDIALOG_UI_H
#define RUNTIMEHARDWARECONFIGDIALOG_UI_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QDialogButtonBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QTreeWidgetItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QTableWidgetItem>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QHeaderView>
#include <gui/style/themecolors.h>

class Ui_RuntimeHardwareConfigDialog
{
public:
    QVBoxLayout *mainLayout;
    QTabWidget *mainTabWidget;
    
    // Hardware Configuration Tab
    QWidget *hardwareConfigTab;
    QVBoxLayout *hardwareConfigLayout;
    QSplitter *hardwareConfigSplitter;
    
    // Left Panel: Configuration Overview (33%)
    QWidget *configOverviewWidget;
    QVBoxLayout *configOverviewLayout;
    QLabel *configOverviewLabel;
    QTreeWidget *configOverviewTree;
    
    // Middle Panel: Hardware Browser (33%)
    QWidget *hardwareBrowserWidget;
    QVBoxLayout *hardwareBrowserLayout;
    QLabel *hardwareBrowserLabel;
    QListWidget *hardwareBrowserList;
    
    // Right Panel: Context-Sensitive Configuration (33%)
    QWidget *configurationWidget;
    QVBoxLayout *configurationLayout;
    QLabel *configurationLabel;
    QWidget *configurationContentWidget;
    
    // Validation Status Bar
    QLabel *validationStatusLabel;
    
    // Library Status Tab
    QWidget *libraryStatusTab;
    QVBoxLayout *libraryStatusLayout;
    QSplitter *libraryStatusSplitter;
    
    // Library Overview (Top)
    QWidget *libraryOverviewWidget;
    QVBoxLayout *libraryOverviewLayout;
    QLabel *libraryOverviewLabel;
    QTableWidget *libraryOverviewTable;
    
    // Library Details Label (now in left panel)
    QLabel *libraryDetailsPanelLabel;
    QTextEdit *libraryDetailsText;
    
    // Library Configuration Panel (Right)
    QWidget *libraryConfigPanel;
    QVBoxLayout *libraryConfigPanelLayout;
    QLabel *libraryConfigPanelLabel;
    
    // Configuration controls
    QGroupBox *libraryPathGroup;
    QFormLayout *libraryPathLayout;
    QLineEdit *userLibraryPathEdit;
    QPushButton *browseLibraryButton;
    QLineEdit *additionalPathsEdit;
    QCheckBox *autoDiscoveryCheckBox;
    QPushButton *testLoadButton;
    QPushButton *refreshLibraryButton;
    
    // Installation guidance
    QGroupBox *installationGuidanceGroup;
    QVBoxLayout *installationGuidanceLayout;
    QTextEdit *installationGuidanceText;
    
    // Dialog Buttons
    QDialogButtonBox *buttonBox;

    void setupUi(QDialog *RuntimeHardwareConfigDialog)
    {
        if (RuntimeHardwareConfigDialog->objectName().isEmpty())
            RuntimeHardwareConfigDialog->setObjectName(QString::fromUtf8("RuntimeHardwareConfigDialog"));
        RuntimeHardwareConfigDialog->setWindowTitle("Hardware Configuration");
        RuntimeHardwareConfigDialog->setModal(true);
        RuntimeHardwareConfigDialog->resize(1200, 650);
        
        // Main layout
        mainLayout = new QVBoxLayout(RuntimeHardwareConfigDialog);
        mainLayout->setSpacing(6);
        mainLayout->setContentsMargins(11, 11, 11, 11);
        mainLayout->setObjectName(QString::fromUtf8("mainLayout"));
        
        // Main tab widget
        mainTabWidget = new QTabWidget(RuntimeHardwareConfigDialog);
        mainTabWidget->setObjectName(QString::fromUtf8("mainTabWidget"));
        
        // Hardware Configuration Tab
        hardwareConfigTab = new QWidget();
        hardwareConfigTab->setObjectName(QString::fromUtf8("hardwareConfigTab"));
        hardwareConfigLayout = new QVBoxLayout(hardwareConfigTab);
        hardwareConfigLayout->setSpacing(6);
        hardwareConfigLayout->setContentsMargins(11, 11, 11, 11);
        hardwareConfigLayout->setObjectName(QString::fromUtf8("hardwareConfigLayout"));
        
        // Horizontal splitter for 3-panel layout
        hardwareConfigSplitter = new QSplitter(Qt::Horizontal, hardwareConfigTab);
        hardwareConfigSplitter->setObjectName(QString::fromUtf8("hardwareConfigSplitter"));
        
        // Left Panel: Configuration Overview (33%)
        configOverviewWidget = new QWidget();
        configOverviewWidget->setObjectName(QString::fromUtf8("configOverviewWidget"));
        configOverviewLayout = new QVBoxLayout(configOverviewWidget);
        configOverviewLayout->setSpacing(6);
        configOverviewLayout->setContentsMargins(6, 6, 6, 6);
        configOverviewLayout->setObjectName(QString::fromUtf8("configOverviewLayout"));
        
        configOverviewLabel = new QLabel(configOverviewWidget);
        configOverviewLabel->setObjectName(QString::fromUtf8("configOverviewLabel"));
        configOverviewLabel->setText("Configuration Overview");
        configOverviewLabel->setAlignment(Qt::AlignCenter);
        configOverviewLabel->setStyleSheet(QString::fromUtf8("QLabel { font-weight: bold; }"));
        configOverviewLayout->addWidget(configOverviewLabel);
        
        configOverviewTree = new QTreeWidget(configOverviewWidget);
        configOverviewTree->setObjectName(QString::fromUtf8("configOverviewTree"));
        configOverviewTree->setHeaderLabels(QStringList() << "Hardware");
        configOverviewTree->setRootIsDecorated(true);
        configOverviewTree->setSelectionMode(QAbstractItemView::SingleSelection);
        configOverviewTree->setContextMenuPolicy(Qt::CustomContextMenu);
        configOverviewTree->setSortingEnabled(true);
        configOverviewTree->sortByColumn(0, Qt::AscendingOrder);
        configOverviewLayout->addWidget(configOverviewTree);
        
        hardwareConfigSplitter->addWidget(configOverviewWidget);
        
        // Middle Panel: Hardware Browser (33%)
        hardwareBrowserWidget = new QWidget();
        hardwareBrowserWidget->setObjectName(QString::fromUtf8("hardwareBrowserWidget"));
        hardwareBrowserLayout = new QVBoxLayout(hardwareBrowserWidget);
        hardwareBrowserLayout->setSpacing(6);
        hardwareBrowserLayout->setContentsMargins(6, 6, 6, 6);
        hardwareBrowserLayout->setObjectName(QString::fromUtf8("hardwareBrowserLayout"));
        
        hardwareBrowserLabel = new QLabel(hardwareBrowserWidget);
        hardwareBrowserLabel->setObjectName(QString::fromUtf8("hardwareBrowserLabel"));
        hardwareBrowserLabel->setText("Hardware Browser");
        hardwareBrowserLabel->setAlignment(Qt::AlignCenter);
        hardwareBrowserLabel->setStyleSheet(QString::fromUtf8("QLabel { font-weight: bold; }"));
        hardwareBrowserLayout->addWidget(hardwareBrowserLabel);
        
        hardwareBrowserList = new QListWidget(hardwareBrowserWidget);
        hardwareBrowserList->setObjectName(QString::fromUtf8("hardwareBrowserList"));
        hardwareBrowserList->setSelectionMode(QAbstractItemView::SingleSelection);
        hardwareBrowserLayout->addWidget(hardwareBrowserList);
        
        hardwareConfigSplitter->addWidget(hardwareBrowserWidget);
        
        // Right Panel: Context-Sensitive Configuration (33%)
        configurationWidget = new QWidget();
        configurationWidget->setObjectName(QString::fromUtf8("configurationWidget"));
        configurationLayout = new QVBoxLayout(configurationWidget);
        configurationLayout->setSpacing(6);
        configurationLayout->setContentsMargins(6, 6, 6, 6);
        configurationLayout->setObjectName(QString::fromUtf8("configurationLayout"));
        
        configurationLabel = new QLabel(configurationWidget);
        configurationLabel->setObjectName(QString::fromUtf8("configurationLabel"));
        configurationLabel->setText("Configuration");
        configurationLabel->setAlignment(Qt::AlignCenter);
        configurationLabel->setStyleSheet(QString::fromUtf8("QLabel { font-weight: bold; }"));
        configurationLayout->addWidget(configurationLabel);
        
        configurationContentWidget = new QWidget(configurationWidget);
        configurationContentWidget->setObjectName(QString::fromUtf8("configurationContentWidget"));
        configurationLayout->addWidget(configurationContentWidget);
        
        hardwareConfigSplitter->addWidget(configurationWidget);
        
        // Set splitter sizes to 33% each
        hardwareConfigSplitter->setSizes({300, 300, 300});
        
        hardwareConfigLayout->addWidget(hardwareConfigSplitter, 1);
        
        // Validation Status Bar at bottom
        validationStatusLabel = new QLabel(hardwareConfigTab);
        validationStatusLabel->setObjectName(QString::fromUtf8("validationStatusLabel"));
        validationStatusLabel->setText("Configuration is valid");
        validationStatusLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        validationStatusLabel->setMargin(6);
        validationStatusLabel->setFrameStyle(QFrame::StyledPanel | QFrame::Sunken);
        hardwareConfigLayout->addWidget(validationStatusLabel, 0);
        
        mainTabWidget->addTab(hardwareConfigTab, "Hardware Configuration");
        
        // Library Status Tab
        libraryStatusTab = new QWidget();
        libraryStatusTab->setObjectName(QString::fromUtf8("libraryStatusTab"));
        libraryStatusLayout = new QVBoxLayout(libraryStatusTab);
        libraryStatusLayout->setSpacing(6);
        libraryStatusLayout->setContentsMargins(11, 11, 11, 11);
        libraryStatusLayout->setObjectName(QString::fromUtf8("libraryStatusLayout"));
        
        // Horizontal splitter for 2-panel layout: left (table + details) and right (configuration)
        libraryStatusSplitter = new QSplitter(Qt::Horizontal, libraryStatusTab);
        libraryStatusSplitter->setObjectName(QString::fromUtf8("libraryStatusSplitter"));
        
        // Left Panel: Library Overview + Details (stacked vertically)
        libraryOverviewWidget = new QWidget();
        libraryOverviewWidget->setObjectName(QString::fromUtf8("libraryOverviewWidget"));
        libraryOverviewLayout = new QVBoxLayout(libraryOverviewWidget);
        libraryOverviewLayout->setSpacing(6);
        libraryOverviewLayout->setContentsMargins(6, 6, 6, 6);
        libraryOverviewLayout->setObjectName(QString::fromUtf8("libraryOverviewLayout"));
        
        // Library Status Table
        libraryOverviewLabel = new QLabel(libraryOverviewWidget);
        libraryOverviewLabel->setObjectName(QString::fromUtf8("libraryOverviewLabel"));
        libraryOverviewLabel->setText("Vendor Library Status");
        libraryOverviewLabel->setAlignment(Qt::AlignCenter);
        libraryOverviewLabel->setStyleSheet(QString::fromUtf8("QLabel { font-weight: bold; }"));
        libraryOverviewLabel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        libraryOverviewLayout->addWidget(libraryOverviewLabel);
        
        libraryOverviewTable = new QTableWidget(libraryOverviewWidget);
        libraryOverviewTable->setObjectName(QString::fromUtf8("libraryOverviewTable"));
        libraryOverviewTable->setColumnCount(4);
        libraryOverviewTable->setHorizontalHeaderLabels(QStringList() << "Library" << "Status" << "Version" << "Load Path");
        libraryOverviewTable->setSelectionBehavior(QAbstractItemView::SelectRows);
        libraryOverviewTable->setSelectionMode(QAbstractItemView::SingleSelection);
        libraryOverviewTable->setAlternatingRowColors(true);
        libraryOverviewTable->horizontalHeader()->setStretchLastSection(true);
        libraryOverviewTable->verticalHeader()->setVisible(false);
        libraryOverviewLayout->addWidget(libraryOverviewTable, 1); // Give stretch to table
        
        // Library Details (below table in left panel)
        libraryDetailsPanelLabel = new QLabel(libraryOverviewWidget);
        libraryDetailsPanelLabel->setObjectName(QString::fromUtf8("libraryDetailsPanelLabel"));
        libraryDetailsPanelLabel->setText("Library Details");
        libraryDetailsPanelLabel->setAlignment(Qt::AlignCenter);
        libraryDetailsPanelLabel->setStyleSheet(QString::fromUtf8("QLabel { font-weight: bold; }"));
        libraryDetailsPanelLabel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        libraryOverviewLayout->addWidget(libraryDetailsPanelLabel);
        
        libraryDetailsText = new QTextEdit(libraryOverviewWidget);
        libraryDetailsText->setObjectName(QString::fromUtf8("libraryDetailsText"));
        libraryDetailsText->setReadOnly(true);
        libraryDetailsText->setMaximumHeight(200);
        libraryOverviewLayout->addWidget(libraryDetailsText, 0); // No stretch for details
        
        libraryStatusSplitter->addWidget(libraryOverviewWidget);
        
        // Right Panel: Library Configuration Controls
        libraryConfigPanel = new QWidget();
        libraryConfigPanel->setObjectName(QString::fromUtf8("libraryConfigPanel"));
        libraryConfigPanelLayout = new QVBoxLayout(libraryConfigPanel);
        libraryConfigPanelLayout->setSpacing(6);
        libraryConfigPanelLayout->setContentsMargins(6, 6, 6, 6);
        libraryConfigPanelLayout->setObjectName(QString::fromUtf8("libraryConfigPanelLayout"));
        
        libraryConfigPanelLabel = new QLabel(libraryConfigPanel);
        libraryConfigPanelLabel->setObjectName(QString::fromUtf8("libraryConfigPanelLabel"));
        libraryConfigPanelLabel->setText("Library Configuration");
        libraryConfigPanelLabel->setAlignment(Qt::AlignCenter);
        libraryConfigPanelLabel->setStyleSheet(QString::fromUtf8("QLabel { font-weight: bold; }"));
        libraryConfigPanelLabel->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        libraryConfigPanelLayout->addWidget(libraryConfigPanelLabel);
        
        // Library Path Configuration Group
        libraryPathGroup = new QGroupBox("Library Path Settings", libraryConfigPanel);
        libraryPathGroup->setObjectName(QString::fromUtf8("libraryPathGroup"));
        libraryPathLayout = new QFormLayout(libraryPathGroup);
        libraryPathLayout->setSpacing(6);
        libraryPathLayout->setContentsMargins(6, 6, 6, 6);
        libraryPathLayout->setObjectName(QString::fromUtf8("libraryPathLayout"));
        
        // User Library Path
        userLibraryPathEdit = new QLineEdit(libraryPathGroup);
        userLibraryPathEdit->setObjectName(QString::fromUtf8("userLibraryPathEdit"));
        userLibraryPathEdit->setPlaceholderText("Enter path to library file (optional)");
        
        browseLibraryButton = new QPushButton("Browse...", libraryPathGroup);
        browseLibraryButton->setObjectName(QString::fromUtf8("browseLibraryButton"));
        browseLibraryButton->setMaximumWidth(80);
        
        QHBoxLayout *pathLayout = new QHBoxLayout();
        pathLayout->addWidget(userLibraryPathEdit);
        pathLayout->addWidget(browseLibraryButton);
        libraryPathLayout->addRow("User Library Path:", pathLayout);
        
        // Additional Search Paths
        additionalPathsEdit = new QLineEdit(libraryPathGroup);
        additionalPathsEdit->setObjectName(QString::fromUtf8("additionalPathsEdit"));
        additionalPathsEdit->setPlaceholderText("Additional search paths (semicolon-separated)");
        libraryPathLayout->addRow("Additional Paths:", additionalPathsEdit);
        
        // Auto Discovery
        autoDiscoveryCheckBox = new QCheckBox("Enable automatic library discovery", libraryPathGroup);
        autoDiscoveryCheckBox->setObjectName(QString::fromUtf8("autoDiscoveryCheckBox"));
        autoDiscoveryCheckBox->setChecked(true);
        libraryPathLayout->addRow(autoDiscoveryCheckBox);
        
        // Action Buttons
        QHBoxLayout *buttonLayout = new QHBoxLayout();
        testLoadButton = new QPushButton("Test Load", libraryPathGroup);
        testLoadButton->setObjectName(QString::fromUtf8("testLoadButton"));
        buttonLayout->addWidget(testLoadButton);
        
        refreshLibraryButton = new QPushButton("Refresh Status", libraryPathGroup);
        refreshLibraryButton->setObjectName(QString::fromUtf8("refreshLibraryButton"));
        buttonLayout->addWidget(refreshLibraryButton);
        
        buttonLayout->addStretch();
        libraryPathLayout->addRow(buttonLayout);
        
        libraryConfigPanelLayout->addWidget(libraryPathGroup);
        
        // Installation Guidance Group
        installationGuidanceGroup = new QGroupBox("Installation Guidance", libraryConfigPanel);
        installationGuidanceGroup->setObjectName(QString::fromUtf8("installationGuidanceGroup"));
        installationGuidanceLayout = new QVBoxLayout(installationGuidanceGroup);
        installationGuidanceLayout->setSpacing(6);
        installationGuidanceLayout->setContentsMargins(6, 6, 6, 6);
        installationGuidanceLayout->setObjectName(QString::fromUtf8("installationGuidanceLayout"));
        
        installationGuidanceText = new QTextEdit(installationGuidanceGroup);
        installationGuidanceText->setObjectName(QString::fromUtf8("installationGuidanceText"));
        installationGuidanceText->setReadOnly(true);
        installationGuidanceText->setMaximumHeight(150);
        installationGuidanceText->setHtml(
            "<p><b>Installing Vendor Libraries:</b></p>"
            "<ul>"
            "<li><b>Spectrum Instrumentation:</b> Install the Spectrum M4i driver package from their website. The library is typically installed to <code>/opt/spectrum/lib/</code></li>"
            "<li><b>LabJack:</b> Install the LabJack Linux driver from their website. The USB library is typically installed to <code>/usr/local/lib/</code></li>"
            "</ul>"
            "<p>If libraries are installed to non-standard locations, use the configuration settings above to specify custom paths.</p>"
        );
        installationGuidanceLayout->addWidget(installationGuidanceText);
        
        libraryConfigPanelLayout->addWidget(installationGuidanceGroup);
        libraryConfigPanelLayout->addStretch();
        
        libraryStatusSplitter->addWidget(libraryConfigPanel);
        
        // Set splitter sizes (50% left panel, 50% right panel)
        libraryStatusSplitter->setSizes({400, 400});
        
        libraryStatusLayout->addWidget(libraryStatusSplitter);
        
        mainTabWidget->addTab(libraryStatusTab, "Library Status");
        
        mainLayout->addWidget(mainTabWidget);
        
        // Dialog button box
        buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, RuntimeHardwareConfigDialog);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->button(QDialogButtonBox::Ok)->setText("Apply Configuration");
        mainLayout->addWidget(buttonBox);
        
        // Phase 2: No placeholder content - will be populated from RuntimeHardwareConfig
        
        // Set initial tab
        mainTabWidget->setCurrentIndex(0);
        
        QMetaObject::connectSlotsByName(RuntimeHardwareConfigDialog);
    } // setupUi
    
private:
    // Placeholder content removed in Phase 2 - actual configuration populated from RuntimeHardwareConfig

    void retranslateUi(QDialog *RuntimeHardwareConfigDialog)
    {
        RuntimeHardwareConfigDialog->setWindowTitle("Hardware Configuration");
    } // retranslateUi

public:    
    void applyValidationStatusStyling(QWidget* parent = nullptr)
    {
        // Apply ThemeColors styling to validation status label (following UnifiedOverlayDialog pattern)
        validationStatusLabel->setStyleSheet(QString("QLabel { color: %1; }")
            .arg(ThemeColors::getCSSColor(ThemeColors::StatusSuccess, parent)));
    }
};

namespace Ui {
    class RuntimeHardwareConfigDialog: public Ui_RuntimeHardwareConfigDialog {};
} // namespace Ui

#endif // RUNTIMEHARDWARECONFIGDIALOG_UI_H
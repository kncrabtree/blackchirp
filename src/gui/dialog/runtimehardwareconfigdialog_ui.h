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
    
    // Library Status Tab (content provided by LibraryStatusWidget)
    QWidget *libraryStatusTab;
    
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
        
        // Library Status Tab — content is provided by LibraryStatusWidget,
        // which the dialog places into this tab.
        libraryStatusTab = new QWidget();
        libraryStatusTab->setObjectName(QString::fromUtf8("libraryStatusTab"));
        auto *libraryStatusLayout = new QVBoxLayout(libraryStatusTab);
        libraryStatusLayout->setSpacing(0);
        libraryStatusLayout->setContentsMargins(0, 0, 0, 0);

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

#include "librarystatuswidget.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QSplitter>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QTextEdit>
#include <QLineEdit>
#include <QCheckBox>
#include <QPushButton>
#include <QLabel>
#include <QDir>
#include <QFileDialog>
#include <QMessageBox>
#include <data/bcglobals.h>
#include <hardware/library/vendorlibrary.h>
#include <hardware/library/spectrumlibrary.h>
#include <hardware/library/labjacklibrary.h>
#include <gui/style/themecolors.h>

LibraryStatusWidget::LibraryStatusWidget(QWidget *parent)
    : QWidget{parent}, p_currentLibrary{nullptr}
{
    // Create widgets
    p_libraryOverviewTable = new QTableWidget(this);
    p_libraryOverviewTable->setColumnCount(4);
    p_libraryOverviewTable->setHorizontalHeaderLabels({"Library", "Status", "Version", "Load Path"});
    p_libraryOverviewTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    p_libraryOverviewTable->setAlternatingRowColors(true);
    p_libraryOverviewTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    p_libraryOverviewTable->setMaximumHeight(150);

    p_libraryDetailsText = new QTextEdit(this);
    p_libraryDetailsText->setReadOnly(true);

    p_userLibraryPathEdit = new QLineEdit(this);
    p_additionalPathsEdit = new QLineEdit(this);
    p_additionalPathsEdit->setPlaceholderText("Semicolon-separated...");
    p_autoDiscoveryCheckBox = new QCheckBox("Enable automatic library discovery", this);

    p_browseLibraryButton = new QPushButton("Browse...", this);
    p_testLoadButton = new QPushButton("Test Load", this);
    p_refreshLibraryButton = new QPushButton("Refresh", this);

    p_libraryConfigPanelLabel = new QLabel("Library Configuration", this);
    QFont boldFont = p_libraryConfigPanelLabel->font();
    boldFont.setBold(true);
    p_libraryConfigPanelLabel->setFont(boldFont);

    p_installationGuidanceText = new QTextEdit(this);
    p_installationGuidanceText->setReadOnly(true);

    // Build right-side panel layout
    auto *rightWidget = new QWidget(this);
    auto *rightLayout = new QVBoxLayout(rightWidget);

    rightLayout->addWidget(p_libraryConfigPanelLabel);

    auto *formLayout = new QFormLayout;
    auto *pathRowLayout = new QHBoxLayout;
    pathRowLayout->addWidget(p_userLibraryPathEdit);
    pathRowLayout->addWidget(p_browseLibraryButton);
    formLayout->addRow("Library Path:", pathRowLayout);
    formLayout->addRow("Additional Paths:", p_additionalPathsEdit);
    formLayout->addRow("Auto-Discovery:", p_autoDiscoveryCheckBox);
    rightLayout->addLayout(formLayout);

    auto *buttonRowLayout = new QHBoxLayout;
    buttonRowLayout->addWidget(p_testLoadButton);
    buttonRowLayout->addWidget(p_refreshLibraryButton);
    buttonRowLayout->addStretch();
    rightLayout->addLayout(buttonRowLayout);

    rightLayout->addWidget(p_installationGuidanceText);

    // Build splitter
    auto *splitter = new QSplitter(Qt::Horizontal, this);
    splitter->addWidget(p_libraryDetailsText);
    splitter->addWidget(rightWidget);

    // Build main layout
    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(p_libraryOverviewTable);
    mainLayout->addWidget(splitter);

    // Connect signals
    connect(p_libraryOverviewTable, &QTableWidget::currentItemChanged,
            this, &LibraryStatusWidget::onLibrarySelectionChanged);

    connect(p_userLibraryPathEdit, &QLineEdit::textChanged,
            this, &LibraryStatusWidget::onLibraryPathChanged);

    connect(p_additionalPathsEdit, &QLineEdit::textChanged,
            this, &LibraryStatusWidget::onLibraryPathChanged);

    connect(p_autoDiscoveryCheckBox, &QCheckBox::toggled,
            this, &LibraryStatusWidget::onLibraryPathChanged);

    connect(p_browseLibraryButton, &QPushButton::clicked,
            this, &LibraryStatusWidget::onBrowseLibraryPath);

    connect(p_testLoadButton, &QPushButton::clicked,
            this, &LibraryStatusWidget::onTestLoadLibrary);

    connect(p_refreshLibraryButton, &QPushButton::clicked,
            this, &LibraryStatusWidget::refreshLibraryStatus);

    // Initialize installation guidance with platform-specific generic instructions
    p_installationGuidanceText->setHtml(getGenericInstallationGuidance());

    // Initialize library status display
    refreshLibraryStatus();
}

bool LibraryStatusWidget::hasUnstagedChanges() const
{
    return SpectrumLibrary::instance().hasUnstagedChanges() ||
           LabjackLibrary::instance().hasUnstagedChanges();
}

void LibraryStatusWidget::revertAllChanges()
{
    // Revert staged changes for all vendor libraries
    SpectrumLibrary::instance().revertChanges();
    LabjackLibrary::instance().revertChanges();

    // Update UI to reflect reverted state if library is currently selected
    if (p_currentLibrary != nullptr) {
        updateLibraryConfiguration(*p_currentLibrary);
    }
}

void LibraryStatusWidget::refreshLibraryStatus()
{
    // Store current selection to restore later
    QTableWidgetItem *currentSelection = p_libraryOverviewTable->currentItem();
    QString selectedLibraryName;
    if (currentSelection != nullptr) {
        // Get the library name from the first column of the current row
        QTableWidgetItem *nameItem = p_libraryOverviewTable->item(currentSelection->row(), 0);
        if (nameItem != nullptr) {
            selectedLibraryName = nameItem->data(Qt::UserRole).toString();
        }
    }

    // Get references to all vendor libraries
    QList<QPair<QString, VendorLibrary *>> libraries;
    libraries.append({"Spectrum M4i", &SpectrumLibrary::instance()});
    libraries.append({"LabJack USB", &LabjackLibrary::instance()});

    // Initialize table structure if it's empty (first run)
    if (p_libraryOverviewTable->rowCount() == 0) {
        p_libraryOverviewTable->setRowCount(libraries.size());

        // Create table items once and set up basic properties
        for (int row = 0; row < libraries.size(); ++row) {
            const QString &displayName = libraries[row].first;

            // Library name (column 0)
            auto *nameItem = new QTableWidgetItem(displayName);
            nameItem->setFlags(nameItem->flags() & ~Qt::ItemIsEditable);
            nameItem->setData(Qt::UserRole, displayName);
            p_libraryOverviewTable->setItem(row, 0, nameItem);

            // Status (column 1)
            auto *statusItem = new QTableWidgetItem();
            statusItem->setFlags(statusItem->flags() & ~Qt::ItemIsEditable);
            p_libraryOverviewTable->setItem(row, 1, statusItem);

            // Version (column 2)
            auto *versionItem = new QTableWidgetItem();
            versionItem->setFlags(versionItem->flags() & ~Qt::ItemIsEditable);
            p_libraryOverviewTable->setItem(row, 2, versionItem);

            // Load path (column 3)
            auto *pathItem = new QTableWidgetItem();
            pathItem->setFlags(pathItem->flags() & ~Qt::ItemIsEditable);
            p_libraryOverviewTable->setItem(row, 3, pathItem);
        }

        // Set up column header stretch settings once
        p_libraryOverviewTable->horizontalHeader()->setStretchLastSection(true);
    }

    // Update existing table items with current library status
    QTableWidgetItem *itemToReselect = nullptr;

    for (int row = 0; row < libraries.size() && row < p_libraryOverviewTable->rowCount(); ++row) {
        const QString &displayName = libraries[row].first;
        VendorLibrary *library = libraries[row].second;

        // Update status (column 1)
        QString statusText = getLibraryStatusText(*library);
        QTableWidgetItem *statusItem = p_libraryOverviewTable->item(row, 1);
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
        QTableWidgetItem *versionItem = p_libraryOverviewTable->item(row, 2);
        if (versionItem != nullptr) {
            versionItem->setText(versionText);
        }

        // Update load path (column 3)
        QString pathText = library->loadedLibraryPath();
        if (pathText.isEmpty()) {
            pathText = "Not loaded";
        }
        QTableWidgetItem *pathItem = p_libraryOverviewTable->item(row, 3);
        if (pathItem != nullptr) {
            pathItem->setText(pathText);
        }

        // Check if this row should be reselected
        if (displayName == selectedLibraryName) {
            itemToReselect = p_libraryOverviewTable->item(row, 0); // Select first column
        }
    }

    // Restore selection if we had one previously
    if (itemToReselect != nullptr) {
        p_libraryOverviewTable->setCurrentItem(itemToReselect);
    }

    // Adjust column widths only if this was the initial setup
    if (selectedLibraryName.isEmpty()) {
        p_libraryOverviewTable->resizeColumnsToContents();
    }

    // Update details if a library is currently selected
    if (p_currentLibrary != nullptr) {
        updateLibraryDetails(*p_currentLibrary);
        updateLibraryConfiguration(*p_currentLibrary);
    }
}

void LibraryStatusWidget::onLibrarySelectionChanged(QTableWidgetItem *current, QTableWidgetItem *previous)
{
    Q_UNUSED(previous)

    if (current == nullptr) {
        p_currentLibrary = nullptr;
        d_currentLibraryKey.clear();
        // Clear details and configuration panels
        p_libraryDetailsText->clear();
        p_userLibraryPathEdit->clear();
        p_additionalPathsEdit->clear();
        p_autoDiscoveryCheckBox->setChecked(true);
        // Restore generic installation guidance
        p_installationGuidanceText->setHtml(getGenericInstallationGuidance());
        return;
    }

    // Get selected library
    int row = current->row();
    QString libraryDisplayName = p_libraryOverviewTable->item(row, 0)->data(Qt::UserRole).toString();

    VendorLibrary *library = nullptr;
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

void LibraryStatusWidget::onLibraryPathChanged()
{
    if (d_currentLibraryKey.isEmpty() || p_currentLibrary == nullptr) {
        return;
    }

    // Update library configuration based on UI input
    QString userPath = p_userLibraryPathEdit->text().trimmed();
    QString additionalPaths = p_additionalPathsEdit->text().trimmed();
    bool autoDiscovery = p_autoDiscoveryCheckBox->isChecked();

    // Apply settings to staged configuration (no immediate effect)
    p_currentLibrary->setStagedUserProvidedPath(userPath);

    // Parse semicolon-separated additional paths
    QStringList pathList;
    if (!additionalPaths.isEmpty()) {
        pathList = additionalPaths.split(';', Qt::SkipEmptyParts);
        for (QString &path : pathList) {
            path = path.trimmed();
        }
    }
    p_currentLibrary->setStagedSearchPaths(pathList);

    p_currentLibrary->setStagedAutoDiscoveryEnabled(autoDiscovery);

    // Update visual indicators for staged changes
    updateStagingIndicators();
    updateAllStagingIndicators();
}

void LibraryStatusWidget::onBrowseLibraryPath()
{
    if (p_currentLibrary == nullptr) {
        return;
    }

    QString currentPath = p_userLibraryPathEdit->text();
    QString startDir = currentPath.isEmpty() ? QDir::homePath() : currentPath;

    QString selectedPath = QFileDialog::getExistingDirectory(
        this,
        QString("Select Directory Containing %1 Library").arg(getLibraryDisplayName(*p_currentLibrary)),
        startDir
    );

    if (!selectedPath.isEmpty()) {
        p_userLibraryPathEdit->setText(selectedPath);
    }
}

void LibraryStatusWidget::onTestLoadLibrary()
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

void LibraryStatusWidget::updateLibraryDetails(VendorLibrary &library)
{
    QString details;

    // Library name and description
    details += QString("<h3>%1</h3>").arg(getLibraryDisplayName(library));
    details += QString("<p><b>Library Name:</b> %1</p>").arg(library.libraryName());

    // Status information
    details += QString("<p><b>Status:</b> %1</p>").arg(getLibraryStatusText(library));

    if (library.isAvailable()) {
        // Available - show success information
        details += QString("<p style='color: %1;'><b>&#10003; Library is available and ready</b></p>")
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
        details += QString("<p style='color: %1;'><b>&#10007; Library is not available</b></p>")
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
        for (const QString &name : platformNames) {
            details += QString("&bull; <code>%1</code><br>").arg(name);
        }
        details += "</p>";
    }

    // Default search paths
    QStringList defaultPaths = library.defaultSearchPaths();
    if (!defaultPaths.isEmpty()) {
        details += "<p><b>Default search paths:</b><br>";
        for (const QString &path : defaultPaths) {
            details += QString("&bull; <code>%1</code><br>").arg(path);
        }
        details += "</p>";
    }

    p_libraryDetailsText->setHtml(details);
}

void LibraryStatusWidget::updateLibraryConfiguration(VendorLibrary &library)
{
    // Block signals to prevent recursive calls
    p_userLibraryPathEdit->blockSignals(true);
    p_additionalPathsEdit->blockSignals(true);
    p_autoDiscoveryCheckBox->blockSignals(true);

    // Load staged settings (what user is editing, not active configuration)
    p_userLibraryPathEdit->setText(library.getStagedUserProvidedPath());

    QStringList userPaths = library.getStagedSearchPaths();
    p_additionalPathsEdit->setText(userPaths.join(";"));

    p_autoDiscoveryCheckBox->setChecked(library.isStagedAutoDiscoveryEnabled());

    // Re-enable signals
    p_userLibraryPathEdit->blockSignals(false);
    p_additionalPathsEdit->blockSignals(false);
    p_autoDiscoveryCheckBox->blockSignals(false);

    // Enable controls
    p_userLibraryPathEdit->setEnabled(true);
    p_browseLibraryButton->setEnabled(true);
    p_additionalPathsEdit->setEnabled(true);
    p_autoDiscoveryCheckBox->setEnabled(true);
    p_testLoadButton->setEnabled(true);

    // Update installation guidance with library-specific instructions
    p_installationGuidanceText->setHtml(library.getInstallationInstructions());

    // Update visual indicators for staging state
    updateStagingIndicators();
}

QString LibraryStatusWidget::getLibraryStatusText(VendorLibrary &library) const
{
    if (library.isAvailable()) {
        return "Available";
    } else if (library.wasLoadingAttempted()) {
        return "Error";
    } else {
        return "Not Found";
    }
}

QString LibraryStatusWidget::getLibraryDisplayName(VendorLibrary &library) const
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

QString LibraryStatusWidget::getLibraryVersion(VendorLibrary &library) const
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
        LabjackLibrary &ljLib = static_cast<LabjackLibrary &>(library);
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

void LibraryStatusWidget::updateStagingIndicators()
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
        p_libraryConfigPanelLabel->setStyleSheet(
            QString("QLabel { font-weight: bold; color: %1; }")
            .arg(ThemeColors::getCSSColor(ThemeColors::StatusInfo, this)));
    } else {
        p_libraryConfigPanelLabel->setStyleSheet("QLabel { font-weight: bold; }");
    }
    p_libraryConfigPanelLabel->setText(configLabel);

    // Update test load button text based on staging state
    if (hasChanges) {
        p_testLoadButton->setText("Test Load (Staged)");
        p_testLoadButton->setStyleSheet(
            QString("QPushButton { color: %1; font-weight: bold; }")
            .arg(ThemeColors::getCSSColor(ThemeColors::StatusInfo, this)));
    } else {
        p_testLoadButton->setText("Test Load");
        p_testLoadButton->setStyleSheet("");
    }

    // Update form controls to show modified state
    updateControlStagingIndicator(p_userLibraryPathEdit,
        p_currentLibrary->getStagedUserProvidedPath() != p_currentLibrary->getActiveUserProvidedPath());

    updateControlStagingIndicator(p_additionalPathsEdit,
        p_currentLibrary->getStagedSearchPaths() != p_currentLibrary->getActiveSearchPaths());

    updateControlStagingIndicator(p_autoDiscoveryCheckBox,
        p_currentLibrary->isStagedAutoDiscoveryEnabled() != p_currentLibrary->isActiveAutoDiscoveryEnabled());
}

void LibraryStatusWidget::updateControlStagingIndicator(QWidget *control, bool isModified)
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

void LibraryStatusWidget::updateAllStagingIndicators()
{
    // Update staging indicators for all libraries
    bool hasAnyChanges = SpectrumLibrary::instance().hasUnstagedChanges() ||
                         LabjackLibrary::instance().hasUnstagedChanges();

    // Notify parent that staging state has changed
    emit stagingStateChanged(hasAnyChanges);

    // Update current library's indicators if one is selected
    if (p_currentLibrary != nullptr) {
        updateStagingIndicators();
    }
}

QString LibraryStatusWidget::getGenericInstallationGuidance() const
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
        "<li><b>Security:</b> Allow installation from identified developers in System Preferences > Security &amp; Privacy</li>"
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
        "<li>Check System Preferences > Security &amp; Privacy for blocked extensions</li>"
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

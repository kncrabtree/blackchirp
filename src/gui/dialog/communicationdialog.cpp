#include <gui/dialog/communicationdialog.h>
#include <gui/widget/protocolwidget.h>
#include <gui/widget/rs232protocolwidget.h>
#include <gui/widget/tcpprotocolwidget.h>
#include <gui/widget/virtualprotocolwidget.h>
#include <gui/widget/customprotocolwidget.h>
#include <gui/widget/gpibprotocolwidget.h>
#include <gui/style/themecolors.h>
#include <data/settings/hardwarekeys.h> // Hardware discovery now uses RuntimeHardwareConfig; settings access still uses SettingsStorage

#include <QApplication>
#include <QMessageBox>
#include <QMetaEnum>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QSplitter>
#include <QListWidget>
#include <QListWidgetItem>
#include <QStackedWidget>
#include <QComboBox>
#include <QSpinBox>
#include <QLineEdit>
#include <QLabel>
#include <QPushButton>
#include <QGroupBox>
#include <QDialogButtonBox>

#include <hardware/core/communication/communicationprotocol.h>
#include <hardware/core/hardwaremanager.h>
#include <hardware/core/hardwareobject.h>
#include <data/storage/settingsstorage.h>

CommunicationDialog::CommunicationDialog(QWidget *parent) :
     QDialog(parent), p_hardwareManager(nullptr)
{
    setWindowTitle("Communication Settings");
    setWindowIcon(ThemeColors::createThemedIcon(":/icons/bc_logo_trans.svg", ThemeColors::IconPrimary, this));
    resize(800, 600);
    
    // Get HardwareManager instance
    p_hardwareManager = &const_cast<HardwareManager&>(HardwareManager::constInstance());
    
    setupUI();
    loadDeviceInfo();
    populateDeviceList();
    connectSignals();
    
    // Request initial GPIB controllers list
    QMetaObject::invokeMethod(p_hardwareManager, &HardwareManager::getActiveGpibControllers, Qt::QueuedConnection);
    
    // Select first device if available
    if (p_deviceList->count() > 0) {
        p_deviceList->setCurrentRow(0);
        onDeviceSelectionChanged();
    }
}

CommunicationDialog::~CommunicationDialog()
{
    // Clean up protocol widgets
    qDeleteAll(d_protocolWidgets);
}

void CommunicationDialog::setupUI()
{
    auto mainLayout = new QHBoxLayout(this);
    
    // Create splitter for master-detail layout
    auto splitter = new QSplitter(Qt::Horizontal, this);
    
    setupLeftPanel();
    setupRightPanel();
    
    splitter->addWidget(p_deviceList);
    splitter->addWidget(p_deviceConfigGroup);
    splitter->setStretchFactor(0, 1); // Device list gets 1/3
    splitter->setStretchFactor(1, 2); // Config panel gets 2/3
    
    mainLayout->addWidget(splitter);
    
    // Bottom buttons
    auto buttonLayout = new QVBoxLayout;
    buttonLayout->addWidget(splitter, 1);
    
    auto buttonBox = new QDialogButtonBox(this);
    p_testAllButton = new QPushButton("Test All Devices", this);
    p_testAllButton->setIcon(ThemeColors::createThemedIcon(":/icons/check-circle.svg", ThemeColors::IconPrimary, this));
    buttonBox->addButton(p_testAllButton, QDialogButtonBox::ActionRole);
    buttonBox->addButton(QDialogButtonBox::Close);
    
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    
    buttonLayout->addWidget(buttonBox);
    mainLayout->addLayout(buttonLayout);
}

void CommunicationDialog::setupLeftPanel()
{
    p_deviceList = new QListWidget(this);
    p_deviceList->setMinimumWidth(250);
    p_deviceList->setSelectionMode(QAbstractItemView::SingleSelection);
}

void CommunicationDialog::setupRightPanel()
{
    p_deviceConfigGroup = new QGroupBox("Device Configuration", this);
    p_deviceConfigGroup->setMinimumWidth(400);
    
    auto layout = new QVBoxLayout(p_deviceConfigGroup);
    
    // Device name header
    p_deviceNameLabel = new QLabel("No device selected", this);
    p_deviceNameLabel->setStyleSheet("font-weight: bold; font-size: 12pt;");
    layout->addWidget(p_deviceNameLabel);
    
    // Protocol selection
    auto protocolLayout = new QFormLayout;
    p_protocolCombo = new QComboBox(this);
    protocolLayout->addRow("Communication Protocol:", p_protocolCombo);
    
    // GPIB controller selection (initially hidden)
    p_gpibControllerCombo = new QComboBox(this);
    p_gpibControllerCombo->hide();
    protocolLayout->addRow("GPIB Controller:", p_gpibControllerCombo);
    
    layout->addLayout(protocolLayout);
    
    // Protocol-specific settings stack
    p_protocolStack = new QStackedWidget(this);
    layout->addWidget(p_protocolStack);
    
    // Common read options
    p_readOptionsGroup = new QGroupBox("Read Options", this);
    auto readLayout = new QFormLayout(p_readOptionsGroup);
    
    p_timeoutSpinBox = new QSpinBox(this);
    p_timeoutSpinBox->setRange(0, 60000);
    p_timeoutSpinBox->setSuffix(" ms");
    p_timeoutSpinBox->setValue(1000);
    p_timeoutSpinBox->setSpecialValueText("No timeout");
    readLayout->addRow("Timeout:", p_timeoutSpinBox);
    
    p_termCharEdit = new QLineEdit(this);
    p_termCharEdit->setPlaceholderText("Leave empty to disable");
    readLayout->addRow("Termination Character:", p_termCharEdit);
    
    layout->addWidget(p_readOptionsGroup);
    
    // Test button
    p_testDeviceButton = new QPushButton("Test Connection", this);
    p_testDeviceButton->setIcon(ThemeColors::createThemedIcon(":/icons/check-circle.svg", ThemeColors::IconPrimary, this));
    p_testDeviceButton->setEnabled(false);
    layout->addWidget(p_testDeviceButton);
    
    layout->addStretch();
}

void CommunicationDialog::connectSignals()
{
    connect(p_deviceList, &QListWidget::itemSelectionChanged, this, &CommunicationDialog::onDeviceSelectionChanged);
    connect(p_protocolCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &CommunicationDialog::onProtocolChanged);
    connect(p_testDeviceButton, &QPushButton::clicked, this, &CommunicationDialog::onTestDevice);
    connect(p_testAllButton, &QPushButton::clicked, this, &CommunicationDialog::onTestAllDevices);
    connect(p_timeoutSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &CommunicationDialog::onProtocolSettingsChanged);
    connect(p_termCharEdit, &QLineEdit::textChanged, this, &CommunicationDialog::onProtocolSettingsChanged);
    
    // HardwareManager signal connections
    connect(p_hardwareManager, &HardwareManager::gpibControllersAvailable,
            this, &CommunicationDialog::onGpibControllersAvailable);
    connect(p_hardwareManager, &HardwareManager::connectionResult,
            this, &CommunicationDialog::onConnectionResult);
}

void CommunicationDialog::loadDeviceInfo()
{
    // Get hardware discovery from RuntimeHardwareConfig instead of SettingsStorage
    const auto& config = RuntimeHardwareConfig::constInstance();
    auto currentHardware = config.getCurrentHardware();
    
    d_deviceInfo.clear();
    
    for(const auto& [hwKey, implementation] : currentHardware) {
        DeviceInfo info;
        info.hwKey = hwKey;               // hwKey is in "type.label" format
        info.subKey = implementation;     // implementation is the subKey (e.g., "mks647c")
        
        // Load display name from SettingsStorage (this is persistent configuration, not runtime state)
        SettingsStorage hwSettings(info.hwKey, SettingsStorage::Hardware);
        info.name = hwSettings.get(BC::Key::HW::name, info.hwKey); // Fall back to hwKey if name not found
        
        // Initialize with defaults - will be updated via HardwareManager signals
        info.currentProtocol = CommunicationProtocol::Virtual;
        info.supportedProtocols.clear();
        info.connected = false;
        info.tested = false;
        
        d_deviceInfo[info.hwKey] = info;
        
        // Request current communication info from HardwareManager
        QMetaObject::invokeMethod(p_hardwareManager, "getHardwareCommunicationInfo", 
                                 Qt::QueuedConnection,
                                 Q_ARG(QString, hwKey));
    }
}

void CommunicationDialog::populateDeviceList()
{
    p_deviceList->clear();
    
    for(auto it = d_deviceInfo.begin(); it != d_deviceInfo.end(); ++it) {
        const auto& info = it.value();
        
        auto item = new QListWidgetItem(getDeviceDisplayText(info));
        item->setIcon(getStatusIcon(info));
        item->setData(Qt::UserRole, info.hwKey);
        
        p_deviceList->addItem(item);
    }
}

QString CommunicationDialog::getDeviceDisplayText(const DeviceInfo& info)
{
    auto protocolEnum = QMetaEnum::fromType<CommunicationProtocol::CommType>();
    QString protocolName = protocolEnum.valueToKey(static_cast<int>(info.currentProtocol));
    
    return QString("%1 [%2]").arg(info.name).arg(protocolName);
}

QIcon CommunicationDialog::getStatusIcon(const DeviceInfo& info)
{
    if(!info.tested) {
        return ThemeColors::createThemedIcon(":/icons/help-circle.svg", ThemeColors::StatusWarning, this);
    } else if(info.connected) {
        return ThemeColors::createThemedIcon(":/icons/check-circle.svg", ThemeColors::StatusSuccess, this);
    } else {
        return ThemeColors::createThemedIcon(":/icons/x-circle.svg", ThemeColors::StatusError, this);
    }
}

void CommunicationDialog::onDeviceSelectionChanged()
{
    auto currentItem = p_deviceList->currentItem();
    if(!currentItem) {
        p_deviceConfigGroup->setEnabled(false);
        p_testDeviceButton->setEnabled(false);
        return;
    }
    
    d_currentDeviceKey = currentItem->data(Qt::UserRole).toString();
    updateRightPanel();
    
    p_deviceConfigGroup->setEnabled(true);
    p_testDeviceButton->setEnabled(true);
}

void CommunicationDialog::updateRightPanel()
{
    if(d_currentDeviceKey.isEmpty() || !d_deviceInfo.contains(d_currentDeviceKey)) {
        return;
    }
    
    const auto& info = d_deviceInfo[d_currentDeviceKey];
    
    // Update device name
    p_deviceNameLabel->setText(QString("Configuring: %1").arg(info.name));
    
    // Update protocol combo
    p_protocolCombo->blockSignals(true);
    p_protocolCombo->clear();
    
    auto protocolEnum = QMetaEnum::fromType<CommunicationProtocol::CommType>();
    int currentIndex = 0;
    
    for(int i = 0; i < info.supportedProtocols.size(); ++i) {
        auto protocol = info.supportedProtocols[i];
        QString protocolName = protocolEnum.valueToKey(static_cast<int>(protocol));
        p_protocolCombo->addItem(protocolName, static_cast<int>(protocol));
        
        if(protocol == info.currentProtocol) {
            currentIndex = i;
        }
    }
    
    p_protocolCombo->setCurrentIndex(currentIndex);
    p_protocolCombo->blockSignals(false);
    
    // Load current GPIB controller if this is a GPIB instrument
    if (info.currentProtocol == CommunicationProtocol::Gpib) {
        loadCurrentGpibController();
    }
    
    // Update protocol stack and read options
    onProtocolChanged();
}

void CommunicationDialog::onProtocolChanged()
{
    if(d_currentDeviceKey.isEmpty()) {
        return;
    }
    
    auto currentProtocol = static_cast<CommunicationProtocol::CommType>(
        p_protocolCombo->currentData().toInt()
    );
    
    
    // Show/hide GPIB controller selection based on protocol
    if(currentProtocol == CommunicationProtocol::Gpib) {
        p_gpibControllerCombo->show();
        p_gpibControllerCombo->parentWidget()->layout()->itemAt(1)->widget()->show(); // Show label
        // Load current GPIB controller when switching to GPIB protocol
        loadCurrentGpibController();
    } else {
        p_gpibControllerCombo->hide();
        p_gpibControllerCombo->parentWidget()->layout()->itemAt(1)->widget()->hide(); // Hide label
    }
    
    // Create unique key for device + protocol combination
    QString widgetKey = QString("%1:%2").arg(d_currentDeviceKey).arg(static_cast<int>(currentProtocol));
    
    // Create protocol widget if it doesn't exist
    if(!d_protocolWidgets.contains(widgetKey)) {
        ProtocolWidget* widget = nullptr;
        
        switch(currentProtocol) {
        case CommunicationProtocol::Rs232:
            widget = new Rs232ProtocolWidget(d_currentDeviceKey, this);
            break;
        case CommunicationProtocol::Tcp:
            widget = new TcpProtocolWidget(d_currentDeviceKey, this);
            break;
        case CommunicationProtocol::Virtual:
            widget = new VirtualProtocolWidget(d_currentDeviceKey, this);
            break;
        case CommunicationProtocol::Custom:
            widget = new CustomProtocolWidget(d_currentDeviceKey, this);
            break;
        case CommunicationProtocol::Gpib:
            widget = new GpibProtocolWidget(d_currentDeviceKey, this);
            break;
        default:
            break;
        }
        
        if(widget) {
            d_protocolWidgets[widgetKey] = widget;
            p_protocolStack->addWidget(widget);
            
            // Connect settings changed signal
            connect(widget, &ProtocolWidget::settingsChanged, this, &CommunicationDialog::onProtocolSettingsChanged);
        }
    }
    
    // Switch to the correct protocol widget
    auto currentWidget = d_protocolWidgets.value(widgetKey, nullptr);
    if(currentWidget) {
        p_protocolStack->setCurrentWidget(currentWidget);
        
        // Load settings into the widget
        currentWidget->loadProtocolSettings();
        
        // Load read options from settings and update UI
        loadReadOptions(currentProtocol);
    }
}


void CommunicationDialog::onTestDevice()
{
    if(d_currentDeviceKey.isEmpty()) {
        return;
    }
    
    // Read current protocol selection from UI
    CommunicationProtocol::CommType selectedProtocol = static_cast<CommunicationProtocol::CommType>(p_protocolCombo->currentData().toInt());
    
    // For GPIB protocol, get controller from UI
    QString gpibController;
    if (selectedProtocol == CommunicationProtocol::CommType::Gpib) {
        gpibController = p_gpibControllerCombo->currentData().toString();
        if (gpibController.isEmpty()) {
            QMessageBox::warning(this, "Missing GPIB Controller", 
                               "Please select a GPIB controller for GPIB communication.");
            return;
        }
    }
    
    // Apply protocol change via HardwareManager
    QMetaObject::invokeMethod(p_hardwareManager, &HardwareManager::setHardwareProtocol,
                             Qt::QueuedConnection,
                             d_currentDeviceKey, selectedProtocol, gpibController);
}

void CommunicationDialog::onTestAllDevices()
{
    for(auto it = d_deviceInfo.begin(); it != d_deviceInfo.end(); ++it) {
        QMetaObject::invokeMethod(p_hardwareManager, "testObjectConnection",
                                 Qt::QueuedConnection,
                                 Q_ARG(QString, it.key()));
    }
}

void CommunicationDialog::onProtocolSettingsChanged()
{
    // Settings will be saved when test is clicked or dialog is closed
}

void CommunicationDialog::saveDeviceSettings()
{
    if(d_currentDeviceKey.isEmpty()) {
        return;
    }
    
    auto currentProtocol = static_cast<CommunicationProtocol::CommType>(
        p_protocolCombo->currentData().toInt()
    );
    
    // Get current protocol widget and save through it
    QString widgetKey = QString("%1:%2").arg(d_currentDeviceKey).arg(static_cast<int>(currentProtocol));
    auto currentWidget = d_protocolWidgets.value(widgetKey, nullptr);
    if(currentWidget) {
        // Save through protocol widget (handles protocol type, read options, and protocol-specific settings)
        currentWidget->saveProtocolSettings(currentProtocol, p_timeoutSpinBox->value(), p_termCharEdit->text());
    }
    
    // Update local info
    d_deviceInfo[d_currentDeviceKey].currentProtocol = currentProtocol;
    updateDeviceListItem(d_currentDeviceKey);
}

void CommunicationDialog::loadReadOptions(CommunicationProtocol::CommType protocolType)
{
    if(d_currentDeviceKey.isEmpty()) {
        // Use defaults if no device selected
        p_timeoutSpinBox->setValue(1000);
        p_termCharEdit->clear();
        return;
    }
    
    // Create a temporary SettingsStorage to access the current device's settings
    SettingsStorage storage(d_currentDeviceKey, SettingsStorage::Hardware);
    
    // Get the protocol key for group access
    QString protocolKey;
    switch(protocolType) {
    case CommunicationProtocol::Rs232:
        protocolKey = BC::Key::Comm::rs232;
        break;
    case CommunicationProtocol::Tcp:
        protocolKey = BC::Key::Comm::tcp;
        break;
    case CommunicationProtocol::Gpib:
        protocolKey = BC::Key::Comm::gpib;
        break;
    case CommunicationProtocol::Custom:
        protocolKey = BC::Key::Comm::custom;
        break;
    case CommunicationProtocol::Virtual:
        protocolKey = BC::Key::Comm::hwVirtual;
        break;
    default:
        // Use defaults for None or unknown protocols
        p_timeoutSpinBox->setValue(1000);
        p_termCharEdit->clear();
        return;
    }
    
    // Load read options from group settings with sensible defaults
    int timeout = storage.getGroupValue<int>(protocolKey, BC::Key::Comm::timeout, 1000);
    QString termChar = storage.getGroupValue<QString>(protocolKey, BC::Key::Comm::termChar, QString());
    
    // Update UI controls
    p_timeoutSpinBox->setValue(timeout);
    p_termCharEdit->setText(termChar);
}

void CommunicationDialog::updateDeviceListItem(const QString& hwKey)
{
    if(!d_deviceInfo.contains(hwKey)) {
        return;
    }
    
    const auto& info = d_deviceInfo[hwKey];
    
    for(int i = 0; i < p_deviceList->count(); ++i) {
        auto item = p_deviceList->item(i);
        if(item->data(Qt::UserRole).toString() == hwKey) {
            item->setText(getDeviceDisplayText(info));
            item->setIcon(getStatusIcon(info));
            break;
        }
    }
}

void CommunicationDialog::onConnectionResult(const QString& hwKey, bool success, const QString& msg)
{
    // Find device by hwKey and update status
    for(auto it = d_deviceInfo.begin(); it != d_deviceInfo.end(); ++it) {
        if(it.key() == hwKey || it.value().name == hwKey) {
            it.value().tested = true;
            it.value().connected = success;
            updateDeviceListItem(it.key());
            break;
        }
    }
    
    if(!success && !msg.isEmpty()) {
        QMessageBox::warning(this, "Connection Test Failed", 
                           QString("Device: %1\nError: %2").arg(hwKey).arg(msg));
    }
}

void CommunicationDialog::loadCurrentGpibController()
{
    if (d_currentDeviceKey.isEmpty()) {
        return;
    }
    
    // Load the currently configured GPIB controller from settings
    SettingsStorage storage(d_currentDeviceKey, SettingsStorage::Hardware);
    QString currentController = storage.getGroupValue<QString>(BC::Key::Comm::gpib, BC::Key::GPIB::gpibController, QString());
    
    if (!currentController.isEmpty()) {
        // Find and select the current controller in the combo box
        int index = p_gpibControllerCombo->findData(currentController);
        if (index >= 0) {
            p_gpibControllerCombo->setCurrentIndex(index);
        }
    }
}

void CommunicationDialog::onGpibControllersAvailable(QStringList controllerKeys)
{
    // Populate GPIB controller combo box
    p_gpibControllerCombo->clear();
    p_gpibControllerCombo->addItem("Select Controller...", QString());
    
    for (const QString& controllerKey : controllerKeys) {
        p_gpibControllerCombo->addItem(controllerKey, controllerKey);
    }
    
    // If we have a current device selected and it uses GPIB, load its controller
    if (!d_currentDeviceKey.isEmpty() && d_deviceInfo.contains(d_currentDeviceKey)) {
        const auto& info = d_deviceInfo[d_currentDeviceKey];
        if (info.currentProtocol == CommunicationProtocol::Gpib) {
            loadCurrentGpibController();
        }
    }
}
#include <gui/widget/gpibprotocolwidget.h>
#include <hardware/core/communication/gpibinstrument.h>
#include <hardware/optional/gpibcontroller/gpibcontroller.h>
#include <hardware/core/hardwaremanager.h>
#include <hardware/core/runtimehardwareconfig.h>
#include <data/storage/settingsstorage.h>
#include <data/settings/hardwarekeys.h>

#include <QFormLayout>
#include <QComboBox>
#include <QSpinBox>

GpibProtocolWidget::GpibProtocolWidget(const QString& hwKey, QWidget *parent)
    : ProtocolWidget(hwKey, parent)
{
    setupUI();
    connectSignals();
    populateControllerList();
}

void GpibProtocolWidget::setupUI()
{
    p_layout = new QFormLayout(this);
    
    // GPIB Controller selection
    p_controllerCombo = new QComboBox(this);
    p_controllerCombo->setToolTip("Select GPIB controller for this device");
    p_layout->addRow("GPIB Controller:", p_controllerCombo);
    
    // GPIB Address
    p_addressSpinBox = new QSpinBox(this);
    p_addressSpinBox->setRange(0, 30); // Standard GPIB address range
    p_addressSpinBox->setValue(1); // Default address
    p_addressSpinBox->setToolTip("GPIB address for this device (0-30)");
    p_layout->addRow("GPIB Address:", p_addressSpinBox);
}

void GpibProtocolWidget::connectSignals()
{
    // Emit settingsChanged when any control changes
    connect(p_controllerCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &ProtocolWidget::settingsChanged);
    connect(p_addressSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), 
            this, &ProtocolWidget::settingsChanged);
}

void GpibProtocolWidget::populateControllerList()
{
    p_controllerCombo->clear();

    const auto& config = RuntimeHardwareConfig::constInstance();
    QStringList controllerKeys;
    for (const auto& [hwKey, impl] : config.getCurrentHardware()) {
        auto [hwType, label] = BC::Key::parseKey(hwKey);
        if (hwType == QString(GpibController::staticMetaObject.className()))
            controllerKeys.append(hwKey);
    }
    controllerKeys.sort();

    if (controllerKeys.isEmpty()) {
        p_controllerCombo->addItem("No GPIB controller configured", QString());
        return;
    }

    for (const auto& key : controllerKeys) {
        SettingsStorage s(key, SettingsStorage::Hardware);
        QString name = s.get(BC::Key::HW::name, key);
        p_controllerCombo->addItem(name, key);
    }
}

void GpibProtocolWidget::loadProtocolSettings()
{
    using namespace BC::Key::GPIB;
    
    // Load GPIB-specific settings using group-based storage with fallback
    QString controllerKey = getGroupValue(BC::Key::Comm::gpib, gpibController, 
                                         get(gpibController, QString()));
    int address = getGroupValue<int>(BC::Key::Comm::gpib, gpibAddress, 
                                    get<int>(gpibAddress, 1));
    
    // Update controller selection
    if(!controllerKey.isEmpty()) {
        int controllerIndex = p_controllerCombo->findData(controllerKey);
        if(controllerIndex >= 0) {
            p_controllerCombo->setCurrentIndex(controllerIndex);
        }
        // If controller not found, leave current selection (may be placeholder)
    }
    
    // Update address (with bounds checking)
    if(address >= 0 && address <= 30) {
        p_addressSpinBox->setValue(address);
    }
}

QString GpibProtocolWidget::selectedController() const
{
    return p_controllerCombo->currentData().toString();
}

void GpibProtocolWidget::saveProtocolSpecificSettings()
{
    using namespace BC::Key::GPIB;
    
    // Save GPIB-specific settings using group-based storage
    QString controllerKey = p_controllerCombo->currentData().toString();
    int address = p_addressSpinBox->value();
    
    setGroupValue(BC::Key::Comm::gpib, gpibController, controllerKey);
    setGroupValue(BC::Key::Comm::gpib, gpibAddress, address);
}
#include <gui/widget/gpibprotocolwidget.h>
#include <hardware/core/communication/gpibinstrument.h>
#include <hardware/optional/gpibcontroller/gpibcontroller.h>
#include <hardware/core/hardwaremanager.h>
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
    
    // Find all available GPIB controllers from hardware settings
    // Future-proofed for multiple controllers
    SettingsStorage hwStorage(BC::Key::hw);
    auto allHwCount = hwStorage.getArraySize(BC::Key::allHw);
    
    bool foundController = false;
    for(std::size_t i = 0; i < allHwCount; ++i) {
        QString hwKey = hwStorage.getArrayValue<QString>(BC::Key::allHw, i, BC::Key::HW::key);
        if(hwKey.startsWith(QString(GpibController::staticMetaObject.className()))) {
            QString hwName = hwStorage.getArrayValue<QString>(BC::Key::allHw, i, BC::Key::HW::name);
            p_controllerCombo->addItem(hwName, hwKey);
            foundController = true;
        }
    }
    
    // If no controllers found, add placeholder
    if(!foundController) {
        p_controllerCombo->addItem("No GPIB controller configured", QString());
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

void GpibProtocolWidget::saveProtocolSpecificSettings()
{
    using namespace BC::Key::GPIB;
    
    // Save GPIB-specific settings using group-based storage
    QString controllerKey = p_controllerCombo->currentData().toString();
    int address = p_addressSpinBox->value();
    
    setGroupValue(BC::Key::Comm::gpib, gpibController, controllerKey);
    setGroupValue(BC::Key::Comm::gpib, gpibAddress, address);
}
#include <gui/widget/rs232protocolwidget.h>
#include <hardware/core/communication/rs232instrument.h>
#include <data/settings/hardwarekeys.h>

#include <QFormLayout>
#include <QComboBox>
#include <QLineEdit>
#include <QMetaEnum>

Rs232ProtocolWidget::Rs232ProtocolWidget(const QString& hwKey, QWidget *parent)
    : ProtocolWidget(hwKey, parent)
{
    setupUI();
    connectSignals();
}

void Rs232ProtocolWidget::setupUI()
{
    p_layout = new QFormLayout(this);
    
    // Device ID
    p_deviceIdEdit = new QLineEdit(this);
    p_deviceIdEdit->setPlaceholderText("e.g., /dev/ttyUSB0 or COM3");
    p_layout->addRow("Device ID:", p_deviceIdEdit);
    
    // Baud Rate
    p_baudRateCombo = new QComboBox(this);
    p_baudRateCombo->addItems({
        "1200", "2400", "4800", "9600", "19200", "38400", 
        "57600", "115200", "230400", "460800", "921600"
    });
    p_baudRateCombo->setCurrentText("57600"); // Default
    p_layout->addRow("Baud Rate:", p_baudRateCombo);
    
    // Data Bits
    p_dataBitsCombo = new QComboBox(this);
    auto dataBitsEnum = QMetaEnum::fromType<Rs232Instrument::DataBits>();
    for(int i = 0; i < dataBitsEnum.keyCount(); ++i) {
        auto key = dataBitsEnum.key(i);
        auto value = dataBitsEnum.value(i);
        p_dataBitsCombo->addItem(key, value);
    }
    p_dataBitsCombo->setCurrentText("Data8"); // Default
    p_layout->addRow("Data Bits:", p_dataBitsCombo);
    
    // Parity
    p_parityCombo = new QComboBox(this);
    auto parityEnum = QMetaEnum::fromType<Rs232Instrument::Parity>();
    for(int i = 0; i < parityEnum.keyCount(); ++i) {
        auto key = parityEnum.key(i);
        auto value = parityEnum.value(i);
        p_parityCombo->addItem(key, value);
    }
    p_parityCombo->setCurrentText("NoParity"); // Default
    p_layout->addRow("Parity:", p_parityCombo);
    
    // Stop Bits
    p_stopBitsCombo = new QComboBox(this);
    auto stopBitsEnum = QMetaEnum::fromType<Rs232Instrument::StopBits>();
    for(int i = 0; i < stopBitsEnum.keyCount(); ++i) {
        auto key = stopBitsEnum.key(i);
        auto value = stopBitsEnum.value(i);
        p_stopBitsCombo->addItem(key, value);
    }
    p_stopBitsCombo->setCurrentText("OneStop"); // Default
    p_layout->addRow("Stop Bits:", p_stopBitsCombo);
    
    // Flow Control
    p_flowControlCombo = new QComboBox(this);
    auto flowControlEnum = QMetaEnum::fromType<Rs232Instrument::FlowControl>();
    for(int i = 0; i < flowControlEnum.keyCount(); ++i) {
        auto key = flowControlEnum.key(i);
        auto value = flowControlEnum.value(i);
        p_flowControlCombo->addItem(key, value);
    }
    p_flowControlCombo->setCurrentText("NoFlowControl"); // Default
    p_layout->addRow("Flow Control:", p_flowControlCombo);
}

void Rs232ProtocolWidget::connectSignals()
{
    // Emit settingsChanged when any control changes
    connect(p_deviceIdEdit, &QLineEdit::textChanged, this, &ProtocolWidget::settingsChanged);
    connect(p_baudRateCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &ProtocolWidget::settingsChanged);
    connect(p_dataBitsCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &ProtocolWidget::settingsChanged);
    connect(p_parityCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &ProtocolWidget::settingsChanged);
    connect(p_stopBitsCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &ProtocolWidget::settingsChanged);
    connect(p_flowControlCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &ProtocolWidget::settingsChanged);
}

void Rs232ProtocolWidget::loadProtocolSettings()
{
    using namespace BC::Key::RS232;
    
    // Load RS232-specific settings using group-based storage with backward compatibility
    auto deviceId = getGroupValue(BC::Key::Comm::rs232, id, get(id, QString("")));
    auto baudRate = getGroupValue<qint32>(BC::Key::Comm::rs232, baud, get<qint32>(baud, 57600));
    auto databitsSetting = static_cast<Rs232Instrument::DataBits>(getGroupValue<int>(BC::Key::Comm::rs232, dataBits, 
                                                                                     get<int>(dataBits, static_cast<int>(Rs232Instrument::Data8))));
    auto paritySetting = static_cast<Rs232Instrument::Parity>(getGroupValue<int>(BC::Key::Comm::rs232, parity, 
                                                                                  get<int>(parity, static_cast<int>(Rs232Instrument::NoParity))));
    auto stopbitsSetting = static_cast<Rs232Instrument::StopBits>(getGroupValue<int>(BC::Key::Comm::rs232, stopBits, 
                                                                                      get<int>(stopBits, static_cast<int>(Rs232Instrument::OneStop))));
    auto flowcontrolSetting = static_cast<Rs232Instrument::FlowControl>(getGroupValue<int>(BC::Key::Comm::rs232, flowControl, 
                                                                                            get<int>(flowControl, static_cast<int>(Rs232Instrument::NoFlowControl))));
    
    // Update UI controls
    p_deviceIdEdit->setText(deviceId);
    p_baudRateCombo->setCurrentText(QString::number(baudRate));
    
    // Set enum combo boxes by finding matching data values
    for(int i = 0; i < p_dataBitsCombo->count(); ++i) {
        if(p_dataBitsCombo->itemData(i).toInt() == static_cast<int>(databitsSetting)) {
            p_dataBitsCombo->setCurrentIndex(i);
            break;
        }
    }
    
    for(int i = 0; i < p_parityCombo->count(); ++i) {
        if(p_parityCombo->itemData(i).toInt() == static_cast<int>(paritySetting)) {
            p_parityCombo->setCurrentIndex(i);
            break;
        }
    }
    
    for(int i = 0; i < p_stopBitsCombo->count(); ++i) {
        if(p_stopBitsCombo->itemData(i).toInt() == static_cast<int>(stopbitsSetting)) {
            p_stopBitsCombo->setCurrentIndex(i);
            break;
        }
    }
    
    for(int i = 0; i < p_flowControlCombo->count(); ++i) {
        if(p_flowControlCombo->itemData(i).toInt() == static_cast<int>(flowcontrolSetting)) {
            p_flowControlCombo->setCurrentIndex(i);
            break;
        }
    }
}

void Rs232ProtocolWidget::saveProtocolSpecificSettings()
{
    using namespace BC::Key::RS232;
    
    // Save RS232-specific settings using group-based storage
    // Base class will handle final save() call
    setGroupValue(BC::Key::Comm::rs232, id, p_deviceIdEdit->text());
    setGroupValue(BC::Key::Comm::rs232, baud, p_baudRateCombo->currentText().toInt());
    setGroupValue(BC::Key::Comm::rs232, dataBits, p_dataBitsCombo->currentData().toInt());
    setGroupValue(BC::Key::Comm::rs232, parity, p_parityCombo->currentData().toInt());
    setGroupValue(BC::Key::Comm::rs232, stopBits, p_stopBitsCombo->currentData().toInt());
    setGroupValue(BC::Key::Comm::rs232, flowControl, p_flowControlCombo->currentData().toInt());
}
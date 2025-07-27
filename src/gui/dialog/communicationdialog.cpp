#include <gui/dialog/communicationdialog.h>
#include "ui_communicationdialog.h"
#include <gui/style/themecolors.h>

#include <QApplication>
#include <QMessageBox>
#include <QMetaEnum>
#include <QSpinBox>
#include <QLineEdit>
#include <QFormLayout>

#include <hardware/core/communication/communicationprotocol.h>
#include <hardware/core/hardwaremanager.h>
#include <hardware/core/hardwareobject.h>

CommunicationDialog::CommunicationDialog(QWidget *parent) :
     QDialog(parent),
     ui(new Ui::CommunicationDialog), d_storage(BC::Key::hw)
{
	ui->setupUi(this);
	
	// Set BlackChirp branding
	setWindowIcon(ThemeColors::createThemedIcon(":/icons/bc_logo_trans.svg", ThemeColors::IconPrimary, this));
	
	// Override test button icons with theme-aware versions
	ui->gpibTestButton->setIcon(ThemeColors::createThemedIcon(":/icons/check-circle.svg", ThemeColors::IconPrimary, this));
	ui->tcpTestButton->setIcon(ThemeColors::createThemedIcon(":/icons/check-circle.svg", ThemeColors::IconPrimary, this));
	ui->rs232TestButton->setIcon(ThemeColors::createThemedIcon(":/icons/check-circle.svg", ThemeColors::IconPrimary, this));
	ui->customTestButton->setIcon(ThemeColors::createThemedIcon(":/icons/check-circle.svg", ThemeColors::IconPrimary, this));

	// Populate devices by reading allHw array and grouping by current protocol
    populateDevicesByProtocol();

    ui->gpibDeviceComboBox->setCurrentIndex(-1);
	ui->gpibTestButton->setEnabled(false);
	ui->busAddressSpinBox->setEnabled(false);

	ui->tcpDeviceComboBox->setCurrentIndex(-1);
	ui->ipLineEdit->setEnabled(false);
	ui->portSpinBox->setEnabled(false);
	ui->tcpTestButton->setEnabled(false);

	ui->rs232DeviceComboBox->setCurrentIndex(-1);
	ui->rs232DeviceIDLineEdit->setEnabled(false);
	ui->baudRateComboBox->setCurrentIndex(-1);
	ui->baudRateComboBox->setEnabled(false);

    ui->dataBitsComboBox->setCurrentIndex(-1);
    auto db = QMetaEnum::fromType<Rs232Instrument::DataBits>();
    for(int i=0; i<db.keyCount(); ++i)
        ui->dataBitsComboBox->addItem(db.key(i),static_cast<Rs232Instrument::DataBits>(db.value(i)));
    ui->dataBitsComboBox->setEnabled(false);

    ui->stopBitsComboBox->setCurrentIndex(-1);
    db = QMetaEnum::fromType<Rs232Instrument::StopBits>();
    for(int i=0; i<db.keyCount(); ++i)
        ui->stopBitsComboBox->addItem(db.key(i),static_cast<Rs232Instrument::StopBits>(db.value(i)));
    ui->stopBitsComboBox->setEnabled(false);

    ui->parityComboBox->setCurrentIndex(-1);
    db = QMetaEnum::fromType<Rs232Instrument::Parity>();
    for(int i=0; i<db.keyCount(); ++i)
        ui->parityComboBox->addItem(db.key(i),static_cast<Rs232Instrument::Parity>(db.value(i)));
    ui->parityComboBox->setEnabled(false);

    ui->flowControlComboBox->setCurrentIndex(-1);
    db = QMetaEnum::fromType<Rs232Instrument::FlowControl>();
    for(int i=0; i<db.keyCount(); ++i)
        ui->flowControlComboBox->addItem(db.key(i),static_cast<Rs232Instrument::FlowControl>(db.value(i)));
    ui->flowControlComboBox->setEnabled(false);

	ui->rs232TestButton->setEnabled(false);

    ui->customDeviceComboBox->setCurrentIndex(-1);
    ui->customTestButton->setEnabled(false);

    // Setup read options UI for each protocol
    setupReadOptionsUI();
    loadReadOptionsFromSettings();

    auto cbChanged = static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged);

    connect(ui->gpibDeviceComboBox,cbChanged,this,&CommunicationDialog::gpibDeviceChanged);
    connect(ui->tcpDeviceComboBox,cbChanged,this,&CommunicationDialog::tcpDeviceChanged);
    connect(ui->rs232DeviceComboBox,cbChanged,this,&CommunicationDialog::rs232DeviceChanged);
    connect(ui->customDeviceComboBox,cbChanged,this,&CommunicationDialog::customDeviceChanged);

    connect(ui->gpibTestButton,&QPushButton::clicked,this,&CommunicationDialog::testGpib);
    connect(ui->tcpTestButton,&QPushButton::clicked,this,&CommunicationDialog::testTcp);
    connect(ui->rs232TestButton,&QPushButton::clicked,this,&CommunicationDialog::testRs232);
    connect(ui->customTestButton,&QPushButton::clicked,this,&CommunicationDialog::testCustom);
}

CommunicationDialog::~CommunicationDialog()
{
	delete ui;
}

void CommunicationDialog::startTest(QString type, QString key)
{
	//configure UI
	setEnabled(false);
	setCursor(QCursor(Qt::BusyCursor));

	emit testConnection(type,key);
}

void CommunicationDialog::populateDevicesByProtocol()
{
    // Read all hardware objects and group by current communication protocol
    auto allHwCount = d_storage.getArraySize(BC::Key::allHw);
    
    QVector<QPair<QString, QString>> gpibDevices, tcpDevices, rs232Devices, customDevices;
    
    for(std::size_t i = 0; i < allHwCount; ++i) {
        auto hwKey = d_storage.getArrayValue<QString>(BC::Key::allHw, i, BC::Key::HW::key);
        auto hwSubKey = d_storage.getArrayValue<QString>(BC::Key::allHw, i, BC::Key::HW::subKey);
        auto hwName = d_storage.getArrayValue<QString>(BC::Key::allHw, i, BC::Key::HW::name);
        
        // Read the current communication protocol from the hardware's settings
        SettingsStorage hwSettings(hwKey, SettingsStorage::Hardware);
        auto currentProtocol = static_cast<CommunicationProtocol::CommType>(
            hwSettings.get(BC::Key::HW::commType, static_cast<int>(CommunicationProtocol::Virtual))
        );
        
        // Group by protocol type
        QPair<QString, QString> deviceInfo{hwKey, hwName};
        switch(currentProtocol) {
        case CommunicationProtocol::Gpib:
            gpibDevices.append(deviceInfo);
            break;
        case CommunicationProtocol::Tcp:
            tcpDevices.append(deviceInfo);
            break;
        case CommunicationProtocol::Rs232:
            rs232Devices.append(deviceInfo);
            break;
        case CommunicationProtocol::Custom:
            customDevices.append(deviceInfo);
            break;
        default:
            // Skip Virtual and None protocols
            break;
        }
    }
    
    // Populate combo boxes
    for(const auto& device : gpibDevices) {
        ui->gpibDeviceComboBox->addItem(device.second, device.first);
    }
    ui->gpibBox->setEnabled(!gpibDevices.isEmpty());
    
    for(const auto& device : tcpDevices) {
        ui->tcpDeviceComboBox->addItem(device.second, device.first);
    }
    ui->tcpBox->setEnabled(!tcpDevices.isEmpty());
    
    for(const auto& device : rs232Devices) {
        ui->rs232DeviceComboBox->addItem(device.second, device.first);
    }
    ui->rs232Box->setEnabled(!rs232Devices.isEmpty());
    
    for(const auto& device : customDevices) {
        ui->customDeviceComboBox->addItem(device.second, device.first);
    }
    ui->customBox->setEnabled(!customDevices.isEmpty());
}

void CommunicationDialog::gpibDeviceChanged(int index)
{
	if(index < 0)
	{
		ui->busAddressSpinBox->setValue(0);
		ui->gpibTestButton->setEnabled(false);
		ui->busAddressSpinBox->setEnabled(false);
		return;
	}

    auto key = ui->gpibDeviceComboBox->currentData().toString();

    SettingsStorage s(key,SettingsStorage::Hardware);

	ui->busAddressSpinBox->setEnabled(true);
    ui->busAddressSpinBox->setValue(s.get<int>(BC::Key::GPIB::gpibAddress,0));
	ui->gpibTestButton->setEnabled(true);

}

void CommunicationDialog::tcpDeviceChanged(int index)
{
	if(index < 0)
	{
		ui->ipLineEdit->clear();
		ui->ipLineEdit->setEnabled(false);
		ui->portSpinBox->setValue(0);
		ui->portSpinBox->setEnabled(false);
		ui->tcpTestButton->setEnabled(false);
		return;
	}

    auto key = ui->tcpDeviceComboBox->currentData().toString();

    SettingsStorage s(key,SettingsStorage::Hardware);
	ui->ipLineEdit->setEnabled(true);
    ui->ipLineEdit->setText(s.get<QString>(BC::Key::TCP::ip,""));
	ui->portSpinBox->setEnabled(true);
    ui->portSpinBox->setValue(s.get<int>(BC::Key::TCP::port,0));
	ui->tcpTestButton->setEnabled(true);
}

void CommunicationDialog::rs232DeviceChanged(int index)
{
	if(index < 0)
	{
		ui->rs232DeviceIDLineEdit->clear();
		ui->rs232DeviceIDLineEdit->setEnabled(false);
		ui->baudRateComboBox->setCurrentIndex(-1);
		ui->baudRateComboBox->setEnabled(false);
        ui->dataBitsComboBox->setCurrentIndex(-1);
        ui->dataBitsComboBox->setEnabled(false);
        ui->stopBitsComboBox->setCurrentIndex(-1);
        ui->stopBitsComboBox->setEnabled(false);
        ui->parityComboBox->setCurrentIndex(-1);
        ui->parityComboBox->setEnabled(false);
        ui->flowControlComboBox->setCurrentIndex(-1);
        ui->flowControlComboBox->setEnabled(false);
		ui->rs232TestButton->setEnabled(false);
        return;
	}

    auto key = ui->rs232DeviceComboBox->currentData().toString();

    SettingsStorage s(key,SettingsStorage::Hardware);
	ui->rs232DeviceIDLineEdit->setEnabled(true);
    ui->rs232DeviceIDLineEdit->setText(s.get<QString>(BC::Key::RS232::id,""));

    auto br = s.get<qint32>(BC::Key::RS232::baud,-1);
	ui->baudRateComboBox->setEnabled(true);
	ui->baudRateComboBox->setCurrentIndex(-1);
	for(int i=0;i<ui->baudRateComboBox->count();i++)
	{
        if(br == static_cast<qint32>(ui->baudRateComboBox->itemText(i).toInt()))
			ui->baudRateComboBox->setCurrentIndex(i);
	}

    auto idx = ui->dataBitsComboBox->findData(s.get(BC::Key::RS232::dataBits,
                                                    QVariant::fromValue(Rs232Instrument::Data8)));
    ui->dataBitsComboBox->setCurrentIndex(qBound(0,idx,ui->dataBitsComboBox->count()-1));
    ui->dataBitsComboBox->setEnabled(true);

    idx = ui->stopBitsComboBox->findData(s.get(BC::Key::RS232::stopBits,
                                         QVariant::fromValue(Rs232Instrument::OneStop)));
    ui->stopBitsComboBox->setCurrentIndex(qBound(0,idx,ui->stopBitsComboBox->count()-1));
    ui->stopBitsComboBox->setEnabled(true);

    idx = ui->parityComboBox->findData(s.get(BC::Key::RS232::parity,
                                             QVariant::fromValue(Rs232Instrument::NoParity)));
    ui->parityComboBox->setCurrentIndex(qBound(0,idx,ui->parityComboBox->count()-1));
    ui->parityComboBox->setEnabled(true);

    idx = ui->flowControlComboBox->findData(s.get(BC::Key::RS232::flowControl,
                                                  QVariant::fromValue(Rs232Instrument::NoFlowControl)));
    ui->flowControlComboBox->setCurrentIndex(qBound(0,idx,ui->flowControlComboBox->count()-1));
    ui->flowControlComboBox->setEnabled(true);

    ui->rs232TestButton->setEnabled(true);
}

void CommunicationDialog::customDeviceChanged(int index)
{
    //remove all widgets from previous hardware
    while(!d_customInfoList.isEmpty())
    {
        auto ci = d_customInfoList.takeFirst();
        ci.labelWidget->hide();
        ci.labelWidget->deleteLater();
        ci.displayWidget->hide();
        ci.displayWidget->deleteLater();
    }

    if(index < 0)
    {
        ui->customTestButton->setEnabled(false);
        return;
    }


    auto key = ui->customDeviceComboBox->currentData().toString();

    SettingsStorage s(key,SettingsStorage::Hardware);
    using namespace BC::Key::Custom;

    auto count = s.getArraySize(comm);
    for(std::size_t i=0; i<count; ++i)
    {
        CustomInfo ci;
        ci.type = s.getArrayValue(comm,i,type,stringKey);
        ci.key = s.getArrayValue(comm,i,key,QString("key"));
        if(ci.type == intKey)
        {
            QSpinBox *sb = new QSpinBox;
            sb->setMinimum(s.getArrayValue(comm,i,intMin,-__INT_MAX__));
            sb->setMaximum(s.getArrayValue(comm,i,intMax,__INT_MAX__));
            sb->setValue(s.get(ci.key,0));
            ci.displayWidget = sb;
        }
        else
        {
            QLineEdit *le = new QLineEdit;
            le->setMaxLength(s.getArrayValue(comm,i,maxLen,255));
            le->setText(s.get(ci.key,QString("")));
            ci.displayWidget = le;
        }

        ci.labelWidget = new QLabel(s.getArrayValue(comm,i,label,QString("ID")));

        ui->customBoxLayout->insertRow(1,ci.labelWidget,ci.displayWidget);
        d_customInfoList.append(ci);
    }

    if(count > 0)
        ui->customTestButton->setEnabled(true);

}

void CommunicationDialog::testGpib()
{
	int index = ui->gpibDeviceComboBox->currentIndex();
	if(index < 0)
		return;

    auto key = ui->gpibDeviceComboBox->currentData().toString();
    SettingsStorage hwSettings(key, SettingsStorage::Hardware);
    auto subKey = hwSettings.get(BC::Key::HW::subKey, QString(""));

    //one of the few times to invoke QSettings directly: need to edit the settings
    //for the hardware object itself
    QSettings s(QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(key);
    s.beginGroup(subKey);
    s.setValue(BC::Key::GPIB::gpibAddress,ui->busAddressSpinBox->value());
    s.endGroup();
    s.endGroup();
	s.sync();

    // Save read options for GPIB protocol
    saveProtocolReadOptions(BC::Key::Comm::gpib, p_gpibTimeoutSpinBox, p_gpibTermCharEdit, key, subKey);

    startTest(BC::Key::Comm::gpib,key);
}

void CommunicationDialog::testTcp()
{
	int index = ui->tcpDeviceComboBox->currentIndex();
	if(index < 0)
		return;

    auto key = ui->tcpDeviceComboBox->currentData().toString();
    SettingsStorage hwSettings(key, SettingsStorage::Hardware);
    auto subKey = hwSettings.get(BC::Key::HW::subKey, QString(""));

    //one of the few times to invoke QSettings directly: need to edit the settings
    //for the hardware object itself
    QSettings s(QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(key);
    s.beginGroup(subKey);
    s.setValue(BC::Key::TCP::ip,ui->ipLineEdit->text());
    s.setValue(BC::Key::TCP::port,ui->portSpinBox->value());
    s.endGroup();
    s.endGroup();
	s.sync();

    // Save read options for TCP protocol
    saveProtocolReadOptions(BC::Key::Comm::tcp, p_tcpTimeoutSpinBox, p_tcpTermCharEdit, key, subKey);

    startTest(BC::Key::Comm::tcp,key);
}

void CommunicationDialog::testRs232()
{
	int index = ui->rs232DeviceComboBox->currentIndex();
	if(index < 0)
		return;

    auto key = ui->rs232DeviceComboBox->currentData().toString();
    SettingsStorage hwSettings(key, SettingsStorage::Hardware);
    auto subKey = hwSettings.get(BC::Key::HW::subKey, QString(""));

    using namespace BC::Key::RS232;
    //one of the few times to invoke QSettings directly: need to edit the settings
    //for the hardware object itself
    QSettings s(QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(key);
    s.beginGroup(subKey);
    s.setValue(BC::Key::RS232::id,ui->rs232DeviceIDLineEdit->text());
    s.setValue(BC::Key::RS232::baud,ui->baudRateComboBox->currentText().toInt());
    s.setValue(BC::Key::RS232::dataBits,ui->dataBitsComboBox->currentData().value<Rs232Instrument::DataBits>());
    s.setValue(BC::Key::RS232::stopBits,ui->stopBitsComboBox->currentData().value<Rs232Instrument::StopBits>());
    s.setValue(BC::Key::RS232::parity,ui->parityComboBox->currentData().value<Rs232Instrument::Parity>());
    s.setValue(BC::Key::RS232::flowControl,ui->flowControlComboBox->currentData().value<Rs232Instrument::FlowControl>());
    s.endGroup();
    s.endGroup();
	s.sync();

    // Save read options for RS232 protocol
    saveProtocolReadOptions(BC::Key::Comm::rs232, p_rs232TimeoutSpinBox, p_rs232TermCharEdit, key, subKey);

    startTest(BC::Key::Comm::rs232,key);
}

void CommunicationDialog::testCustom()
{
    int index = ui->customDeviceComboBox->currentIndex();
    if(index < 0)
        return;

    auto key = ui->customDeviceComboBox->currentData().toString();
    SettingsStorage hwSettings(key, SettingsStorage::Hardware);
    auto subKey = hwSettings.get(BC::Key::HW::subKey, QString(""));

    //one of the few times to invoke QSettings directly: need to edit the settings
    //for the hardware object itself
    QSettings s(QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(key);
    s.beginGroup(subKey);

    using namespace BC::Key::Custom;
    for(int i=0; i<d_customInfoList.size(); i++)
    {
        auto ci = d_customInfoList.at(i);
        if(ci.type == intKey)
        {
            auto sb = dynamic_cast<QSpinBox*>(ci.displayWidget);
            s.setValue(ci.key,sb->value());
        }
        else
        {
            auto le = dynamic_cast<QLineEdit*>(ci.displayWidget);
            s.setValue(ci.key,le->text());
        }
    }

    s.endGroup();
    s.endGroup();
    s.sync();

    // Save read options for Custom protocol
    saveProtocolReadOptions(BC::Key::Comm::custom, p_customTimeoutSpinBox, p_customTermCharEdit, key, subKey);

    startTest(BC::Key::Comm::custom,key);
}


void CommunicationDialog::testComplete(QString device, bool success, QString msg)
{
	//configure ui
	setEnabled(true);
	setCursor(QCursor());

	if(success)
		QMessageBox::information(this,QString("Connection Successful"),
							QString("%1 connected successfully!").arg(device),QMessageBox::Ok);
	else
        QMessageBox::critical(this,QString("Connection Failed"),
						  QString("%1 connection failed!\n%2").arg(device).arg(msg),QMessageBox::Ok);
}

void CommunicationDialog::setupReadOptionsUI()
{
    // Helper function to add read options to a group box
    auto addReadOptions = [](QGroupBox* box, QSpinBox*& timeoutSpinBox, QLineEdit*& termCharEdit) {
        auto layout = qobject_cast<QFormLayout*>(box->layout());
        if (!layout) return;
        
        // Timeout setting
        timeoutSpinBox = new QSpinBox;
        timeoutSpinBox->setRange(0, 60000); // 0 to 60 seconds (0 = no timeout)
        timeoutSpinBox->setSuffix(" ms");
        timeoutSpinBox->setSpecialValueText("No timeout");
        layout->addRow("Timeout:", timeoutSpinBox);
        
        // Termination character setting
        termCharEdit = new QLineEdit;
        termCharEdit->setMaxLength(10);
        termCharEdit->setPlaceholderText("Empty = no termination");
        layout->addRow("Term Char:", termCharEdit);
    };
    
    // Add read options to each protocol group box
    addReadOptions(ui->gpibBox, p_gpibTimeoutSpinBox, p_gpibTermCharEdit);
    addReadOptions(ui->tcpBox, p_tcpTimeoutSpinBox, p_tcpTermCharEdit);
    addReadOptions(ui->rs232Box, p_rs232TimeoutSpinBox, p_rs232TermCharEdit);
    addReadOptions(ui->customBox, p_customTimeoutSpinBox, p_customTermCharEdit);
}

void CommunicationDialog::loadReadOptionsFromSettings()
{
    // Helper function to load settings for a protocol
    auto loadProtocolSettings = [this](const QString& protocolKey, QSpinBox* timeoutSpinBox, QLineEdit* termCharEdit) {
        int timeout = d_storage.getArrayValue(protocolKey, 0, BC::Key::Comm::timeout, 1000);
        QString termChar = d_storage.getArrayValue(protocolKey, 0, BC::Key::Comm::termChar, QString(""));
        
        timeoutSpinBox->setValue(timeout);
        termCharEdit->setText(termChar);
    };
    
    // Load settings for each protocol
    loadProtocolSettings(BC::Key::Comm::gpib, p_gpibTimeoutSpinBox, p_gpibTermCharEdit);
    loadProtocolSettings(BC::Key::Comm::tcp, p_tcpTimeoutSpinBox, p_tcpTermCharEdit);
    loadProtocolSettings(BC::Key::Comm::rs232, p_rs232TimeoutSpinBox, p_rs232TermCharEdit);
    loadProtocolSettings(BC::Key::Comm::custom, p_customTimeoutSpinBox, p_customTermCharEdit);
}

void CommunicationDialog::saveProtocolReadOptions(const QString& protocolKey, QSpinBox* timeoutSpinBox, QLineEdit* termCharEdit, const QString& hwKey, const QString& subKey)
{
    // Save read options to the hardware object's settings using the same pattern as test methods
    QSettings s(QApplication::organizationName(), QApplication::applicationName());
    s.beginGroup(hwKey);
    s.beginGroup(subKey);
    s.beginGroup(protocolKey); // Store protocol-specific settings in a map
    s.setValue(BC::Key::Comm::timeout, timeoutSpinBox->value());
    s.setValue(BC::Key::Comm::termChar, termCharEdit->text());
    s.endGroup();
    s.endGroup();
    s.endGroup();
    s.sync();
}

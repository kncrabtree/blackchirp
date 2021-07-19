#include <gui/dialog/communicationdialog.h>
#include "ui_communicationdialog.h"

#include <QApplication>
#include <QMessageBox>

#include <hardware/core/communication/communicationprotocol.h>
#include <hardware/core/hardwaremanager.h>
#include <hardware/core/hardwareobject.h>

CommunicationDialog::CommunicationDialog(QWidget *parent) :
     QDialog(parent),
     ui(new Ui::CommunicationDialog), d_storage(BC::Key::hw)
{
	ui->setupUi(this);

	//populate GPIB devices
    auto count = d_storage.getArraySize(BC::Key::Comm::gpib);
    for(std::size_t i=0; i<count; ++i)
        ui->gpibDeviceComboBox->addItem(d_storage.getArrayValue<QString>(BC::Key::Comm::gpib,i,BC::Key::HW::name),QVariant::fromValue(i));

    if(count == 0)
		ui->gpibBox->setEnabled(false);

	//populate TCP devices
    count = d_storage.getArraySize(BC::Key::Comm::tcp);
    for(std::size_t i=9; i<count; ++i)
	{
        ui->tcpDeviceComboBox->addItem(d_storage.getArrayValue<QString>(BC::Key::Comm::tcp,i,BC::Key::HW::name),QVariant::fromValue(i));
	}

    if(count == 0)
		ui->tcpBox->setEnabled(false);

	//populate RS232 devices
    count = d_storage.getArraySize(BC::Key::Comm::rs232);
    for(std::size_t i=0; i<count; ++i)
	{
        ui->rs232DeviceComboBox->addItem(d_storage.getArrayValue<QString>(BC::Key::Comm::rs232,i,BC::Key::HW::name),QVariant::fromValue(i));
	}

    if(count==0)
        ui->rs232Box->setEnabled(false);

    //populate custom devices
    count = d_storage.getArraySize(BC::Key::Comm::custom);
    for(std::size_t i=0; i<count; ++i)
    {
        ui->customDeviceComboBox->addItem(d_storage.getArrayValue<QString>(BC::Key::Comm::custom,i,BC::Key::HW::name),QVariant::fromValue(i));
    }

    if(count == 0)
        ui->customBox->setEnabled(false);

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
    ui->dataBitsComboBox->setItemData(0,QVariant::fromValue(QSerialPort::Data5));
    ui->dataBitsComboBox->setItemData(1,QVariant::fromValue(QSerialPort::Data6));
    ui->dataBitsComboBox->setItemData(2,QVariant::fromValue(QSerialPort::Data7));
    ui->dataBitsComboBox->setItemData(3,QVariant::fromValue(QSerialPort::Data8));
    ui->dataBitsComboBox->setEnabled(false);

    ui->stopBitsComboBox->setCurrentIndex(-1);
    ui->stopBitsComboBox->setItemData(0,QVariant::fromValue(QSerialPort::OneStop));
    ui->stopBitsComboBox->setItemData(1,QVariant::fromValue(QSerialPort::OneAndHalfStop));
    ui->stopBitsComboBox->setItemData(2,QVariant::fromValue(QSerialPort::TwoStop));
    ui->stopBitsComboBox->setEnabled(false);

    ui->parityComboBox->setCurrentIndex(-1);
    ui->parityComboBox->setItemData(0,QVariant::fromValue(QSerialPort::NoParity));
    ui->parityComboBox->setItemData(1,QVariant::fromValue(QSerialPort::EvenParity));
    ui->parityComboBox->setItemData(2,QVariant::fromValue(QSerialPort::OddParity));
    ui->parityComboBox->setItemData(3,QVariant::fromValue(QSerialPort::SpaceParity));
    ui->parityComboBox->setItemData(4,QVariant::fromValue(QSerialPort::MarkParity));
    ui->parityComboBox->setEnabled(false);

    ui->flowControlComboBox->setCurrentIndex(-1);
    ui->flowControlComboBox->setItemData(0,QVariant::fromValue(QSerialPort::NoFlowControl));
    ui->flowControlComboBox->setItemData(1,QVariant::fromValue(QSerialPort::HardwareControl));
    ui->flowControlComboBox->setItemData(2,QVariant::fromValue(QSerialPort::SoftwareControl));
    ui->flowControlComboBox->setEnabled(false);

	ui->rs232TestButton->setEnabled(false);

    ui->customDeviceComboBox->setCurrentIndex(-1);
    ui->customTestButton->setEnabled(false);

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

void CommunicationDialog::gpibDeviceChanged(int index)
{
	if(index < 0)
	{
		ui->busAddressSpinBox->setValue(0);
		ui->gpibTestButton->setEnabled(false);
		ui->busAddressSpinBox->setEnabled(false);
		return;
	}

    std::size_t i = static_cast<std::size_t>(index);
    auto key = d_storage.getArrayValue<QString>(BC::Key::Comm::gpib,i,BC::Key::HW::key);

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

    std::size_t i = static_cast<std::size_t>(index);
    auto key = d_storage.getArrayValue<QString>(BC::Key::Comm::tcp,i,BC::Key::HW::key);

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

    std::size_t i = static_cast<std::size_t>(index);
    auto key = d_storage.getArrayValue<QString>(BC::Key::Comm::rs232,i,BC::Key::HW::key);

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
                                                    QVariant::fromValue(QSerialPort::Data8)));
    ui->dataBitsComboBox->setCurrentIndex(qBound(0,idx,ui->dataBitsComboBox->count()-1));
    ui->dataBitsComboBox->setEnabled(true);

    idx = ui->stopBitsComboBox->findData(s.get(BC::Key::RS232::stopBits,
                                         QVariant::fromValue(QSerialPort::OneStop)));
    ui->stopBitsComboBox->setCurrentIndex(qBound(0,idx,ui->stopBitsComboBox->count()-1));
    ui->stopBitsComboBox->setEnabled(true);

    idx = ui->parityComboBox->findData(s.get(BC::Key::RS232::parity,
                                             QVariant::fromValue(QSerialPort::NoParity)));
    ui->parityComboBox->setCurrentIndex(qBound(0,idx,ui->parityComboBox->count()-1));
    ui->parityComboBox->setEnabled(true);

    idx = ui->flowControlComboBox->findData(s.get(BC::Key::RS232::flowControl,
                                                  QVariant::fromValue(QSerialPort::NoFlowControl)));
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


    std::size_t idx = static_cast<std::size_t>(index);
    auto key = d_storage.getArrayValue<QString>(BC::Key::Comm::custom,idx,BC::Key::HW::key);

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

    auto i = static_cast<std::size_t>(index);
    auto key = d_storage.getArrayValue<QString>(BC::Key::Comm::gpib,i,BC::Key::HW::key);
    auto subKey = d_storage.getArrayValue<QString>(BC::Key::Comm::gpib,i,BC::Key::HW::subKey);

    //one of the few times to invoke QSettings directly: need to edit the settings
    //for the hardware object itself
    QSettings s(QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(key);
    s.beginGroup(subKey);
    s.setValue(BC::Key::GPIB::gpibAddress,ui->busAddressSpinBox->value());
    s.endGroup();
    s.endGroup();
	s.sync();

    startTest(BC::Key::Comm::gpib,key);
}

void CommunicationDialog::testTcp()
{
	int index = ui->tcpDeviceComboBox->currentIndex();
	if(index < 0)
		return;

    auto i = static_cast<std::size_t>(index);
    auto key = d_storage.getArrayValue<QString>(BC::Key::Comm::tcp,i,BC::Key::HW::key);
    auto subKey = d_storage.getArrayValue<QString>(BC::Key::Comm::tcp,i,BC::Key::HW::subKey);

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

    startTest(BC::Key::Comm::tcp,key);
}

void CommunicationDialog::testRs232()
{
	int index = ui->rs232DeviceComboBox->currentIndex();
	if(index < 0)
		return;

    auto i = static_cast<std::size_t>(index);
    auto key = d_storage.getArrayValue<QString>(BC::Key::Comm::rs232,i,BC::Key::HW::key);
    auto subKey = d_storage.getArrayValue<QString>(BC::Key::Comm::rs232,i,BC::Key::HW::subKey);

    using namespace BC::Key::RS232;
    //one of the few times to invoke QSettings directly: need to edit the settings
    //for the hardware object itself
    QSettings s(QApplication::organizationName(),QApplication::applicationName());
    s.beginGroup(key);
    s.beginGroup(subKey);
    s.setValue(BC::Key::RS232::id,ui->rs232DeviceIDLineEdit->text());
    s.setValue(BC::Key::RS232::baud,ui->baudRateComboBox->currentText().toInt());
    s.setValue(BC::Key::RS232::dataBits,ui->dataBitsComboBox->currentData());
    s.setValue(BC::Key::RS232::stopBits,ui->stopBitsComboBox->currentData());
    s.setValue(BC::Key::RS232::parity,ui->parityComboBox->currentData());
    s.setValue(BC::Key::RS232::flowControl,ui->flowControlComboBox->currentData());
    s.endGroup();
    s.endGroup();
	s.sync();

    startTest(BC::Key::Comm::rs232,key);
}

void CommunicationDialog::testCustom()
{
    int index = ui->customDeviceComboBox->currentIndex();
    if(index < 0)
        return;

    auto i = static_cast<std::size_t>(index);
    auto key = d_storage.getArrayValue<QString>(BC::Key::Comm::custom,i,BC::Key::HW::key);
    auto subKey = d_storage.getArrayValue<QString>(BC::Key::Comm::custom,i,BC::Key::HW::subKey);

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

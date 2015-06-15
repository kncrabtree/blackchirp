#include "communicationdialog.h"
#include "ui_communicationdialog.h"

#include <QSettings>
#include <QApplication>
#include <QMessageBox>

CommunicationDialog::CommunicationDialog(QWidget *parent) :
     QDialog(parent),
     ui(new Ui::CommunicationDialog)
{
	ui->setupUi(this);

	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());

	//populate GPIB devices
	s.beginGroup(QString("gpib"));
	int numGpib = s.beginReadArray(QString("instruments"));
	for(int i=0;i<numGpib;i++)
	{
		s.setArrayIndex(i);
		QString key = s.value(QString("key"),QString("")).toString();
        QString subKey = s.value(QString("subKey"),QString("")).toString();

        if(!key.isEmpty() && !subKey.isEmpty())
            d_gpibDevices.append(qMakePair(key,subKey));
	}
	s.endArray();
	s.endGroup();

    for(int i=0;i<d_gpibDevices.size();i++)
	{
        QString name = s.value(QString("%1/prettyName").arg(d_gpibDevices.at(i).first),d_gpibDevices.at(i).first).toString();
		ui->gpibDeviceComboBox->addItem(name);
	}

    if(d_gpibDevices.isEmpty())
		ui->gpibBox->setEnabled(false);

	//populate TCP devices
	s.beginGroup(QString("tcp"));
	int numTcp = s.beginReadArray(QString("instruments"));
	for(int i=0;i<numTcp;i++)
	{
		s.setArrayIndex(i);
		QString key = s.value(QString("key"),QString("")).toString();
        QString subKey = s.value(QString("subKey"),QString("")).toString();

        if(!key.isEmpty() && !subKey.isEmpty())
            d_tcpDevices.append(qMakePair(key,subKey));
	}
	s.endArray();
	s.endGroup();

    for(int i=0;i<d_tcpDevices.size();i++)
	{
        QString name = s.value(QString("%1/prettyName").arg(d_tcpDevices.at(i).first),d_tcpDevices.at(i).first).toString();
		ui->tcpDeviceComboBox->addItem(name);
	}

    if(d_tcpDevices.isEmpty())
		ui->tcpBox->setEnabled(false);

	//populate RS232 devices
	s.beginGroup(QString("rs232"));
	int numRs232 = s.beginReadArray(QString("instruments"));
	for(int i=0;i<numRs232;i++)
	{
		s.setArrayIndex(i);
		QString key = s.value(QString("key"),QString("")).toString();
        QString subKey = s.value(QString("subKey"),QString("")).toString();

        if(!key.isEmpty() && !subKey.isEmpty())
            d_rs232Devices.append(qMakePair(key,subKey));
	}
	s.endArray();
	s.endGroup();

    for(int i=0;i<d_rs232Devices.size();i++)
	{
        QString name = s.value(QString("%1/prettyName").arg(d_rs232Devices.at(i).first),d_rs232Devices.at(i).first).toString();
		ui->rs232DeviceComboBox->addItem(name);
	}

    if(d_rs232Devices.isEmpty())
		ui->rs232Box->setEnabled(false);

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
	ui->rs232TestButton->setEnabled(false);

    auto cbChanged = static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged);

    connect(ui->gpibDeviceComboBox,cbChanged,this,&CommunicationDialog::gpibDeviceChanged);
    connect(ui->tcpDeviceComboBox,cbChanged,this,&CommunicationDialog::tcpDeviceChanged);
    connect(ui->rs232DeviceComboBox,cbChanged,this,&CommunicationDialog::rs232DeviceChanged);

    connect(ui->gpibTestButton,&QPushButton::clicked,this,&CommunicationDialog::testGpib);
    connect(ui->tcpTestButton,&QPushButton::clicked,this,&CommunicationDialog::testTcp);
    connect(ui->rs232TestButton,&QPushButton::clicked,this,&CommunicationDialog::testRs232);
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

	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    QString key = d_gpibDevices.at(index).first;
    QString subKey = d_gpibDevices.at(index).second;

	ui->busAddressSpinBox->setEnabled(true);
    ui->busAddressSpinBox->setValue(s.value(QString("%1/%2/address").arg(key).arg(subKey),0).toInt());
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

	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    QString key = d_tcpDevices.at(index).first;
    QString subKey = d_tcpDevices.at(index).second;

	ui->ipLineEdit->setEnabled(true);
    ui->ipLineEdit->setText(s.value(QString("%1/%2/ip").arg(key).arg(subKey),QString("")).toString());
	ui->portSpinBox->setEnabled(true);
    ui->portSpinBox->setValue(s.value(QString("%1/%2/port").arg(key).arg(subKey),0).toInt());
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
		ui->rs232TestButton->setEnabled(false);
	}

	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    QString key = d_rs232Devices.at(index).first;
    QString subKey = d_rs232Devices.at(index).second;

	ui->rs232DeviceIDLineEdit->setEnabled(true);
    ui->rs232DeviceIDLineEdit->setText(s.value(QString("%1/%2/id").arg(key).arg(subKey),QString("")).toString());
    int br = s.value(QString("%1/%2/baudrate").arg(key).arg(subKey),-1).toInt();
	ui->baudRateComboBox->setEnabled(true);
	ui->baudRateComboBox->setCurrentIndex(-1);
	for(int i=0;i<ui->baudRateComboBox->count();i++)
	{
		if(br == ui->baudRateComboBox->itemText(i).toInt())
			ui->baudRateComboBox->setCurrentIndex(i);
	}

	ui->rs232TestButton->setEnabled(true);
}

void CommunicationDialog::testGpib()
{
	int index = ui->gpibDeviceComboBox->currentIndex();
	if(index < 0)
		return;

	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    QString key = d_gpibDevices.at(index).first;
    QString subKey = d_gpibDevices.at(index).second;

    s.setValue(QString("%1/%2/address").arg(key).arg(subKey),ui->busAddressSpinBox->value());
	s.sync();

	startTest(QString("gpib"),key);
}

void CommunicationDialog::testTcp()
{
	int index = ui->tcpDeviceComboBox->currentIndex();
	if(index < 0)
		return;

	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    QString key = d_tcpDevices.at(index).first;
    QString subKey = d_tcpDevices.at(index).second;

    s.setValue(QString("%1/%2/ip").arg(key).arg(subKey),ui->ipLineEdit->text());
    s.setValue(QString("%1/%2/port").arg(key).arg(subKey),ui->portSpinBox->value());
	s.sync();

	startTest(QString("tcp"),key);
}

void CommunicationDialog::testRs232()
{
	int index = ui->rs232DeviceComboBox->currentIndex();
	if(index < 0)
		return;

	QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    QString key = d_rs232Devices.at(index).first;
    QString subKey = d_rs232Devices.at(index).second;

    s.setValue(QString("%1/%2/id").arg(key).arg(subKey),ui->rs232DeviceIDLineEdit->text());
	int brIndex = ui->baudRateComboBox->currentIndex();
	if(brIndex < 0)
        s.setValue(QString("%1/%2/baudrate").arg(key).arg(subKey),0);
	else
        s.setValue(QString("%1/%2/baudrate").arg(key).arg(subKey),ui->baudRateComboBox->itemText(brIndex).toInt());
	s.sync();

	startTest(QString("rs232"),key);
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
		QMessageBox::critical(this,QString("Connection failed"),
						  QString("%1 connection failed!\n%2").arg(device).arg(msg),QMessageBox::Ok);
}

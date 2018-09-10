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

    //populate custom devices
    s.beginGroup(QString("custom"));
    int numCustom = s.beginReadArray(QString("instruments"));
    for(int i=0; i<numCustom; i++)
    {
        s.setArrayIndex(i);
        QString key = s.value(QString("key"),QString("")).toString();
        QString subKey = s.value(QString("subKey"),QString("")).toString();

        if(!key.isEmpty() && !subKey.isEmpty())
            d_customDevices.append(qMakePair(key,subKey));
    }
    s.endArray();
    s.endGroup();

    for(int i=0; i<d_customDevices.size(); i++)
    {
        QString name = s.value(QString("%1/prettyName").arg(d_customDevices.at(i).first),d_customDevices.at(i).first).toString();
        ui->customDeviceComboBox->addItem(name);
    }

    if(d_customDevices.isEmpty())
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
        return;
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

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    QSettings s2(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    QString key = d_customDevices.at(index).first;
    QString subKey = d_customDevices.at(index).second;

    s.beginGroup(key);
    s.beginGroup(subKey);
    s2.beginGroup(key);
    s2.beginGroup(subKey);
    int n = s.beginReadArray(QString("comm"));
    for(int i=0; i<n; i++)
    {
        s.setArrayIndex(i);
        CustomInfo ci;
        ci.type = s.value(QString("type"),QString("string")).toString();
        ci.key = s.value(QString("key"),QString("key")).toString();
        if(ci.type.startsWith(QString("int"),Qt::CaseInsensitive))
        {
            QSpinBox *sb = new QSpinBox;
            sb->setMinimum(s.value(QString("min"),-2147483647).toInt());
            sb->setMaximum(s.value(QString("max"),2147483647).toInt());
            sb->setValue(s2.value(ci.key,0).toInt());
            ci.displayWidget = sb;
        }
        else
        {
            QLineEdit *le = new QLineEdit;
            le->setMaxLength(s.value(QString("length"),100).toInt());
            le->setText(s2.value(ci.key,QString("")).toString());
            ci.displayWidget = le;
        }

        ci.labelWidget = new QLabel(s.value(QString("name"),QString("ID")).toString());

        ui->customBoxLayout->insertRow(1,ci.labelWidget,ci.displayWidget);
        d_customInfoList.append(ci);
    }

    if(n > 0)
        ui->customTestButton->setEnabled(true);

    s.endArray();
    s.endGroup();
    s.endGroup();
    return;


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

void CommunicationDialog::testCustom()
{
    int index = ui->customDeviceComboBox->currentIndex();
    if(index < 0)
        return;

    QSettings s(QSettings::SystemScope,QApplication::organizationName(),QApplication::applicationName());
    QString key = d_customDevices.at(index).first;
    QString subKey = d_customDevices.at(index).second;

    s.beginGroup(key);
    s.beginGroup(subKey);

    for(int i=0; i<d_customInfoList.size(); i++)
    {
        auto ci = d_customInfoList.at(i);
        if(ci.type.startsWith(QString("int")))
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

    startTest(QString("custom"),key);
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
